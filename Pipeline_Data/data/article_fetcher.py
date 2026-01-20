
import pandas as pd
import requests
from newspaper import Article
from bs4 import BeautifulSoup
from langdetect import detect, LangDetectException
from typing import Optional, Dict
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from utils.logger import get_logger

logger = get_logger(__name__)


class ArticleFetcher:
    """Fetches and extracts article content from URLs"""
    
    def __init__(self, timeout: int = 10, max_retries: int = 2):
        self.timeout = timeout
        self.max_retries = max_retries
        
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Accept-Encoding': 'gzip, deflate',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        logger.info(f"Article Fetcher initialized (timeout={timeout}s)")
    
    def fetch_article(self, url: str) -> Dict[str, any]:
        result = {
            'title': None,
            'content': None,
            'language': None,
            'author': None,
            'publish_date': None,
            'success': False,
            'error': None
        }
        
        if pd.isna(url) or not url:
            result['error'] = 'Empty URL'
            return result
        # Try with newspaper3k first
        try:
            article = Article(url, language='en')
            article.download()
            article.parse()
            
            result['title'] = article.title
            result['content'] = article.text
            result['author'] = ', '.join(article.authors) if article.authors else None
            result['publish_date'] = article.publish_date
            
            # Detect language
            if result['content']:
                try:
                    result['language'] = detect(result['content'][:500])
                except LangDetectException:
                    result['language'] = 'unknown'
            
            result['success'] = True
            return result
        except Exception as e:
            logger.debug(f"newspaper3k failed for {url[:50]}...: {e}")
        
        try:
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'lxml')
            title_tag = soup.find('title')
            if title_tag:
                result['title'] = title_tag.get_text().strip()
            
            # Extract main content (try multiple selectors)
            content_selectors = [
                'article',
                {'class': 'article-content'},
                {'class': 'article-body'},
                {'class': 'post-content'},
                {'class': 'entry-content'},
                {'id': 'article-body'},
                'main'
            ]
            
            for selector in content_selectors:
                if isinstance(selector, str):
                    content_tag = soup.find(selector)
                else:
                    content_tag = soup.find('div', selector)
                
                if content_tag:
                    # Get all paragraphs
                    paragraphs = content_tag.find_all('p')
                    if paragraphs:
                        result['content'] = '\n\n'.join([p.get_text().strip() for p in paragraphs])
                        break
            
            # If no content found, try all paragraphs
            if not result['content']:
                paragraphs = soup.find_all('p')
                if len(paragraphs) > 3:  # Minimum 3 paragraphs
                    result['content'] = '\n\n'.join([p.get_text().strip() for p in paragraphs])
            
            # Detect language
            if result['content']:
                try:
                    result['language'] = detect(result['content'][:500])
                except LangDetectException:
                    result['language'] = 'unknown'
            
            result['success'] = result['content'] is not None
            return result
            
        except requests.Timeout:
            result['error'] = 'Timeout'
        except requests.RequestException as e:
            result['error'] = f'Request error: {str(e)[:50]}'
        except Exception as e:
            result['error'] = f'Parse error: {str(e)[:50]}'
        
        return result
    
    
    def fetch_articles_batch(self, df: pd.DataFrame, 
                         url_column: str = 'SOURCEURL',
                         delay: float = 0.5,
                         max_articles: int = None) -> pd.DataFrame:
        
    
        logger.info(f" Fetching articles for {len(df)} events...")
        
        # Create a working copy
        df_work = df.copy()
        
        
        if max_articles and len(df_work) > max_articles:
            df_work = df_work.head(max_articles)
            logger.info(f" Limited to {max_articles} articles for testing")
        df_work = df_work.reset_index(drop=True)
        
        # Collect results with ALL original columns preserved
        results = []
        success_count = 0
        failed_count = 0
        
        # Progress bar
        pbar = tqdm(total=len(df_work), desc="Fetching articles")
        
        for idx, row in df_work.iterrows():
            url = row.get(url_column, '')
            
            # Skip if URL is missing
            if not url or pd.isna(url):
                failed_count += 1
                pbar.update(1)
                pbar.set_postfix({
                    'success': success_count,
                    'failed': failed_count,
                    'rate': f"{success_count/(success_count+failed_count)*100:.1f}%" if (success_count+failed_count) > 0 else "0.0%"
                })
                continue
            
            # Fetch article
            article_data = self.fetch_article(url)
            
            # If successful, combine GDELT data + article data
            if article_data.get('success', False):
                # Start with ALL original GDELT columns
                combined_row = row.to_dict()
                
                # Add article columns
                combined_row['article_title'] = article_data.get('title', '')
                combined_row['article_content'] = article_data.get('content', '')
                combined_row['article_language'] = article_data.get('language', '')
                combined_row['article_author'] = article_data.get('author', '')
                combined_row['article_publish_date'] = article_data.get('publish_date', None)
                combined_row['fetch_success'] = True
                combined_row['content_length'] = len(article_data.get('content', ''))
                
                results.append(combined_row)
                success_count += 1
            else:
                failed_count += 1
                logger.debug(f"Failed to fetch: {url[:50]}... - {article_data.get('error', 'Unknown error')}")
            
            # Update progress
            pbar.update(1)
            pbar.set_postfix({
                'success': success_count,
                'failed': failed_count,
                'rate': f"{success_count/(success_count+failed_count)*100:.1f}%" if (success_count+failed_count) > 0 else "0.0%"
            })
            
            # Delay to avoid rate limiting
            time.sleep(delay)
        
        pbar.close()
        
        # Create DataFrame from results
        if not results:
            logger.warning("⚠️ No articles fetched successfully")
            return pd.DataFrame()
        
        df_success = pd.DataFrame(results)
        
        # Verify critical GDELT columns are preserved
        critical_cols = ['GlobalEventID', 'Day', 'DATEADDED']
        missing_cols = [col for col in critical_cols if col not in df_success.columns]
        
        if missing_cols:
            logger.error(f" CRITICAL: Missing columns after fetch: {missing_cols}")
            logger.error(f"   Available columns: {df_success.columns.tolist()}")
            
            # Try to diagnose
            logger.error(f"   Original df columns: {df.columns.tolist()[:20]}")
            logger.error(f"   Sample original row keys: {list(df.iloc[0].to_dict().keys())[:20] if len(df) > 0 else 'N/A'}")
        else:
            logger.info(f"All critical GDELT columns preserved")
            
            # Verify no NULLs in critical columns
            for col in critical_cols:
                null_count = df_success[col].isnull().sum()
                if null_count > 0:
                    logger.warning(f"⚠️ Column {col} has {null_count} NULL values")
        
        logger.info(f"Successfully fetched {success_count} articles ({success_count/len(df_work)*100:.1f}%)")
        logger.info(f"Failed to fetch {failed_count} articles ({failed_count/len(df_work)*100:.1f}%)")
        
        return df_success
    


    
    
    def filter_by_language(self, df: pd.DataFrame, 
                          languages: list = ['en', 'fr', 'de', 'es']) -> pd.DataFrame:
       
        if 'article_language' not in df.columns:
            logger.warning("⚠️ No article_language column found")
            return df
        
        logger.info(f" Filtering articles by languages: {languages}")
        
        initial_count = len(df)
        df_filtered = df[df['article_language'].isin(languages)].copy()
        final_count = len(df_filtered)
        
        logger.info(f" Kept {final_count}/{initial_count} articles ({final_count/initial_count*100:.1f}%)")
        
        # Language distribution
        lang_dist = df_filtered['article_language'].value_counts()
        logger.info(f"   Language distribution:")
        for lang, count in lang_dist.items():
            logger.info(f"   - {lang}: {count} ({count/final_count*100:.1f}%)")
        
        return df_filtered
    
    def clean_content(self, df: pd.DataFrame) -> pd.DataFrame:
        
        logger.info(" Cleaning article content...")
        
        initial_count = len(df)
        
        # Remove articles with no content
        df = df[df['article_content'].notna()].copy()
        
        # Remove very short articles (< 100 characters)
        df['content_length'] = df['article_content'].str.len()
        df = df[df['content_length'] >= 100].copy()
        
        # Remove duplicates based on content
        df = df.drop_duplicates(subset=['article_content'], keep='first')
        
        final_count = len(df)
        removed = initial_count - final_count
        
        logger.info(f"✅ Removed {removed} articles ({removed/initial_count*100:.1f}%)")
        logger.info(f"   - Final count: {final_count}")
        
        return df