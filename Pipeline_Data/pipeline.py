"""
Main Pipeline for GDELT Oil Price Prediction
"""
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional
import time

from data.gdelt_fetcher import GDELTFetcher
from data.filter import GDELTFilter
from data.article_fetcher import ArticleFetcher
from models.trieur_agent import TrieurAgent
from models.rag_agent import RAGAgent
from Pipeline_Data.database.mysql_connector2 import MySQLConnector
from utils.logger import get_logger, PipelineLogger, setup_logger
from utils.config_loader import get_config

logger = get_logger(__name__)


class GDELTPipeline:
    """Complete pipeline for GDELT data processing"""
    
    def __init__(self):
        """Initialize pipeline components"""
        self.config = get_config()
        
        # Setup logger with file output
        log_file = f"{self.config.get('paths.logs_dir')}/pipeline.log"
        setup_logger('gdelt_pipeline', log_file=log_file)
        
        logger.info("🚀 Initializing GDELT Oil Pipeline...")
        
        # Initialize components
        self.gdelt_fetcher = GDELTFetcher()
        self.filter = GDELTFilter()
        self.article_fetcher = ArticleFetcher(timeout=10, max_retries=2)
        self.trieur = None  # Lazy load (heavy model)
        self.rag = None  # Lazy load
        self.db = MySQLConnector()
        
        logger.info("✅ Pipeline initialized successfully")
    
    def _load_trieur(self):
        """Lazy load Trieur agent"""
        if self.trieur is None:
            self.trieur = TrieurAgent()
        return self.trieur
    
    def _load_rag(self):
        """Lazy load RAG agent"""
        if self.rag is None:
            self.rag = RAGAgent()
        return self.rag
    
    def run_single_day(self, date: str, 
                      fetch_articles: bool = True,
                      score_articles: bool = True,
                      save_to_db: bool = True) -> pd.DataFrame:
        """
        Run pipeline for a single day
        
        Args:
            date: Date in format 'YYYY-MM-DD'
            fetch_articles: Whether to fetch full articles
            score_articles: Whether to score with LLM
            save_to_db: Whether to save to database
            
        Returns:
            DataFrame with processed articles
        """
        start_time = datetime.now()
        stats = {'date': date}
        
        logger.info(f"\n{'='*80}")
        logger.info(f"📅 Processing date: {date}")
        logger.info(f"{'='*80}\n")
        
        # Step 1: Fetch GDELT data
        with PipelineLogger(logger, f"Step 1: Fetch GDELT data for {date}"):
            df_events = self.gdelt_fetcher.fetch_day(date, coverage=True)
            stats['total_events_downloaded'] = len(df_events)
            
            if df_events.empty:
                logger.warning(f"⚠️ No events found for {date}")
                return pd.DataFrame()
        
        # Step 2: Apply filters
        with PipelineLogger(logger, "Step 2: Apply filters"):
            df_filtered = self.filter.apply_filters(
                df_events,
                use_keywords=True,
                use_countries=True,
                use_event_codes=False
            )
            stats['events_after_filtering'] = len(df_filtered)
            
            if df_filtered.empty:
                logger.warning(f"⚠️ No events passed filters for {date}")
                self._save_stats(stats, start_time)
                return pd.DataFrame()
        
        # Step 3: Fetch full articles (optional)
        if fetch_articles:
            with PipelineLogger(logger, "Step 3: Fetch full articles"):
                df_with_articles = self.article_fetcher.fetch_articles_batch(
                    df_filtered,
                    url_column='SOURCEURL',
                    delay=0.5,
                    max_articles=5  # No limit
                )
                stats['articles_fetched'] = len(df_with_articles)
                
                # Filter by language
                languages = self.config.get('filtering.languages', ['en', 'fr', 'de', 'es'])
                df_with_articles = self.article_fetcher.filter_by_language(
                    df_with_articles,
                    languages=languages
                )
                
                # Clean content
                df_with_articles = self.article_fetcher.clean_content(df_with_articles)
        else:
            df_with_articles = df_filtered
            stats['articles_fetched'] = 0
        
        if df_with_articles.empty:
            logger.warning(f"⚠️ No articles fetched for {date}")
            self._save_stats(stats, start_time)
            return pd.DataFrame()
        
        # Step 4: Score articles with LLM (optional)
        if score_articles:
            with PipelineLogger(logger, "Step 4: Score articles with LLM"):
                trieur = self._load_trieur()
                
                # Score
                df_scored = trieur.score_articles_batch(df_with_articles)
                print("--------------------------------------------------------------------")
                print (df_scored)
                time.sleep(2)
                # Compute final score
                #FONCTION DE MERDE CAR SCORE ARTICLE BATCH LE FAIT DEJA LE FINAL SCORE ENCULER
                #df_scored = trieur.compute_final_score(df_scored)
                
                # Filter top articles
                df_top = trieur.filter_top_articles(df_scored)
                
                stats['articles_scored'] = len(df_scored)
                stats['articles_kept'] = len(df_top)
                stats['avg_score'] = df_top['final_score'].mean() if len(df_top) > 0 else 0
        else:
            df_top = df_with_articles
            stats['articles_scored'] = 0
            stats['articles_kept'] = len(df_top)
            stats['avg_score'] = 0
        
        # Step 5: Save to database (optional)
        if save_to_db and not df_top.empty:
            with PipelineLogger(logger, "Step 5: Save to database"):
                inserted = self.db.insert_articles(df_top, batch_size=100)
                logger.info(f"✅ Inserted {inserted} articles into database")
        
        # Save stats
        self._save_stats(stats, start_time)
        
        # Summary
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"\n{'='*80}")
        logger.info(f"✅ Pipeline completed for {date}")
        logger.info(f"   Duration: {duration:.1f}s")
        logger.info(f"   Events downloaded: {stats['total_events_downloaded']}")
        logger.info(f"   Events after filtering: {stats['events_after_filtering']}")
        logger.info(f"   Articles fetched: {stats['articles_fetched']}")
        logger.info(f"   Articles scored: {stats['articles_scored']}")
        logger.info(f"   Articles kept: {stats['articles_kept']}")
        logger.info(f"   Average score: {stats['avg_score']:.2f}")
        logger.info(f"{'='*80}\n")
        
        return df_top
    
    def run_date_range(self, start_date: str, end_date: str,
                      delay_between_days: int = 5) -> pd.DataFrame:
        """
        Run pipeline for a date range
        
        Args:
            start_date: Start date 'YYYY-MM-DD'
            end_date: End date 'YYYY-MM-DD'
            delay_between_days: Delay in seconds between days
            
        Returns:
            Combined DataFrame with all articles
        """
        logger.info(f"\n{'#'*80}")
        logger.info(f"🚀 Starting pipeline for date range: {start_date} to {end_date}")
        logger.info(f"{'#'*80}\n")
        
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        all_results = []
        current = start
        
        while current <= end:
            date_str = current.strftime('%Y-%m-%d')
            
            # Run pipeline for day
            df_day = self.run_single_day(date_str)
            
            if not df_day.empty:
                all_results.append(df_day)
            
            # Move to next day
            current += timedelta(days=1)
            
            # Delay between days
            if current <= end:
                logger.info(f"⏳ Waiting {delay_between_days}s before next day...\n")
                time.sleep(delay_between_days)
        
        # Combine all results
        if all_results:
            df_combined = pd.concat(all_results, ignore_index=True)
            
            logger.info(f"\n{'#'*80}")
            logger.info(f"✅ Date range processing complete")
            logger.info(f"   Total articles: {len(df_combined)}")
            logger.info(f"   Date range: {start_date} to {end_date}")
            logger.info(f"{'#'*80}\n")
            
            return df_combined
        else:
            logger.warning("⚠️ No articles collected for the date range")
            return pd.DataFrame()
    
    def run_backfill(self, start_date: str = None, end_date: str = None):
        """
        Run backfill for December 2024 + January 2025
        
        Args:
            start_date: Override start date (default from config)
            end_date: Override end date (default from config)
        """
        if start_date is None:
            start_date = self.config.get('gdelt.start_date', '2024-12-01')
        if end_date is None:
            end_date = self.config.get('gdelt.end_date', '2025-01-31')
        
        logger.info(f"🔙 Starting BACKFILL for {start_date} to {end_date}")
        
        df_all = self.run_date_range(start_date, end_date, delay_between_days=2)
        
        # Update FAISS index
        if not df_all.empty:
            logger.info("🔄 Updating FAISS index with backfill data...")
            rag = self._load_rag()
            rag.rebuild_from_database(min_score=50, limit=10000)
        
        return df_all
    
    def run_daily_update(self):
        """Run daily update for yesterday's data"""
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        logger.info(f"📅 Running daily update for {yesterday}")
        
        df = self.run_single_day(yesterday)
        
        # Update FAISS index
        if not df.empty:
            logger.info("🔄 Updating FAISS index with new data...")
            rag = self._load_rag()
            
            # Load existing index or rebuild
            if not rag.load_index():
                logger.warning("⚠️ No existing index, rebuilding from database...")
                rag.rebuild_from_database()
            else:
                # Update with new articles
                rag.update_index(df)
        
        return df
    
    def _save_stats(self, stats: dict, start_time: datetime):
        """
        Save statistics to database
        
        Args:
            stats: Statistics dictionary
            start_time: Pipeline start time
        """
        try:
            duration = (datetime.now() - start_time).total_seconds()
            stats['processing_time_seconds'] = int(duration)
            
            self.db.update_daily_stats(stats)
        except Exception as e:
            logger.error(f"❌ Failed to save stats: {e}")
    
    def query_rag(self, query: str, k: int = 5) -> pd.DataFrame:
        """
        Query the RAG system
        
        Args:
            query: User query
            k: Number of results
            
        Returns:
            DataFrame with results
        """
        rag = self._load_rag()
        
        # Load index if not loaded
        if rag.index is None:
            if not rag.load_index():
                logger.warning("⚠️ No index found, rebuilding from database...")
                rag.rebuild_from_database()
        
        # Query
        results = rag.query(query, k=k)
        
        # Display
        rag.display_results(results)
        
        return results


# Standalone functions for easy usage
def run_pipeline_for_date(date: str) -> pd.DataFrame:
    """Run pipeline for a single date"""
    pipeline = GDELTPipeline()
    return pipeline.run_single_day(date)


def run_backfill(start_date: str = '2024-12-01', end_date: str = '2025-01-31') -> pd.DataFrame:
    """Run backfill for date range"""
    pipeline = GDELTPipeline()
    return pipeline.run_backfill(start_date, end_date)


def run_daily() -> pd.DataFrame:
    """Run daily update"""
    pipeline = GDELTPipeline()
    return pipeline.run_daily_update()


def query(query_text: str, k: int = 5) -> pd.DataFrame:
    """Query RAG system"""
    pipeline = GDELTPipeline()
    return pipeline.query_rag(query_text, k=k)