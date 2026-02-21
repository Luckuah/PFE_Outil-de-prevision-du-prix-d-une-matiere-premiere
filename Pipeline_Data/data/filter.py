import pandas as pd
import re
from typing import List, Set
from utils.logger import get_logger
from utils.config_loader import get_config

logger = get_logger(__name__)


class GDELTFilter:

    def __init__(self):

        self.config = get_config()
        #Main keywords to keep articles : oil, gas, organisations, crisis
        self.oil_keywords = self._normalize_keywords(
            self.config.get_keywords('oil')
        )
        self.gas_keywords = self._normalize_keywords(
            self.config.get_keywords('gas')
        )
        self.org_keywords = self._normalize_keywords(
            self.config.get_keywords('organizations')
        )
        
        self.all_keywords = self.oil_keywords | self.gas_keywords | self.org_keywords
        
        # Get oil producing countries
        self.oil_countries = set(
            self.config.get('filtering.oil_producing_countries', [])
        )
        
        # Get relevant event codes
        self.relevant_codes = set(
            self.config.get('filtering.relevant_event_codes', [])
        )
        
        logger.info(f" Filter initialized with {len(self.all_keywords)} keywords")
        logger.info(f"   - Oil keywords: {len(self.oil_keywords)}")
        logger.info(f"   - Gas keywords: {len(self.gas_keywords)}")
        logger.info(f"   - Organizations: {len(self.org_keywords)}")
        logger.info(f"   - Oil countries: {len(self.oil_countries)}")
    
    def _normalize_keywords(self, keywords: List[str]) -> Set[str]:

        normalized = set()
        for kw in keywords:
            # Convert to lowercase
            kw_lower = kw.lower().strip()
            normalized.add(kw_lower)
            
            # Add variations without special characters
            kw_clean = re.sub(r'[^\w\s]', '', kw_lower)
            if kw_clean != kw_lower:
                normalized.add(kw_clean)
        
        return normalized
    
    def _contains_keywords(self, text: str, keywords: Set[str]) -> tuple:

        if pd.isna(text):
            return False, []
        
        text_lower = str(text).lower()
        matched = []
        
        for keyword in keywords:
            # Use word boundaries for better matching
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, text_lower):
                matched.append(keyword)
        
        return len(matched) > 0, matched
    
    def filter_by_keywords(self, df: pd.DataFrame) -> pd.DataFrame:
        
        logger.info(f" Filtering {len(df)} events by keywords...")
        
        # Initialize column for matched keywords
        df['keyword_matches'] = None
        df['has_keywords'] = False
        
        matches_list = []
        
        for idx, row in df.iterrows():
            # Combine text fields to search
            search_text = ' '.join([
                str(row.get('Actor1Name', '')),
                str(row.get('Actor2Name', '')),
                str(row.get('SOURCEURL', '')),
                str(row.get('ActionGeo_Fullname', ''))
            ])
            
            # Check for keywords
            has_match, matched_kws = self._contains_keywords(search_text, self.all_keywords)
            
            matches_list.append({
                'has_keywords': has_match,
                'keyword_matches': ','.join(matched_kws) if matched_kws else None
            })
        
        # Update DataFrame
        matches_df = pd.DataFrame(matches_list)
        df['has_keywords'] = matches_df['has_keywords']
        df['keyword_matches'] = matches_df['keyword_matches']
        
        # Filter
        df_filtered = df[df['has_keywords'] == True].copy()
        
        logger.info(f" Kept {len(df_filtered)} events after keyword filtering ({len(df_filtered)/len(df)*100:.1f}%)")
        
        return df_filtered
    
    def filter_by_countries(self, df: pd.DataFrame) -> pd.DataFrame:
        
        logger.info(f" Filtering by oil-producing countries...")
        
        df['is_oil_country'] = False
        
        # Check if any actor or action location is in oil country
        for idx, row in df.iterrows():
            countries = [
                row.get('Actor1CountryCode'),
                row.get('Actor2CountryCode'),
                row.get('Actor1Geo_CountryCode'),
                row.get('Actor2Geo_CountryCode'),
                row.get('ActionGeo_CountryCode')
            ]
            
            # Check if any country matches
            if any(c in self.oil_countries for c in countries if pd.notna(c)):
                df.at[idx, 'is_oil_country'] = True
        
        # Filter
        df_filtered = df[df['is_oil_country'] == True].copy()
        
        logger.info(f" Kept {len(df_filtered)} events from oil countries ({len(df_filtered)/len(df)*100:.1f}%)")
        
        return df_filtered
    
    def filter_by_event_codes(self, df: pd.DataFrame) -> pd.DataFrame:
        
        if not self.relevant_codes:
            return df
        
        logger.info(f" Filtering by {len(self.relevant_codes)} relevant event codes...")
        
        # Filter by EventRootCode
        if 'EventRootCode' in df.columns:
            df_filtered = df[df['EventRootCode'].isin(self.relevant_codes)].copy()
            logger.info(f"Kept {len(df_filtered)} events with relevant codes ({len(df_filtered)/len(df)*100:.1f}%)")
            return df_filtered
        else:
            logger.warning(" EventRootCode column not found, skipping event code filtering")
            return df
    
    def apply_filters(self, df: pd.DataFrame, 
                     use_keywords: bool = True,
                     use_countries: bool = True,
                     use_event_codes: bool = False) -> pd.DataFrame:
        
        logger.info(f" Applying filters to {len(df)} events...")
        
        initial_count = len(df)
        df_filtered = df.copy()
        
        # Apply filters sequentially
        if use_keywords:
            df_filtered = self.filter_by_keywords(df_filtered)
        
        if use_countries and len(df_filtered) > 0:
            df_filtered = self.filter_by_countries(df_filtered)
        
        if use_event_codes and len(df_filtered) > 0:
            df_filtered = self.filter_by_event_codes(df_filtered)
        
        # Log final statistics
        final_count = len(df_filtered)
        retention_rate = (final_count / initial_count * 100) if initial_count > 0 else 0
        
        logger.info(f"Filtering complete:")
        logger.info(f"   - Initial events: {initial_count}")
        logger.info(f"   - Final events: {final_count}")
        logger.info(f"   - Retention rate: {retention_rate:.1f}%")
        
        return df_filtered
    
    def get_filter_stats(self, df: pd.DataFrame) -> dict:
            
        stats = {
            'total_events': len(df),
            'events_with_keywords': df['has_keywords'].sum() if 'has_keywords' in df.columns else 0,
            'events_oil_countries': df['is_oil_country'].sum() if 'is_oil_country' in df.columns else 0,
            'top_keywords': {},
            'top_countries': {},
            'top_event_codes': {}
        }
        
        # Top keywords
        if 'keyword_matches' in df.columns:
            all_keywords = []
            for kws in df['keyword_matches'].dropna():
                all_keywords.extend(kws.split(','))
            
            if all_keywords:
                from collections import Counter
                keyword_counts = Counter(all_keywords)
                stats['top_keywords'] = dict(keyword_counts.most_common(10))
        
        # Top countries
        if 'ActionGeo_CountryCode' in df.columns:
            country_counts = df['ActionGeo_CountryCode'].value_counts().head(10)
            stats['top_countries'] = country_counts.to_dict()
        
        # Top event codes
        if 'EventRootCode' in df.columns:
            code_counts = df['EventRootCode'].value_counts().head(10)
            stats['top_event_codes'] = code_counts.to_dict()
        
        return stats