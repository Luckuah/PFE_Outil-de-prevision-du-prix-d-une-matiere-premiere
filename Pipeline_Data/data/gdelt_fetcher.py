"""
GDELT Data Fetcher using gdelt Python library
"""
import gdelt
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional
import time
from tqdm import tqdm

from ..utils.logger import get_logger
from ..utils.config_loader import get_config

logger = get_logger(__name__)


class GDELTFetcher:
    """Fetcher for GDELT data using gdelt library"""
    
    def __init__(self):
        """Initialize GDELT fetcher"""
        self.gd = gdelt.gdelt(version=2)  # GDELT 2.0
        self.config = get_config()
        
        # Get columns to extract from config
        self.columns = self.config.get('gdelt.columns', [])
        
        logger.info("✅ GDELT Fetcher initialized (version 2.0)")
    
    def fetch_day(self, date: str, coverage: bool = True) -> pd.DataFrame:
        """
        Fetch GDELT data for a specific day
        
        Args:
            date: Date in format 'YYYY-MM-DD' or 'YYYYMMDD'
            coverage: If True, gets full day coverage. If False, gets only last update
            
        Returns:
            DataFrame with GDELT events
        """
        try:
            # Normalize date format
            if '-' in date:
                date_obj = datetime.strptime(date, '%Y-%m-%d')
            else:
                date_obj = datetime.strptime(date, '%Y%m%d')
            
            date_str = date_obj.strftime('%Y %m %d')
            
            logger.info(f"📥 Fetching GDELT data for {date_str}")
            
            # Fetch data
            if coverage:
                # Get full day coverage
                df = self.gd.Search(date_str, table='events', coverage=True)
            else:
                # Get last update only
                df = self.gd.Search(date_str, table='events', coverage=False)
            
            if df is None or df.empty:
                logger.warning(f"⚠️ No data found for {date_str}")
                return pd.DataFrame()
            
            # Select only needed columns if they exist
            available_cols = [col for col in self.columns if col in df.columns]
            if available_cols:
                df = df[available_cols]
            
            logger.info(f"✅ Fetched {len(df)} events for {date_str}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error fetching data for {date}: {e}")
            return pd.DataFrame()
    
    def fetch_date_range(self, start_date: str, end_date: str, 
                        delay: int = 2, save_raw: bool = True) -> pd.DataFrame:
        """
        Fetch GDELT data for a date range
        
        Args:
            start_date: Start date in format 'YYYY-MM-DD'
            end_date: End date in format 'YYYY-MM-DD'
            delay: Delay in seconds between requests (to avoid rate limiting)
            save_raw: If True, save raw data to disk
            
        Returns:
            Combined DataFrame with all events
        """
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        all_data = []
        current = start
        
        # Create progress bar
        total_days = (end - start).days + 1
        pbar = tqdm(total=total_days, desc="Fetching GDELT data")
        
        while current <= end:
            date_str = current.strftime('%Y-%m-%d')
            
            # Fetch data for the day
            df_day = self.fetch_day(date_str, coverage=True)
            
            if not df_day.empty:
                # Add date column if not present
                if 'Day' not in df_day.columns:
                    df_day['Day'] = int(current.strftime('%Y%m%d'))
                
                all_data.append(df_day)
                
                # Save raw data if requested
                if save_raw:
                    self._save_raw_data(df_day, date_str)
            
            # Update progress
            pbar.update(1)
            pbar.set_postfix({'date': date_str, 'events': len(df_day)})
            
            # Move to next day
            current += timedelta(days=1)
            
            # Delay to avoid rate limiting
            if current <= end:
                time.sleep(delay)
        
        pbar.close()
        
        # Combine all data
        if all_data:
            df_combined = pd.concat(all_data, ignore_index=True)
            logger.info(f"✅ Total events fetched: {len(df_combined)}")
            return df_combined
        else:
            logger.warning("⚠️ No data fetched for the date range")
            return pd.DataFrame()
    
    def fetch_recent(self, hours: int = 24) -> pd.DataFrame:
        """
        Fetch recent GDELT data (last N hours)
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            DataFrame with recent events
        """
        # Calculate date range
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=hours)
        
        # Fetch data
        return self.fetch_date_range(
            start_date.strftime('%Y-%m-%d'),
            end_date.strftime('%Y-%m-%d'),
            delay=1
        )
    
    def _save_raw_data(self, df: pd.DataFrame, date_str: str):
        """
        Save raw GDELT data to disk
        
        Args:
            df: DataFrame to save
            date_str: Date string for filename
        """
        try:
            raw_dir = self.config.get('paths.raw_data')
            filename = f"gdelt_{date_str.replace('-', '')}.csv"
            filepath = f"{raw_dir}/{filename}"
            
            df.to_csv(filepath, index=False, encoding='utf-8')
            logger.debug(f"💾 Saved raw data to {filepath}")
            
        except Exception as e:
            logger.error(f"❌ Error saving raw data: {e}")
    
    def load_raw_data(self, date_str: str) -> pd.DataFrame:
        """
        Load raw GDELT data from disk
        
        Args:
            date_str: Date string (YYYY-MM-DD or YYYYMMDD)
            
        Returns:
            DataFrame with raw data
        """
        try:
            raw_dir = self.config.get('paths.raw_data')
            date_clean = date_str.replace('-', '')
            filename = f"gdelt_{date_clean}.csv"
            filepath = f"{raw_dir}/{filename}"
            
            df = pd.read_csv(filepath, encoding='utf-8')
            logger.info(f"✅ Loaded {len(df)} events from {filepath}")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading raw data: {e}")
            return pd.DataFrame()


# Standalone functions for quick usage
def fetch_day(date: str) -> pd.DataFrame:
    """Quick function to fetch one day of data"""
    fetcher = GDELTFetcher()
    return fetcher.fetch_day(date)


def fetch_range(start_date: str, end_date: str) -> pd.DataFrame:
    """Quick function to fetch date range"""
    fetcher = GDELTFetcher()
    return fetcher.fetch_date_range(start_date, end_date)