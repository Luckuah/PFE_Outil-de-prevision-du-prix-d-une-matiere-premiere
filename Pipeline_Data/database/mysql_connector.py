"""
MySQL Database connector for GDELT Oil Pipeline
"""
import pymysql
from pymysql import Error
import pandas as pd
from typing import List, Dict, Any, Optional
import json
from datetime import datetime

from utils.logger import get_logger
from utils.config_loader import get_config

logger = get_logger(__name__)


class MySQLConnector:
    """MySQL database connector"""
    
    def __init__(self):
        """Initialize MySQL connector"""
        self.config = get_config()
        self.db_config = self.config.get_db_config()
        self.connection = None
        
        logger.info(f"✅ MySQL connector initialized (host={self.db_config['host']})")
    
    def get_connection(self):
        """Get database connection"""
        try:
            if self.connection is None or not self.connection.open:
                self.connection = pymysql.connect(
                    host=self.db_config['host'],
                    port=int(self.db_config.get('port', 3306)),
                    user=self.db_config['user'],
                    password=self.db_config['password'],
                    database=self.db_config['database'],
                    charset='utf8mb4',
                    autocommit=False
                )
            return self.connection
        except Exception as e:
            logger.error(f"❌ Failed to connect to database: {e}")
            raise
    
    def test_connection(self) -> bool:
        """
        Test database connection
        
        Returns:
            True if connection successful
        """
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            cursor.close()
            
            logger.info("✅ Database connection test successful")
            return True
            
        except Exception as e:
            logger.error(f"❌ Database connection test failed: {e}")
            return False
    
    def create_tables(self, schema_file: str = None):
        """
        Create database tables from schema file
        
        Args:
            schema_file: Path to schema.sql file
        """
        if schema_file is None:
            schema_file = f"{self.config['paths']['project_root']}/src/database/schema.sql"
        
        try:
            with open(schema_file, 'r', encoding='utf-8') as f:
                schema_sql = f.read()
            
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Split by delimiter and execute
            statements = schema_sql.split(';')
            for statement in statements:
                statement = statement.strip()
                if statement and not statement.startswith('--'):
                    try:
                        cursor.execute(statement)
                    except Exception as e:
                        # Ignore DROP TABLE errors
                        if 'DROP TABLE' not in statement:
                            logger.warning(f"⚠️ SQL warning: {e}")
            
            conn.commit()
            cursor.close()
            
            logger.info("✅ Database tables created successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to create tables: {e}")
            raise
    
    def insert_articles(self, df: pd.DataFrame, batch_size: int = 100) -> int:
        """
        Insert articles into database with detailed error handling
        
        Args:
            df: DataFrame with articles
            batch_size: Number of records per batch
            
        Returns:
            Number of inserted records
        """
        if df.empty:
            logger.warning("⚠️ No articles to insert")
            return 0
        
        logger.info(f"💾 Inserting {len(df)} articles into database...")
        
        insert_query = """
        INSERT INTO gdelt_articles_scored (
            global_event_id, day, month_year, year, date_added,
            actor1_name, actor1_country_code, actor2_name, actor2_country_code,
            event_code, event_root_code, quad_class, goldstein_scale,
            num_mentions, num_sources, num_articles, avg_tone,
            actor1_geo_country_code, actor2_geo_country_code,
            action_geo_country_code, action_geo_fullname,
            action_geo_lat, action_geo_long,
            source_url, article_title, article_content, article_language,
            article_author, article_publish_date,
            llm_score, llm_justification, final_score,
            is_oil_country, keyword_matches
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
        )
        ON DUPLICATE KEY UPDATE
            final_score = VALUES(final_score),
            llm_score = VALUES(llm_score),
            llm_justification = VALUES(llm_justification),
            updated_at = CURRENT_TIMESTAMP
        """
        
        inserted = 0
        failed = 0
        failed_details = []
        
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            # Process in batches
            for i in range(0, len(df), batch_size):
                batch = df.iloc[i:i+batch_size]
                
                for idx, row in batch.iterrows():
                    try:
                        # Prepare data tuple
                        data = self._prepare_row_data(row)
                        cursor.execute(insert_query, data)
                        inserted += 1
                        
                    except Exception as e:
                        failed += 1
                        error_msg = str(e)
                        event_id = row.get('GlobalEventID', 'N/A')
                        
                        # Log détaillé pour débug
                        logger.error(f"❌ Failed to insert event {event_id}")
                        logger.error(f"   Error: {error_msg}")
                        
                        # Diagnostiquer le problème
                        if "Data too long" in error_msg:
                            # Trouver quelle colonne est trop longue
                            for col in ['article_title', 'article_content', 'actor1_name', 'actor2_name']:
                                if col in row and pd.notna(row[col]):
                                    length = len(str(row[col]))
                                    logger.error(f"   {col} length: {length} chars")
                        
                        elif "Incorrect" in error_msg or "invalid" in error_msg.lower():
                            # Problème de type de données
                            logger.error(f"   Day: {row.get('Day')}, DATEADDED: {row.get('DATEADDED')}")
                        
                        elif "cannot be null" in error_msg.lower():
                            # Champ NULL non autorisé
                            logger.error(f"   Null fields detected")
                            for field in ['GlobalEventID', 'Day', 'DATEADDED']:
                                if pd.isna(row.get(field)):
                                    logger.error(f"   {field} is NULL!")
                        
                        # Sauvegarder pour rapport
                        failed_details.append({
                            'event_id': event_id,
                            'error': error_msg[:200]
                        })
                
                # Commit batch
                try:
                    conn.commit()
                except Exception as e:
                    logger.error(f"❌ Failed to commit batch: {e}")
                    conn.rollback()
                
                if (i + batch_size) % 1000 == 0:
                    logger.info(f"   Inserted {inserted} / {len(df)} articles...")
            
            cursor.close()
            
            # Rapport final
            if inserted > 0:
                logger.info(f"✅ Inserted {inserted} articles successfully")
            
            if failed > 0:
                logger.warning(f"⚠️ Failed to insert {failed} articles")
                
                # Afficher les 3 premières erreurs
                for detail in failed_details[:3]:
                    logger.warning(f"   Event {detail['event_id']}: {detail['error']}")
                
                if len(failed_details) > 3:
                    logger.warning(f"   ... and {len(failed_details) - 3} more errors")
            
            return inserted
            
        except Exception as e:
            logger.error(f"❌ Critical error during insertion: {e}")
            raise
    
    def _prepare_row_data(self, row: pd.Series) -> tuple:
        """
        Prepare row data for insertion with validation
        
        Args:
            row: DataFrame row
            
        Returns:
            Tuple of values for SQL insert
        """
        # 1. GlobalEventID (OBLIGATOIRE)
        global_event_id = row.get('GlobalEventID')
        if pd.isna(global_event_id):
            raise ValueError("GlobalEventID cannot be NULL")
        global_event_id = int(global_event_id)
        
        # 2. Convert Day
        day_str = str(row.get('Day', ''))
        if len(day_str) == 8 and day_str.isdigit():
            day = datetime.strptime(day_str, '%Y%m%d').date()
        else:
            logger.warning(f"Invalid Day format: {day_str}")
            day = None
        
        # 3. Convert DATEADDED
        dateadded_str = str(row.get('DATEADDED', ''))
        if len(dateadded_str) == 14 and dateadded_str.isdigit():
            dateadded = datetime.strptime(dateadded_str, '%Y%m%d%H%M%S')
        else:
            logger.warning(f"Invalid DATEADDED format: {dateadded_str}")
            dateadded = None
        
        return (
            # Identifiants (5)
            global_event_id,
            day,
            int(row.get('MonthYear', 0)) if pd.notna(row.get('MonthYear')) else None,
            int(row.get('Year', 0)) if pd.notna(row.get('Year')) else None,
            dateadded,
            
            # Acteurs (4)
            str(row.get('Actor1Name', ''))[:500] if pd.notna(row.get('Actor1Name')) else None,
            str(row.get('Actor1CountryCode', ''))[:3] if pd.notna(row.get('Actor1CountryCode')) else None,
            str(row.get('Actor2Name', ''))[:500] if pd.notna(row.get('Actor2Name')) else None,
            str(row.get('Actor2CountryCode', ''))[:3] if pd.notna(row.get('Actor2CountryCode')) else None,
            
            # Événement (4)
            str(row.get('EventCode', ''))[:10] if pd.notna(row.get('EventCode')) else None,
            str(row.get('EventRootCode', ''))[:2] if pd.notna(row.get('EventRootCode')) else None,
            int(row.get('QuadClass', 0)) if pd.notna(row.get('QuadClass')) else None,
            float(row.get('GoldsteinScale', 0)) if pd.notna(row.get('GoldsteinScale')) else None,
            
            # Métriques GDELT (4)
            int(row.get('NumMentions', 0)) if pd.notna(row.get('NumMentions')) else 0,
            int(row.get('NumSources', 0)) if pd.notna(row.get('NumSources')) else 0,
            int(row.get('NumArticles', 0)) if pd.notna(row.get('NumArticles')) else 0,
            float(row.get('AvgTone', 0)) if pd.notna(row.get('AvgTone')) else None,
            
            # Géographie (6)
            str(row.get('Actor1Geo_CountryCode', ''))[:3] if pd.notna(row.get('Actor1Geo_CountryCode')) else None,
            str(row.get('Actor2Geo_CountryCode', ''))[:3] if pd.notna(row.get('Actor2Geo_CountryCode')) else None,
            str(row.get('ActionGeo_CountryCode', ''))[:3] if pd.notna(row.get('ActionGeo_CountryCode')) else None,
            str(row.get('ActionGeo_Fullname', ''))[:500] if pd.notna(row.get('ActionGeo_Fullname')) else None,
            float(row.get('ActionGeo_Lat', 0)) if pd.notna(row.get('ActionGeo_Lat')) else None,
            float(row.get('ActionGeo_Long', 0)) if pd.notna(row.get('ActionGeo_Long')) else None,
            
            # Article (6)
            str(row.get('SOURCEURL', ''))[:2000] if pd.notna(row.get('SOURCEURL')) else None,  # Limiter URL
            str(row.get('article_title', ''))[:2000] if pd.notna(row.get('article_title')) else None,  # Limiter titre
            str(row.get('article_content', '')) if pd.notna(row.get('article_content')) else None,
            str(row.get('article_language', ''))[:5] if pd.notna(row.get('article_language')) else None,
            str(row.get('article_author', ''))[:255] if pd.notna(row.get('article_author')) else None,
            row.get('article_publish_date') if pd.notna(row.get('article_publish_date')) else None,
            
            # Scoring (3)
            int(row.get('llm_score', 0)) if pd.notna(row.get('llm_score')) else None,
            str(row.get('llm_justification', ''))[:2000] if pd.notna(row.get('llm_justification')) else None,  # Limiter
            float(row.get('final_score', 0)) if pd.notna(row.get('final_score')) else None,
            
            # Features (2)
            bool(row.get('is_oil_country', False)),
            str(row.get('keyword_matches', ''))[:1000] if pd.notna(row.get('keyword_matches')) else None
        )
    
    def validate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate DataFrame before insertion
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Validated DataFrame
        """
        logger.info("🔍 Validating DataFrame...")
        
        # Colonnes obligatoires
        required_cols = ['GlobalEventID', 'Day', 'DATEADDED']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Vérifier les valeurs NULL
        null_count = df[required_cols].isnull().sum()
        for col, count in null_count.items():
            if count > 0:
                logger.warning(f"⚠️ {col} has {count} NULL values")
        
        # Supprimer les lignes avec GlobalEventID NULL
        initial_len = len(df)
        df = df[df['GlobalEventID'].notna()].copy()
        removed = initial_len - len(df)
        
        if removed > 0:
            logger.warning(f"⚠️ Removed {removed} rows with NULL GlobalEventID")
        
        logger.info(f"✅ Validation complete: {len(df)} valid rows")
        
        return df
    
    def get_articles(self, limit: int = 100, 
                    min_score: float = 50,
                    order_by: str = 'final_score DESC') -> pd.DataFrame:
        """
        Get articles from database
        
        Args:
            limit: Maximum number of articles
            min_score: Minimum final_score
            order_by: ORDER BY clause
            
        Returns:
            DataFrame with articles
        """
        query = f"""
        SELECT * FROM gdelt_articles_scored
        WHERE final_score >= %s
        ORDER BY {order_by}
        LIMIT %s
        """
        
        try:
            conn = self.get_connection()
            df = pd.read_sql(query, conn, params=(min_score, limit))
            
            logger.info(f"✅ Retrieved {len(df)} articles from database")
            return df
            
        except Exception as e:
            logger.error(f"❌ Failed to get articles: {e}")
            return pd.DataFrame()
    
    def update_daily_stats(self, stats: Dict[str, Any]):
        """
        Update daily statistics
        
        Args:
            stats: Dictionary with statistics
        """
        query = """
        INSERT INTO daily_stats (
            date, total_events_downloaded, events_after_filtering,
            articles_fetched, articles_scored, articles_kept,
            avg_score, processing_time_seconds
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            total_events_downloaded = VALUES(total_events_downloaded),
            events_after_filtering = VALUES(events_after_filtering),
            articles_fetched = VALUES(articles_fetched),
            articles_scored = VALUES(articles_scored),
            articles_kept = VALUES(articles_kept),
            avg_score = VALUES(avg_score),
            processing_time_seconds = VALUES(processing_time_seconds)
        """
        
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            data = (
                stats.get('date'),
                stats.get('total_events_downloaded', 0),
                stats.get('events_after_filtering', 0),
                stats.get('articles_fetched', 0),
                stats.get('articles_scored', 0),
                stats.get('articles_kept', 0),
                stats.get('avg_score', 0.0),
                stats.get('processing_time_seconds', 0)
            )
            
            cursor.execute(query, data)
            conn.commit()
            cursor.close()
            
            logger.info(f"✅ Updated daily stats for {stats.get('date')}")
            
        except Exception as e:
            logger.error(f"❌ Failed to update daily stats: {e}")
    
    def get_stats(self, days: int = 30) -> pd.DataFrame:
        """
        Get daily statistics
        
        Args:
            days: Number of days to retrieve
            
        Returns:
            DataFrame with stats
        """
        query = """
        SELECT * FROM daily_stats
        WHERE date >= DATE_SUB(CURDATE(), INTERVAL %s DAY)
        ORDER BY date DESC
        """
        
        try:
            conn = self.get_connection()
            df = pd.read_sql(query, conn, params=(days,))
            return df
        except Exception as e:
            logger.error(f"❌ Failed to get stats: {e}")
            return pd.DataFrame()
    
    def close(self):
        """Close database connection"""
        if self.connection and self.connection.open:
            self.connection.close()
            logger.info("✅ Database connection closed")