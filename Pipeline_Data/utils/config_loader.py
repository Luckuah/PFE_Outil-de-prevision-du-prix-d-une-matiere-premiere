"""
Configuration loader for GDELT Oil Pipeline
"""
import os
import yaml
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

class Config:
    """Configuration manager for the pipeline"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize configuration
        
        Args:
            config_path: Path to config.yaml file
        """
        # Load environment variables
        load_dotenv()
        
        # Determine project root
        self.project_root = Path(__file__).parent.parent.parent
        
        # Load YAML config
        if config_path is None:
            config_path = self.project_root / "config" / "config.yaml"
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
        
        # Add environment variables
        self.config['database']['host'] = os.getenv('DB_HOST', 'localhost')
        self.config['database']['port'] = int(os.getenv('DB_PORT', 3306))
        self.config['database']['database'] = os.getenv('DB_NAME', 'gdelt')
        self.config['database']['user'] = os.getenv('DB_USER', 'root')
        self.config['database']['password'] = os.getenv('DB_PASSWORD', '')
        
        # Add paths
        self.config['paths'] = {
            'project_root': str(self.project_root),
            'data_dir': str(self.project_root / 'data'),
            'models_dir': str(self.project_root / 'models'),
            'logs_dir': str(self.project_root / 'logs'),
            'raw_data': str(self.project_root / 'data' / 'raw'),
            'processed_data': str(self.project_root / 'data' / 'processed'),
            'faiss_index': str(self.project_root / 'data' / 'faiss_index')
        }
        
        # Create directories if they don't exist
        for path in self.config['paths'].values():
            Path(path).mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value by dot notation key
        
        Args:
            key: Configuration key (e.g., 'database.host')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_db_config(self) -> Dict[str, Any]:
        """Get database configuration"""
        return {
            'host': self.config['database']['host'],
            'port': self.config['database']['port'],
            'database': self.config['database']['database'],
            'user': self.config['database']['user'],
            'password': self.config['database']['password']
        }
    
    def get_keywords(self, category: str = 'oil') -> list:
        """
        Get all keywords for a category across all languages
        
        Args:
            category: 'oil', 'gas', or 'organizations'
            
        Returns:
            List of keywords
        """
        if category == 'organizations':
            return self.config['keywords']['organizations']
        
        keywords = []
        for lang, words in self.config['keywords'][category].items():
            keywords.extend(words)
        return keywords
    
    def __getitem__(self, key: str) -> Any:
        """Allow dict-like access"""
        return self.get(key)
    
    def __repr__(self) -> str:
        return f"Config(project_root={self.config['paths']['project_root']})"


# Global config instance
_config_instance = None

def get_config(config_path: str = None) -> Config:
    """
    Get or create global config instance
    
    Args:
        config_path: Path to config.yaml file
        
    Returns:
        Config instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = Config(config_path)
    return _config_instance