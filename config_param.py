"""
Centralise toutes les constantes, paramètres et métadonnées du projet.
"""

from datetime import datetime
from typing import Dict, Tuple

# ==================== CONFIGURATION DES DONNÉES ====================

class DataConfig:
    """Configuration pour le téléchargement et le traitement des données."""
    
    # Tickers Yahoo Finance
    OIL_TICKER = 'BZ=F'  # Brent Crude Oil (unifié pour tout le projet)
    VIX_TICKER = '^VIX'  # Volatility Index
    
    # Dates de collecte
    START_DATE = '2010-01-01'  
    END_DATE = datetime.now().strftime('%Y-%m-%d')  # Données jusqu'à aujourd'hui
    
    # Colonnes de features (ordre important pour le modèle)
    FEATURE_COLS = [
        'Open', 'High', 'Low', 'Volume', 'VIX_Close',
        'RSI_14', 'MACD', 'ATR_14', 'ADX_14', 'VROC',
        'BB_Upper', 'BB_Mid', 'BB_Lower', 'Market_Regime'
    ]
    
    # Colonne cible
    TARGET_COL = 'Close'


# ==================== CONFIGURATION DU MODÈLE ====================

class ModelConfig:
    """Configuration pour l'entraînement et les prédictions du LSTM."""
    
    # Architecture
    LOOKBACK = 60  # Jours historiques pour prédire
    FUTURE_STEPS = 10  # Jours à prédire
    LSTM_UNITS = 128  # Unités dans la première couche LSTM
    
    # Entraînement
    EPOCHS = 50
    BATCH_SIZE = 64
    VALIDATION_SPLIT = 0.2  # 20% des données pour validation
    PATIENCE = 20  # Early stopping
    LEARNING_RATE = 0.001
    
    # Intervalles de confiance
    CONFIDENCE_LEVEL = 0.95
    
    # Sauvegarde
    MODEL_PATH = 'lstm_oil_model.h5'


# ==================== DÉFINITION DES CRISES ====================

class CrisesConfig:
    """
    Définition des périodes de crise impactant le marché pétrolier.
    Format: {nom: (date_début, date_fin)}
    """
    
    CRISES: Dict[str, Tuple[str, str]] = {
        # Crises financières
        "Crise financière mondiale": ("2007-08-01", "2009-06-30"),
        "Crise de la dette européenne": ("2010-01-01", "2012-12-31"),
        
        # Géopolitique - Moyen-Orient
        "Printemps arabe": ("2010-12-01", "2012-12-31"),
        "Guerre civile en Libye": ("2011-02-15", "2011-10-23"),
        "Sanctions Iran (cycle 1)": ("2012-01-01", "2015-07-14"),
        "Retrait JCPOA (Iran)": ("2018-05-08", "2019-12-31"),
        
        # Chocs d'offre OPEP
        "Effondrement prix OPEP+": ("2014-06-01", "2016-02-29"),
        "Guerre des prix Russie/OPEP": ("2020-03-01", "2020-04-30"),
        
        # Pandémie et conséquences
        "COVID-19 (choc demande)": ("2020-02-01", "2020-05-31"),
        "Crise énergétique post-COVID": ("2021-10-01", "2022-12-31"),
        
        # Guerre Ukraine
        "Invasion Ukraine": ("2022-02-24", "2023-12-31"),
    }
    
    @classmethod
    def get_crises_list(cls) -> list:
        """Retourne la liste des noms de crises."""
        return list(cls.CRISES.keys())
    
    @classmethod
    def get_crisis_dates(cls, crisis_name: str) -> Tuple[str, str]:
        """Retourne les dates d'une crise spécifique."""
        return cls.CRISES.get(crisis_name, (None, None))


# ==================== CONFIGURATION STREAMLIT ====================

class UIConfig:
    """Configuration de l'interface utilisateur Streamlit."""
    
    # Titre de l'application
    APP_TITLE = "🛢️ Brent Oil Price Forecasting"
    
    # Cache TTL (Time To Live) en secondes
    CACHE_TTL = 3600  # 1 heure
    
    # Pages disponibles
    PAGES = ["Dashboard", "Prédictions", "Paramètres"]
    
    # Graphiques
    PLOT_TEMPLATE = "plotly_white"
    CRISIS_COLOR = "rgba(255, 0, 0, 0.2)"  # Rouge transparent
    PREDICTION_COLOR = "#d62728"  # Rouge vif
    HISTORICAL_COLOR = "#1f77b4"  # Bleu


# ==================== EXPORT ====================

# Export simplifié pour les imports
__all__ = [
    'DataConfig',
    'ModelConfig',
    'CrisesConfig',
    'UIConfig'
]