"""
Module unifié pour le téléchargement, le nettoyage et la préparation des données.
"""

import pandas as pd
import yfinance as yf
import ta
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.trend import ADXIndicator, MACD
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from config_param import DataConfig


# ==================== TÉLÉCHARGEMENT ====================

def download_oil_and_vix(
    start_date: str = DataConfig.START_DATE,
    end_date: str = DataConfig.END_DATE,
    oil_ticker: str = DataConfig.OIL_TICKER,
    vix_ticker: str = DataConfig.VIX_TICKER
) -> pd.DataFrame:
    
    print(f"📊 Téléchargement des données...")
    
    try:
        # 1. Télécharger le pétrole Brent
        oil_df = yf.download(
            oil_ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True # Changé à True pour éviter les problèmes de colonnes
        )
        
        if oil_df.empty:
            raise RuntimeError(f"Aucune donnée récupérée pour {oil_ticker}.")
        
        # 2. Télécharger le VIX
        vix_df = yf.download(
            vix_ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True
        )
        
        # Correction pour les versions récentes de yfinance qui créent des colonnes MultiIndex
        if isinstance(oil_df.columns, pd.MultiIndex):
            oil_df.columns = oil_df.columns.get_level_values(0)
        if isinstance(vix_df.columns, pd.MultiIndex):
            vix_df.columns = vix_df.columns.get_level_values(0)

        if vix_df.empty:
            oil_df['VIX_Close'] = 15.0
        else:
            vix_close = vix_df[['Close']].rename(columns={'Close': 'VIX_Close'})
            oil_df = oil_df.join(vix_close, how='left')
            oil_df['VIX_Close'] = oil_df['VIX_Close'].ffill().bfill()
        
        oil_df = oil_df.ffill().dropna()
        
        # Utilisation de .item() ou conversion explicite pour éviter l'erreur de formatage
        min_p = float(oil_df['Close'].min())
        max_p = float(oil_df['Close'].max())
        print(f"✅ Données chargées: {len(oil_df)} jours")
        print(f"   Prix: ${min_p:.2f} - ${max_p:.2f}")
        
        return oil_df
    
    except Exception as e:
        # On convertit explicitement e en chaîne de caractères pour éviter l'erreur de formatage
        error_msg = str(e)
        raise RuntimeError(f"Erreur lors du téléchargement des données: {error_msg}")
# ==================== INDICATEURS TECHNIQUES ====================

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    
    #Ajoute les indicateurs techniques au DataFrame.
    
    print("\n📊 Calcul des indicateurs techniques...")
    
    # Conversion en séries 1D pour compatibilité ta.py
    high = df['High'].squeeze()
    low = df['Low'].squeeze()
    close = df['Close'].squeeze()
    volume = df['Volume'].squeeze()
    
    # RSI - Relative Strength Index (momentum)
    rsi = RSIIndicator(close=close, window=14)
    df['RSI_14'] = rsi.rsi()
    
    # MACD - Moving Average Convergence Divergence (tendance)
    macd = MACD(close=close, window_fast=12, window_slow=26, window_sign=9)
    df['MACD'] = macd.macd()
    
    # ATR - Average True Range (volatilité)
    atr = AverageTrueRange(high=high, low=low, close=close, window=14)
    df['ATR_14'] = atr.average_true_range()
    
    # Bollinger Bands (support/résistance dynamique)
    bb = BollingerBands(close=close, window=20, window_dev=2)
    df['BB_Upper'] = bb.bollinger_hband()
    df['BB_Mid'] = bb.bollinger_mavg()
    df['BB_Lower'] = bb.bollinger_lband()
    
    # ADX - Average Directional Index (force de la tendance)
    adx = ADXIndicator(high=high, low=low, close=close, window=14)
    df['ADX_14'] = adx.adx()
    
    # VROC - Volume Rate of Change (momentum du volume)
    df['VROC'] = ta.volume.volume_price_trend(close=close, volume=volume)
    
    print("✅ Indicateurs techniques ajoutés")
    
    return df


# ==================== RÉGIMES DE MARCHÉ ====================

def add_market_regime(
    df: pd.DataFrame,
    crises_dict: Optional[Dict[str, Tuple[str, str]]] = None
) -> pd.DataFrame:
    """
    Ajoute la colonne Market_Regime basée sur les périodes de crise.
    
    Logique:
    - 1 : Régime normal (bull/neutral)
    - -1 : Régime de crise (bear/haute volatilité)
    """
    print("\n📍 Ajout des régimes de marché...")
    
    # Initialisation à 1 (régime normal)
    df['Market_Regime'] = 1
    
    if not crises_dict:
        print("   ℹ️  Aucune crise définie - régime normal partout")
        return df
    
    # Application des périodes de crise
    crises_appliquees = 0
    for nom, (debut, fin) in crises_dict.items():
        try:
            date_debut = pd.to_datetime(debut)
            date_fin = pd.to_datetime(fin)
            
            # Validation: début < fin
            if date_debut >= date_fin:
                print(f"   ⚠️  Crise '{nom}' ignorée: début >= fin")
                continue
            
            # Application du masque
            mask = (df.index >= date_debut) & (df.index <= date_fin)
            nb_jours = mask.sum()
            
            if nb_jours > 0:
                df.loc[mask, 'Market_Regime'] = -1
                crises_appliquees += 1
                print(f"   ✓ {nom}: {nb_jours} jours marqués")
            else:
                print(f"   ⚠️  Crise '{nom}' hors période des données")
        
        except Exception as e:
            print(f"   ⚠️  Erreur pour '{nom}': {e}")
    
    print(f"✅ {crises_appliquees}/{len(crises_dict)} crises appliquées")
    
    return df


# ==================== PIPELINE DE PRÉPARATION ====================

def prepare_full_dataset(
    crises_dict: Optional[Dict[str, Tuple[str, str]]] = None,
    start_date: str = DataConfig.START_DATE,
    end_date: str = DataConfig.END_DATE
) -> pd.DataFrame:
    """
    Pipeline complet de préparation des données.
    
    Étapes:
    1. Téléchargement Brent + VIX
    2. Calcul des indicateurs techniques
    3. Ajout des régimes de marché
    4. Nettoyage final
    
    Args:
        crises_dict: Dictionnaire des crises à marquer
        start_date: Date de début
        end_date: Date de fin
    
    Returns:
        DataFrame nettoyé et prêt pour l'entraînement
    """
    # Téléchargement
    df = download_oil_and_vix(start_date, end_date)
    
    # Indicateurs
    df = add_technical_indicators(df)
    
    # Régimes
    df = add_market_regime(df, crises_dict)
    
    # Nettoyage final: supprimer les lignes avec NaN
    nb_avant = len(df)
    df = df.dropna()
    nb_apres = len(df)
    
    if nb_avant != nb_apres:
        print(f"\n🧹 Nettoyage: {nb_avant - nb_apres} lignes avec NaN supprimées")
    
    print(f"\n✅ Dataset final: {len(df)} jours")
    print(f"   Features disponibles: {len(df.columns)} colonnes")
    
    return df


# ==================== FONCTION LEGACY (RÉTROCOMPATIBILITÉ) ====================

def creer_dataframe_brent_avec_plages(ticker: str = DataConfig.OIL_TICKER) -> pd.DataFrame:
    """
    Fonction de compatibilité.
    Utilise le nouveau pipeline mais retourne le format attendu.
    
    Args:
        ticker: Ticker du pétrole (ignoré, utilise config)
    
    Returns:
        DataFrame avec colonnes [Close, Is_Crisis]
    """
    df = download_oil_and_vix()
    df = add_technical_indicators(df)
    df = add_market_regime(df)  # Pas de crises par défaut
    df = df.dropna()
    
    # Format legacy: renommer Market_Regime en Is_Crisis
    df_legacy = df[['Close']].copy()
    df_legacy['Is_Crisis'] = (df['Market_Regime'] == -1).astype(int)
    df_legacy.columns = ['Brent_Price', 'Is_Crisis']
    
    return df_legacy


def create_crisis(df: pd.DataFrame, crises_actives: Dict[str, Tuple]) -> pd.DataFrame:
    """
    Fonction de compatibilité - redirige vers add_market_regime.
    
    DEPRECATED: Utilisez add_market_regime() directement.
    """
    warnings.warn(
        "create_crisis() est dépréciée. Utilisez add_market_regime().",
        DeprecationWarning
    )
    
    # Convertir Is_Crisis en Market_Regime si nécessaire
    if 'Is_Crisis' in df.columns and 'Market_Regime' not in df.columns:
        df['Market_Regime'] = df['Is_Crisis'].apply(lambda x: -1 if x == 1 else 1)
    
    return add_market_regime(df, crises_actives)