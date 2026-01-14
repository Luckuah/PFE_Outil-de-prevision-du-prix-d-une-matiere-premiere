import yfinance as yf
import pandas as pd
from datetime import datetime
from config import SYMBOL

# ============================================
# FONCTIONS DE RÉCUPÉRATION DES DONNÉES
# ============================================

def fetch_yahoo_data_15min(days: int = 60) -> pd.DataFrame:
    """
    Récupère les données 15min de Yahoo Finance.
    Note: Yahoo limite les données intraday à ~60 jours max pour 15min.
    """
    ticker = yf.Ticker(SYMBOL)
    # Pour 15min, Yahoo permet max 60 jours
    df = ticker.history(period=f"{min(days, 60)}d", interval="15m")
    df['timeframe'] = '15min'
    return df

def fetch_yahoo_data_4h(days: int = 60) -> pd.DataFrame:
    """
    Récupère les données 4h de Yahoo Finance.
    Note: Yahoo n'a pas d'interval 4h natif, on resample depuis 1h.
    """
    ticker = yf.Ticker(SYMBOL)
    # Récupérer en 1h puis resample en 4h
    df = ticker.history(period=f"{days}d", interval="1h")
    
    if df.empty:
        return pd.DataFrame()
    
    # Resample en 4h
    df_4h = df.resample('4h').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum'
    }).dropna()
    df_4h['timeframe'] = '4h'
    return df_4h

def fetch_yahoo_data_daily(days: int = 90) -> pd.DataFrame:
    """
    Récupère les données daily de Yahoo Finance.
    """
    ticker = yf.Ticker(SYMBOL)
    df = ticker.history(period=f"{days}d", interval="1d")
    df['timeframe'] = '1D'
    return df

def aggregate_market_data() -> dict:
    """
    Agrège les données de tous les timeframes.
    """
    print("📊 Récupération des données de marché...")
    
    try:
        df_15min = fetch_yahoo_data_15min(60)
        df_4h = fetch_yahoo_data_4h(60)
        df_daily = fetch_yahoo_data_daily(90)
        
        # Statistiques résumées pour chaque timeframe
        def summarize_df(df: pd.DataFrame, name: str) -> dict:
            if df.empty:
                return {"timeframe": name, "error": "No data available"}
            
            recent = df.tail(20)  # 20 dernières périodes
            return {
                "timeframe": name,
                "latest_close": float(df['Close'].iloc[-1]) if not df.empty else None,
                "latest_date": str(df.index[-1]) if not df.empty else None,
                "period_high": float(recent['High'].max()),
                "period_low": float(recent['Low'].min()),
                "period_avg": float(recent['Close'].mean()),
                "trend": "UP" if recent['Close'].iloc[-1] > recent['Close'].iloc[0] else "DOWN",
                "volatility": float(recent['Close'].std()),
                "total_records": len(df),
                "recent_closes": recent['Close'].tail(5).tolist()
            }
        
        return {
            "symbol": SYMBOL,
            "fetch_time": datetime.now().isoformat(),
            "timeframes": {
                "15min": summarize_df(df_15min, "15min"),
                "4h": summarize_df(df_4h, "4h"),
                "daily": summarize_df(df_daily, "daily")
            }
        }
    except Exception as e:
        print(f"❌ Erreur lors de la récupération: {e}")
        return {"error": str(e)}