import ta
import pandas as pd

def add_indicators_ta(df: pd.DataFrame, close_col="Close", ma_windows=[30], ema_windows=[9,21,50]):
    """
    Ajoute MA, EMA et Rendements journaliers avec la librairie 'ta'.
    """
    # S'assurer que la colonne est 1D
    close = df[close_col]
    if isinstance(close, pd.DataFrame):
        close = close.iloc[:,0]  # prend la première colonne si c'est un DataFrame
    
    # Moyennes mobiles simples
    for window in ma_windows:
        df[f"MA{window}"] = ta.trend.SMAIndicator(close, window=window).sma_indicator()
    
    # EMA
    for window in ema_windows:
        df[f"EMA{window}"] = ta.trend.EMAIndicator(close, window=window).ema_indicator()
    
    # Rendement journalier
    df["Returns"] = close.pct_change()
    
    return df