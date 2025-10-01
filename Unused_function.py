import pandas as pd

def add_indicators(data: pd.DataFrame, ma_windows=[30]) -> pd.DataFrame:
    """Ajoute moyenne mobile(s) et rendement journalier sur le Close."""
    for window in ma_windows:
        data[f"MA{window}"] = data["Close"].rolling(window).mean()
    data["Returns"] = data["Close"].pct_change()
    return data

def add_ema(data: pd.DataFrame, ema_windows=[9], close_col="Close") -> pd.DataFrame:
    """
    Ajoute des colonnes EMA pour différentes fenêtres de temps.

    Args:
        data (pd.DataFrame): DataFrame contenant les prix.
        ema_windows (list): Liste des périodes pour calculer les EMA (default [9,21,50]).
        close_col (str): Nom de la colonne de clôture (default "Close").

    Returns:
        pd.DataFrame: DataFrame avec les EMA ajoutées.
    """
    for window in ema_windows:
        data[f"EMA{window}"] = data[close_col].ewm(span=window, adjust=False).mean()
    return data