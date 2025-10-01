import yfinance as yf
import pandas as pd

def download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Télécharge les données depuis Yahoo Finance."""
    data = yf.download(ticker, start=start, end=end)
    return data