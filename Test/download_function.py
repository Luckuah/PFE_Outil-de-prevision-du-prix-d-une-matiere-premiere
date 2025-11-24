import yfinance as yf
import pandas as pd
import json

def download_data(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Télécharge les données depuis Yahoo Finance."""
    data = yf.download(ticker, start=start, end=end,interval="1d")
    return data

def dowload_all_data():
    """
    Lit un fichier JSON contenant des informations de téléchargement 
    et récupère les données pour chaque ticker.

    Exemple de JSON :
    [
        {"ticker": "AAPL", "start": "2020-01-01", "end": "2023-01-01"},
        {"ticker": "MSFT", "start": "2019-01-01", "end": "2023-01-01"}
    ]
    """
    json_path="ticker_info.json"
    with open(json_path, "r", encoding="utf-8") as f:
        tickers_info = json.load(f)

    results = {}
    for info in tickers_info:
        ticker = info["ticker"]
        start = info["start"]
        end = info["end"]
        print(f"Téléchargement de {ticker} de {start} à {end}...")
        results[ticker] = download_data(ticker, start, end)

    return results

import json
import os

def add_ticker_to_json(ticker: str, start: str, end: str):
    """
    Crée ou met à jour un fichier JSON contenant des tickers et leurs périodes.

    Le JSON a ce format :
    [
        {"ticker": "AAPL", "start": "2020-01-01", "end": "2023-01-01"},
        {"ticker": "MSFT", "start": "2019-01-01", "end": "2023-01-01"}
    ]
    """
    json_path="ticker_info.json"
    
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []
    for entry in data:
        if entry["ticker"] == ticker:
            print(f"⚠️ Le ticker {ticker} existe déjà. Mise à jour des dates.")
            entry["start"] = start
            entry["end"] = end
            break
    else:
        data.append({"ticker": ticker, "start": start, "end": end})
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"✅ Ticker {ticker} ajouté ou mis à jour avec succès dans {json_path}")


def remove_ticker_from_json(ticker: str):
    """
    Supprime un ticker spécifique du fichier JSON s’il existe.

    Le JSON a ce format :
    [
        {"ticker": "AAPL", "start": "2020-01-01", "end": "2023-01-01"},
        {"ticker": "MSFT", "start": "2019-01-01", "end": "2023-01-01"}
    ]
    """
    json_path="ticker_info.json"
    if not os.path.exists(json_path):
        print(f"⚠️ Le fichier {json_path} n’existe pas.")
        return
    with open(json_path, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            print("⚠️ Fichier JSON vide ou corrompu.")
            return
            
    new_data = [entry for entry in data if entry["ticker"] != ticker]

    if len(new_data) == len(data):
        print(f"ℹ️ Aucun ticker '{ticker}' trouvé dans {json_path}.")
        return

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=4, ensure_ascii=False)

    print(f"✅ Ticker '{ticker}' supprimé avec succès de {json_path}.")

    