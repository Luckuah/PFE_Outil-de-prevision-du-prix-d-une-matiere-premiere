import pandas as pd
import yfinance as yf

# ==========================
# 1. Fonctions utilitaires
# ==========================

def to_timestamp(date_str):
    """
    Convertit une date (str) en Timestamp pandas.
    Accepte plusieurs formats (YYYY-MM-DD, DD/MM/YYYY, etc.).
    """
    return pd.to_datetime(date_str, dayfirst=False, errors='raise')

def create_crisis (df,crises_actives):
    # 3. On applique le masque pour chaque période
    for nom, (debut, fin) in crises_actives.items():
        # On crée le masque pour la période donnée
        # On s'assure que debut et fin sont au format datetime
        mask = (df.index >= pd.to_datetime(debut)) & (df.index <= pd.to_datetime(fin))
        
        # On assigne 1 là où le masque est vrai
        df.loc[mask, "Is_Crisis"] = 1

    return df

def get_brent_data(ticker="BZ=F"):
    """
    Récupère les données de prix du Brent sur Yahoo Finance (ticker BZ=F)
    entre user_start et user_end (strings).

    - Si user_start ou user_end n'existent pas dans la série de prix,
      on prend le jour de bourse suivant le plus proche.
    - Retourne : (df, adjusted_start, adjusted_end)
      df : DataFrame avec Date en index et colonne 'Brent_Price'
    """
    start = to_timestamp("1900-01-01")
    end = to_timestamp("2100-12-31")

    # On ajoute une marge de quelques jours pour être sûr de récupérer les jours de bourse
    raw = yf.download(
        ticker,
        start=start,
        end=end,
        auto_adjust=False,
        progress=False,
    )

    if raw.empty:
        raise RuntimeError(
            f"Aucune donnée récupérée pour {ticker}. Vérifie la connexion internet ou le ticker."
        )

    # On ne garde qu'une colonne (Adj Close de préférence)
    if "Adj Close" in raw.columns:
        df = raw[["Adj Close"]].rename(columns={"Adj Close": "Brent_Price"})
    else:
        df = raw[["Close"]].rename(columns={"Close": "Brent_Price"})


    return df




# ==========================
# 3. Fonction principale
# ==========================

def creer_dataframe_brent_avec_plages(ticker="BZ=F"):
    """
    Fonction interactive :
      - Affiche les bornes disponibles des données Brent
      - Demande les dates de début / fin de la série Brent à l'utilisateur
      - Récupère les données
      - Demande le nombre de plages à marquer
      - Demande les dates pour chaque plage
      - Ajoute une colonne 0/1 correspondant aux plages
      - Retourne le DataFrame final
    """

    df= get_brent_data(ticker=ticker)
    df["Is_Crisis"] = 0

    return df


# Exemple d'utilisation
if __name__ == "__main__":
    df_resultat = creer_dataframe_brent_avec_plages()
