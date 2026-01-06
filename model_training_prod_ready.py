"""
================================================================================
LSTM OIL PRICE FORECASTING MODEL
Backend pour application Streamlit
================================================================================
Ce module contient toute la logique d'entraînement et de prédiction du modèle 
LSTM pour la prévision du prix du pétrole avec intervalles de confiance.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import yfinance as yf
import ta
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.trend import ADXIndicator, MACD
import warnings
warnings.filterwarnings('ignore')


# ==================== CONFIG ====================
CONFIG = {
    'OIL_TICKER': 'CL=F',
    'VIX_TICKER': '^VIX',
    'START_DATE': '2020-01-01',
    'END_DATE': '2025-01-31',
    'LOOKBACK': 60,
    'FUTURE_STEPS': 10,
    'LSTM_UNITS': 128,
    'EPOCHS': 100,
    'BATCH_SIZE': 32,
    'VALIDATION_SPLIT': 0.2,
    'PATIENCE': 20,
    'CONFIDENCE_LEVEL': 0.95,
}


# ==================== SECTION 1: TÉLÉCHARGEMENT ET PRÉPARATION DES DONNÉES ====================

def download_data(start_date, end_date, oil_ticker, vix_ticker):
    """
    Télécharge les données historiques du pétrole et du VIX depuis Yahoo Finance.
    
    Args:
        start_date (str): Date de début au format 'YYYY-MM-DD'
        end_date (str): Date de fin au format 'YYYY-MM-DD'
        oil_ticker (str): Ticker du pétrole (ex: 'CL=F')
        vix_ticker (str): Ticker du VIX (ex: '^VIX')
    
    Returns:
        pd.DataFrame: DataFrame contenant OHLCV du pétrole + VIX Close
    """
    print("📊 Téléchargement des données...")
    
    # Télécharger prix du pétrole brut
    oil_df = yf.download(oil_ticker, start=start_date, end=end_date, progress=False)
    
    # Télécharger indice VIX (volatilité du marché)
    vix_df = yf.download(vix_ticker, start=start_date, end=end_date, progress=False)
    
    # Merger les deux datasets sur les dates
    vix_close = vix_df[['Close']].rename(columns={'Close': 'VIX_Close'})
    df = oil_df.join(vix_close, how='left')
    
    # Remplir les NaN avec forward fill (dernière valeur connue)
    df = df.fillna(method='ffill').dropna()
    
    print(f"✅ Données chargées: {len(df)} jours")
    return df


def add_technical_indicators(df):
    """
    Ajoute les indicateurs techniques principaux au DataFrame.
    Ces indicateurs capturent les patterns et la volatilité court terme.
    
    Indicateurs utilisés:
    - RSI_14: Momentum (overbought/oversold)
    - MACD: Divergence prix/tendance
    - ATR_14: Volatilité
    - Bollinger Bands: Support/Résistance
    - ADX_14: Force de la tendance
    - VROC: Momentum du volume
    
    Args:
        df (pd.DataFrame): DataFrame avec OHLCV
    
    Returns:
        pd.DataFrame: DataFrame avec indicateurs ajoutés
    """
    print("\n📊 Calcul des indicateurs techniques (TA)...")
    
    # Extraction des séries (conversion en 1D pour ta.py)
    high = df['High'].squeeze()
    low = df['Low'].squeeze()
    close = df['Close'].squeeze()
    volume = df['Volume'].squeeze()
    
    # RSI - Relative Strength Index (mesure le momentum)
    rsi = RSIIndicator(close=close, window=14)
    df['RSI_14'] = rsi.rsi()
    
    # MACD - Moving Average Convergence Divergence (tendance)
    macd = MACD(close=close, window_fast=12, window_slow=26, window_sign=9)
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    
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
    
    # Volume Price Trend (momentum du volume)
    df['VROC'] = ta.volume.volume_price_trend(close=close, volume=volume)
    
    # Return quotidien (changement % du prix)
    df['Price_Return'] = df['Close'].pct_change()
    
    print("✅ Indicateurs techniques ajoutés")
    return df





def add_market_regime(df, crises_dict=None):
    """
    Ajoute la colonne Market_Regime basée sur les crises fournies.
    Si crises_dict est None, utilise une logique par défaut ou 0.
    """
    print("\n📍 Mapping des régimes de marché dynamiques...")
    
    # On initialise tout à 1 (Bull/Normal) par défaut
    df['Market_Regime'] = 1 
    
    if crises_dict:
        for nom, (debut, fin) in crises_dict.items():
            # On passe en régime de crise (-1) pour toutes les plages sélectionnées
            mask = (df.index >= pd.to_datetime(debut)) & (df.index <= pd.to_datetime(fin))
            df.loc[mask, 'Market_Regime'] = -1
            
    print(f"✅ Régime de marché mappé avec {len(crises_dict) if crises_dict else 0} zones de crise.")
    return df


def prepare_data(df, feature_cols, target_col='Close'):
    """
    Prépare et normalise les données pour l'entraînement du LSTM.
    
    Args:
        df (pd.DataFrame): DataFrame nettoyé
        feature_cols (list): Colonnes à utiliser comme features
        target_col (str): Colonne cible (prix)
    
    Returns:
        tuple: (X_scaled, y_scaled, scaler_X, scaler_y)
    """
    print("\n🔄 Préparation et normalisation des données...")
    
    # Extraction des données
    X_data = df[feature_cols].values
    y_data = df[target_col].values.reshape(-1, 1)
    
    print(f"   Features: {len(feature_cols)}")
    print(f"   Samples: {X_data.shape[0]}")
    
    # Normalisation MinMax (ramène toutes les valeurs entre 0 et 1)
    # Cela aide le LSTM à converger plus rapidement
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler_X.fit_transform(X_data)
    
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    y_scaled = scaler_y.fit_transform(y_data)
    
    print("✅ Données normalisées [0, 1]")
    return X_scaled, y_scaled, scaler_X, scaler_y


# ==================== SECTION 2: CRÉATION DES SÉQUENCES ====================

def create_sequences(X, y, lookback):
    """
    Crée des séquences temporelles pour le LSTM.
    
    Le LSTM apprend en regardant des fenêtres de 'lookback' jours précédents
    pour prédire le prix du jour suivant.
    
    Exemple avec lookback=3:
        [jour1, jour2, jour3] → jour4
        [jour2, jour3, jour4] → jour5
        etc.
    
    Args:
        X (np.array): Features normalisées
        y (np.array): Target normalisé
        lookback (int): Nombre de jours historiques à regarder
    
    Returns:
        tuple: (X_seq, y_seq) - séquences prêtes pour le LSTM
    """
    X_seq, y_seq = [], []
    
    for i in range(lookback, len(X)):
        # Prendre 'lookback' jours de features
        X_seq.append(X[i-lookback:i])
        # Prédire le prix du jour i
        y_seq.append(y[i, 0])
    
    return np.array(X_seq), np.array(y_seq)


def prepare_sequences(X_scaled, y_scaled, lookback, test_split):
    """
    Crée les séquences et les divise en train/test.
    
    Args:
        X_scaled (np.array): Features normalisées
        y_scaled (np.array): Target normalisé
        lookback (int): Fenêtre temporelle
        test_split (float): % des données pour le test
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    print("\n📊 Création des séquences...")
    
    # Créer les séquences
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, lookback)
    
    print(f"   Séquences créées: {X_seq.shape}")
    
    # Split train/test (on garde l'ordre temporel!)
    split_idx = int(len(X_seq) * (1 - test_split))
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
    
    print(f"   Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test


# ==================== SECTION 3: CONSTRUCTION ET ENTRAÎNEMENT DU LSTM ====================

def build_lstm_model(lookback, n_features, lstm_units=128):
    """
    Construit l'architecture du modèle LSTM.
    
    Architecture:
    - LSTM 128 (couche 1): Capture les patterns complexes
    - Dropout 30%: Réduit l'overfitting
    - LSTM 64 (couche 2): Affine les patterns
    - Dropout 30%
    - Dense 32 + Dense 16: Traitement supplémentaire
    - Dense 1: Output (prix prédit)
    
    Args:
        lookback (int): Nombre de timesteps d'entrée
        n_features (int): Nombre de features
        lstm_units (int): Nombre d'unités LSTM dans la première couche
    
    Returns:
        tf.keras.Model: Modèle compilé
    """
    print("\n🧠 Construction du modèle LSTM...")
    
    model = Sequential([
        Input(shape=(lookback, n_features)),
        
        # Couche LSTM 1: Capture les dépendances long terme
        LSTM(lstm_units, activation='relu', return_sequences=True),
        Dropout(0.3),  # Éteint 30% des neurones aléatoirement
        
        # Couche LSTM 2: Affine les patterns détectés
        LSTM(64, activation='relu', return_sequences=False),
        Dropout(0.3),
        
        # Couches Dense: Traitement final
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)  # Output: 1 prix prédit
    ])
    
    # Compilation du modèle
    model.compile(
        optimizer=Adam(learning_rate=0.001),  # Optimiseur
        loss='mse',  # Loss function (Mean Squared Error)
        metrics=['mae']  # Métrique à surveiller
    )
    
    print(model.summary())
    return model


def train_model(model, X_train, y_train, epochs, batch_size, patience):
    """
    Entraîne le modèle LSTM.
    
    Args:
        model: Modèle LSTM compilé
        X_train (np.array): Données d'entraînement
        y_train (np.array): Target d'entraînement
        epochs (int): Nombre d'epochs
        batch_size (int): Taille du batch
        patience (int): Patience du early stopping
    
    Returns:
        history: Historique d'entraînement
    """
    print("\n⚙️ Entraînement du modèle...")
    
    # Early stopping: arrête l'entraînement si val_loss ne s'améliore pas
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True
    )
    
    # Entraîner le modèle
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.15,  # 15% des données train pour validation
        callbacks=[early_stop],
        verbose=1
    )
    
    print("✅ Entraînement terminé")
    return history


# ==================== SECTION 4: PRÉDICTIONS ET INTERVALLES ====================

def make_predictions(model, X_train, X_test, y_train, y_test, scaler_y):
    """
    Fait des prédictions et les dénormalise.
    
    Args:
        model: Modèle LSTM entraîné
        X_train, X_test: Données
        y_train, y_test: Targets
        scaler_y: Scaler pour dénormalisation
    
    Returns:
        dict: Prédictions pour train et test (actuelles et dénormalisées)
    """
    print("\n📈 Prédictions...")
    
    # Prédictions en valeurs normalisées
    y_train_pred = model.predict(X_train, verbose=0)
    y_test_pred = model.predict(X_test, verbose=0)
    
    # Dénormalisation (retrouver les vrais prix)
    y_train_actual = scaler_y.inverse_transform(y_train.reshape(-1, 1))
    y_train_pred_actual = scaler_y.inverse_transform(y_train_pred)
    y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1))
    y_test_pred_actual = scaler_y.inverse_transform(y_test_pred)
    
    # Calculer les métriques
    train_rmse = np.sqrt(mean_squared_error(y_train_actual, y_train_pred_actual))
    test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred_actual))
    train_mae = mean_absolute_error(y_train_actual, y_train_pred_actual)
    test_mae = mean_absolute_error(y_test_actual, y_test_pred_actual)
    test_r2 = r2_score(y_test_actual, y_test_pred_actual)
    
    print(f"\n📊 MÉTRIQUES:")
    print(f"Train - RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}")
    print(f"Test  - RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")
    
    return {
        'y_train_actual': y_train_actual,
        'y_train_pred': y_train_pred_actual,
        'y_test_actual': y_test_actual,
        'y_test_pred': y_test_pred_actual,
        'metrics': {
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'test_r2': test_r2
        }
    }


def calculate_confidence_intervals(y_actual, y_pred, confidence_level=0.95):
    """
    Calcule les intervalles de confiance basés sur l'écart-type des résidus.
    
    Logique:
    - Calculer les erreurs (résidus) du modèle
    - Mesurer l'écart-type de ces erreurs
    - Pour 95% de confiance, utiliser ±1.96 * std
    
    Args:
        y_actual (np.array): Valeurs réelles
        y_pred (np.array): Valeurs prédites
        confidence_level (float): Niveau de confiance (default 0.95 = 95%)
    
    Returns:
        tuple: (lower_bound, upper_bound, std, coverage)
    """
    # Calculer les résidus (erreurs)
    residuals = y_actual - y_pred
    std = np.std(residuals)
    
    # Z-score pour le niveau de confiance
    z_score = 1.96 if confidence_level == 0.95 else 1.645
    
    # Créer les bornes
    lower = y_pred - z_score * std
    upper = y_pred + z_score * std
    
    # Calculer la couverture (% de vrais prix dans l'intervalle)
    coverage = np.mean((y_actual >= lower) & (y_actual <= upper))
    
    return lower, upper, std, coverage


def forecast_future(model, last_sequence, steps, n_features, scaler_y, std_val, lookback):
    """
    Prédit les prix futurs jour par jour.
    
    Processus:
    1. Prédire jour +1 avec les 60 derniers jours
    2. Ajouter cette prédiction à la séquence
    3. Utiliser les 59 jours précédents + la prédiction pour jour +2
    4. Répéter
    
    Args:
        model: Modèle LSTM entraîné
        last_sequence (np.array): 60 derniers jours normalisés
        steps (int): Nombre de jours à prédire
        n_features (int): Nombre de features
        scaler_y: Scaler pour dénormalisation
        std_val (float): Écart-type pour les intervalles
        lookback (int): Fenêtre temporelle
    
    Returns:
        tuple: (predictions, lower_bounds, upper_bounds)
    """
    predictions = []
    current_seq = last_sequence.copy()
    
    for _ in range(steps):
        # Prédire le prix normalisé
        pred_scaled = model.predict(
            current_seq.reshape(1, lookback, n_features),
            verbose=0
        )[0][0]
        
        # Dénormaliser
        pred_actual = scaler_y.inverse_transform([[pred_scaled]])[0][0]
        predictions.append(pred_actual)
        
        # Mettre à jour la séquence pour la prédiction suivante
        # Supprimer le premier jour, ajouter la nouvelle prédiction
        current_seq = np.vstack([current_seq[1:], current_seq[-1:]])
    
    predictions = np.array(predictions)
    lower = predictions - 1.96 * std_val
    upper = predictions + 1.96 * std_val
    
    return predictions, lower, upper


# ==================== SECTION 5: PIPELINE PRINCIPAL ====================

def train_full_pipeline(config=CONFIG,crises_dict=None):
    """
    Exécute le pipeline complet d'entraînement.
    À utiliser pour entraîner le modèle une fois.
    
    Args:
        config (dict): Configuration avec tous les paramètres
    
    Returns:
        dict: Résultats complets (modèle, données, prédictions, etc.)
    """
    # Télécharger et préparer les données
    df = download_data(
        config['START_DATE'],
        config['END_DATE'],
        config['OIL_TICKER'],
        config['VIX_TICKER']
    )
    
    df = add_technical_indicators(df)
    df = add_market_regime(df,crises_dict)
    df = df.dropna()
    
    print(f"\n✅ Données nettoyées: {len(df)} jours")
    print(f"   Prix: ${df['Close'].values.min():.2f} - ${df['Close'].values.max():.2f}")
    
    # Préparer les features
    feature_cols = ['Open', 'High', 'Low', 'Volume', 'VIX_Close', 
                    'RSI_14', 'MACD', 'ATR_14', 'ADX_14', 'VROC',
                    'BB_Upper', 'BB_Mid', 'BB_Lower', 'Market_Regime']
    
    X_scaled, y_scaled, scaler_X, scaler_y = prepare_data(df, feature_cols)
    
    # Créer les séquences
    X_train, X_test, y_train, y_test = prepare_sequences(
        X_scaled, y_scaled,
        config['LOOKBACK'],
        config['VALIDATION_SPLIT']
    )
    
    # Construire et entraîner le modèle
    model = build_lstm_model(
        config['LOOKBACK'],
        len(feature_cols),
        config['LSTM_UNITS']
    )
    
    history = train_model(
        model, X_train, y_train,
        config['EPOCHS'],
        config['BATCH_SIZE'],
        config['PATIENCE']
    )
    
    # Faire les prédictions
    preds = make_predictions(model, X_train, X_test, y_train, y_test, scaler_y)
    
    # Calculer les intervalles
    train_lower, train_upper, train_std, _ = calculate_confidence_intervals(
        preds['y_train_actual'], preds['y_train_pred'], config['CONFIDENCE_LEVEL']
    )
    test_lower, test_upper, test_std, coverage = calculate_confidence_intervals(
        preds['y_test_actual'], preds['y_test_pred'], config['CONFIDENCE_LEVEL']
    )
    
    # Prédictions futures
    last_sequence = X_scaled[-config['LOOKBACK']:]
    future_pred, future_lower, future_upper = forecast_future(
        model, last_sequence, config['FUTURE_STEPS'], len(feature_cols),
        scaler_y, test_std, config['LOOKBACK']
    )
    
    print(f"\n📊 INTERVALLES DE CONFIANCE:")
    print(f"Test Std: ${test_std:.4f}")
    print(f"Coverage: {coverage*100:.2f}%")
    
    print(f"\n🔮 PRÉDICTIONS FUTURES ({config['FUTURE_STEPS']} jours):")
    for i, (pred, low, high) in enumerate(zip(future_pred, future_lower, future_upper), 1):
        print(f"  +{i}j: ${pred:.2f} [${low:.2f}, ${high:.2f}]")
    
    # Sauvegarder le modèle
    model.save('lstm_oil_model.h5')
    print("\n💾 Modèle sauvegardé: 'lstm_oil_model.h5'")
    
    return {
        'model': model,
        'df': df,
        'history': history,
        'scaler_X': scaler_X,
        'scaler_y': scaler_y,
        'predictions': preds,
        'intervals': {
            'train': (train_lower, train_upper),
            'test': (test_lower, test_upper),
            'future': (future_lower, future_upper),
            'std': test_std,
            'coverage': coverage
        },
        'future_predictions': future_pred,
        'feature_cols': feature_cols,
        'config': config
    }


def load_and_predict(model_path='lstm_oil_model.h5'):
    """
    Charge un modèle sauvegardé et fait des prédictions futures.
    À utiliser en production/Streamlit pour faire des prédictions rapides.
    
    Args:
        model_path (str): Chemin du modèle sauvegardé
    
    Returns:
        dict: Nouvelles prédictions
    """
    print("🔄 Chargement du modèle...")
    model = load_model(model_path, compile=False)
    
    # Télécharger les dernières données
    df = download_data(
        CONFIG['START_DATE'],
        CONFIG['END_DATE'],
        CONFIG['OIL_TICKER'],
        CONFIG['VIX_TICKER']
    )
    
    df = add_technical_indicators(df)
    df = add_market_regime(df)
    df = df.dropna()
    
    print("✅ Modèle chargé et données récentes récupérées")
    
    return {'model': model, 'df': df}


if __name__ == "__main__":
    """
    Point d'entrée principal - à exécuter une seule fois pour entraîner le modèle.
    """
    results = train_full_pipeline(CONFIG)
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLET EXÉCUTÉ")
    print("="*60)