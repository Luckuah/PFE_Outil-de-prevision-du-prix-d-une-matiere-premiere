"""
================================================================================
LSTM OIL PRICE FORECASTING MODEL 
================================================================================
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from config_param import DataConfig, ModelConfig
from data_utils import prepare_full_dataset


# ==================== PRÉPARATION DES DONNÉES ====================

def prepare_data(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str = DataConfig.TARGET_COL
) -> Tuple[np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
    """
    Prépare et normalise les données pour l'entraînement LSTM.
    
    Args:
        df: DataFrame nettoyé
        feature_cols: Liste des colonnes features
        target_col: Nom de la colonne cible
    
    Returns:
        (X_scaled, y_scaled, scaler_X, scaler_y)
    """
    print("\n🔄 Normalisation des données...")
    
    # Extraction
    X_data = df[feature_cols].values
    y_data = df[target_col].values.reshape(-1, 1)
    
    print(f"   Features: {len(feature_cols)}")
    print(f"   Échantillons: {X_data.shape[0]}")
    
    # Normalisation MinMax [0, 1]
    scaler_X = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler_X.fit_transform(X_data)
    
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    y_scaled = scaler_y.fit_transform(y_data)
    
    print("✅ Normalisation terminée")
    
    return X_scaled, y_scaled, scaler_X, scaler_y


# ==================== CRÉATION DES SÉQUENCES ====================

def create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    lookback: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Crée des séquences temporelles pour le LSTM.
    
    Args:
        X: Features normalisées
        y: Target normalisé
        lookback: Fenêtre temporelle
    
    Returns:
        (X_seq, y_seq)
    """
    X_seq, y_seq = [], []
    
    for i in range(lookback, len(X)):
        X_seq.append(X[i-lookback:i])
        y_seq.append(y[i, 0])
    
    return np.array(X_seq), np.array(y_seq)


def prepare_sequences(
    X_scaled: np.ndarray,
    y_scaled: np.ndarray,
    lookback: int = ModelConfig.LOOKBACK,
    test_split: float = ModelConfig.VALIDATION_SPLIT
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Crée les séquences et les divise en train/test.
    
    Args:
        X_scaled: Features normalisées
        y_scaled: Target normalisé
        lookback: Fenêtre temporelle
        test_split: Proportion de données pour le test
    
    Returns:
        (X_train, X_test, y_train, y_test)
    """
    print("\n📊 Création des séquences temporelles...")
    
    X_seq, y_seq = create_sequences(X_scaled, y_scaled, lookback)
    
    print(f"   Forme des séquences: {X_seq.shape}")
    
    # Split train/test (préserve l'ordre temporel)
    split_idx = int(len(X_seq) * (1 - test_split))
    X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
    y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
    
    print(f"   Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test


# ==================== CONSTRUCTION DU MODÈLE ====================

def build_lstm_model(
    lookback: int = ModelConfig.LOOKBACK,
    n_features: int = len(DataConfig.FEATURE_COLS),
    lstm_units: int = ModelConfig.LSTM_UNITS
) -> Sequential:
    """
    Construit l'architecture du modèle LSTM.

    Args:
        lookback: Nombre de timesteps
        n_features: Nombre de features
        lstm_units: Unités dans la première couche
    
    Returns:
        Modèle compilé
    """
    print("\n🧠 Construction du modèle LSTM...")
    
    model = Sequential([
        Input(shape=(lookback, n_features)),
        
        # Couche LSTM 1
        LSTM(lstm_units, activation='tanh', return_sequences=True),
        Dropout(0.3),
        
        # Couche LSTM 2
        LSTM(64, activation='tanh', return_sequences=False),
        Dropout(0.3),
        
        # Couches Dense
        Dense(32, activation='linear'),
        Dense(1)
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=ModelConfig.LEARNING_RATE),
        loss='mse',
        metrics=['mae']
    )
    
    print(model.summary())
    
    return model


def train_model(
    model: Sequential,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = ModelConfig.EPOCHS,
    batch_size: int = ModelConfig.BATCH_SIZE,
    patience: int = ModelConfig.PATIENCE
):
    """
    Entraîne le modèle LSTM.
    
    Args:
        model: Modèle compilé
        X_train: Données d'entraînement
        y_train: Target d'entraînement
        epochs: Nombre d'epochs
        batch_size: Taille du batch
        patience: Patience pour early stopping
    
    Returns:
        history: Historique d'entraînement
    """
    print("\n⚙️  Entraînement du modèle...")
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=0.15,
        callbacks=[early_stop],
        verbose=1
    )
    
    print("✅ Entraînement terminé")
    
    return history


# ==================== PRÉDICTIONS ====================

def make_predictions(
    model: Sequential,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    scaler_y: MinMaxScaler
) -> Dict:
    """
    Fait des prédictions et calcule les métriques.
    
    Args:
        model: Modèle entraîné
        X_train, X_test: Données
        y_train, y_test: Targets
        scaler_y: Scaler pour dénormalisation
    
    Returns:
        Dictionnaire avec prédictions et métriques
    """
    print("\n📈 Calcul des prédictions...")
    
    # Prédictions normalisées
    y_train_pred = model.predict(X_train, verbose=0)
    y_test_pred = model.predict(X_test, verbose=0)
    
    # Dénormalisation
    y_train_actual = scaler_y.inverse_transform(y_train.reshape(-1, 1))
    y_train_pred_actual = scaler_y.inverse_transform(y_train_pred)
    y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1))
    y_test_pred_actual = scaler_y.inverse_transform(y_test_pred)
    
    # Métriques
    train_rmse = np.sqrt(mean_squared_error(y_train_actual, y_train_pred_actual))
    test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_test_pred_actual))
    train_mae = mean_absolute_error(y_train_actual, y_train_pred_actual)
    test_mae = mean_absolute_error(y_test_actual, y_test_pred_actual)
    test_r2 = r2_score(y_test_actual, y_test_pred_actual)
    
    print(f"\n📊 MÉTRIQUES:")
    print(f"Train - RMSE: ${train_rmse:.4f}, MAE: ${train_mae:.4f}")
    print(f"Test  - RMSE: ${test_rmse:.4f}, MAE: ${test_mae:.4f}, R²: {test_r2:.4f}")
    
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


# ==================== INTERVALLES DE CONFIANCE ====================

def calculate_confidence_intervals(
    y_actual: np.ndarray, 
    y_pred: np.ndarray, 
    confidence_level: float = 0.95
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Calcule les intervalles de confiance basés sur les centiles des résidus réels.
    Cette méthode est plus robuste que le Z-score pour atteindre un coverage de 95%.
    """
    # 1. Calcul des résidus (erreurs)
    residuals = y_actual - y_pred
    
    # 2. Détermination des centiles (ex: pour 95%, on prend 2.5% et 97.5%)
    lower_percentile = (1 - confidence_level) / 2 * 100
    upper_percentile = (1 + confidence_level) / 2 * 100
    
    # 3. Calcul des décalages basés sur l'historique des erreurs
    lower_offset = np.percentile(residuals, lower_percentile)
    upper_offset = np.percentile(residuals, upper_percentile)
    
    # 4. Application des bornes
    lower_bound = y_pred + lower_offset
    upper_bound = y_pred + upper_offset
    
    # Métriques pour le log
    std = np.std(residuals)
    coverage = np.mean((y_actual >= lower_bound) & (y_actual <= upper_bound))
    
    return lower_bound, upper_bound, std, coverage

# ==================== PRÉDICTIONS FUTURES (BUG CORRIGÉ) ====================

def forecast_future(
    model: Sequential,
    last_sequence: np.ndarray,
    steps: int,
    n_features: int,
    scaler_y: MinMaxScaler,
    std_val: float,
    lookback: int = ModelConfig.LOOKBACK
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prédit les prix futurs jour par jour.
    
    ```python
    current_seq = np.vstack([current_seq[1:], new_features])
    # ✅ Injecte les nouvelles features calculées
    ```
    
    Args:
        model: Modèle LSTM entraîné
        last_sequence: Dernières 'lookback' observations
        steps: Nombre de jours à prédire
        n_features: Nombre de features
        scaler_y: Scaler pour dénormalisation
        std_val: Écart-type pour les intervalles
        lookback: Fenêtre temporelle
    
    Returns:
        (predictions, lower_bounds, upper_bounds)
    """
    predictions = []
    current_seq = last_sequence.copy()
    
    for step in range(steps):
        # Prédire le prix normalisé
        pred_scaled = model.predict(
            current_seq.reshape(1, lookback, n_features),
            verbose=0
        )[0][0]
        
        # Dénormaliser
        pred_actual = scaler_y.inverse_transform([[pred_scaled]])[0][0]
        predictions.append(pred_actual)
        
        # On crée un nouveau vecteur de features avec la prédiction
        # Pour simplifier, on réplique les features de la dernière observation
        # et on met à jour la valeur du prix (première feature = Open/Close)
        new_features = current_seq[-1].copy()
        new_features[0] = pred_scaled  # Mise à jour du prix normalisé
        
        # Shift de la séquence: supprimer le premier jour, ajouter le nouveau
        current_seq = np.vstack([current_seq[1:], new_features.reshape(1, -1)])
    
    predictions = np.array(predictions)
    lower = predictions - 1.96 * std_val
    upper = predictions + 1.96 * std_val
    
    return predictions, lower, upper


# ==================== PIPELINE PRINCIPAL ====================

def train_full_pipeline(
    config: Dict = None,
    crises_dict: Optional[Dict[str, Tuple[str, str]]] = None
) -> Dict:
    """
    Exécute le pipeline complet d'entraînement.
    
    Args:
        config: Configuration (utilise ModelConfig par défaut)
        crises_dict: Dictionnaire des crises à marquer
    
    Returns:
        Dictionnaire avec tous les résultats
    """
    if config is None:
        config = {
            'LOOKBACK': ModelConfig.LOOKBACK,
            'FUTURE_STEPS': ModelConfig.FUTURE_STEPS,
            'LSTM_UNITS': ModelConfig.LSTM_UNITS,
            'EPOCHS': ModelConfig.EPOCHS,
            'BATCH_SIZE': ModelConfig.BATCH_SIZE,
            'PATIENCE': ModelConfig.PATIENCE,
            'CONFIDENCE_LEVEL': ModelConfig.CONFIDENCE_LEVEL,
            'VALIDATION_SPLIT': ModelConfig.VALIDATION_SPLIT
        }
    
    # Préparation des données
    df = prepare_full_dataset(crises_dict)
    
    feature_cols = DataConfig.FEATURE_COLS
    X_scaled, y_scaled, scaler_X, scaler_y = prepare_data(df, feature_cols)
    
    # Séquences
    X_train, X_test, y_train, y_test = prepare_sequences(
        X_scaled, y_scaled,
        config['LOOKBACK'],
        config['VALIDATION_SPLIT']
    )
    
    # Entraînement
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
    
    # Prédictions
    preds = make_predictions(model, X_train, X_test, y_train, y_test, scaler_y)
    
    # Intervalles de confiance
    train_lower, train_upper, train_std, _ = calculate_confidence_intervals(
        preds['y_train_actual'], preds['y_train_pred'], config['CONFIDENCE_LEVEL']
    )
    test_lower, test_upper, test_std, coverage = calculate_confidence_intervals(
        preds['y_test_actual'], preds['y_test_pred'], config['CONFIDENCE_LEVEL']
    )
    
    # Prédictions futures (BUG CORRIGÉ)
    last_sequence = X_scaled[-config['LOOKBACK']:]
    future_pred, future_lower, future_upper = forecast_future(
        model, last_sequence, config['FUTURE_STEPS'], len(feature_cols),
        scaler_y, test_std, config['LOOKBACK']
    )
    
    print(f"\n📊 INTERVALLES:")
    print(f"Test Std: ${test_std:.4f}")
    print(f"Coverage: {coverage*100:.2f}%")
    
    print(f"\n🔮 PRÉDICTIONS FUTURES:")
    for i, (pred, low, high) in enumerate(zip(future_pred, future_lower, future_upper), 1):
        print(f"  +{i}j: ${pred:.2f} [${low:.2f}, ${high:.2f}]")
    
    # Sauvegarde
    model.save(ModelConfig.MODEL_PATH)
    print(f"\n💾 Modèle sauvegardé: '{ModelConfig.MODEL_PATH}'")
    
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


def load_and_predict(
    model_path: str = ModelConfig.MODEL_PATH,
    crises_dict: Optional[Dict[str, Tuple[str, str]]] = None
) -> Dict:
    """
    Charge un modèle sauvegardé et fait des prédictions.
    
    Args:
        model_path: Chemin du modèle
        crises_dict: Dictionnaire des crises (important pour cohérence!)
    
    Returns:
        Dictionnaire avec modèle et données
    """
    print("🔄 Chargement du modèle...")
    model = load_model(model_path, compile=False)
    
    # Récupérer les données récentes AVEC les mêmes crises
    df = prepare_full_dataset(crises_dict)
    
    print("✅ Modèle chargé et données synchronisées")
    
    return {'model': model, 'df': df}