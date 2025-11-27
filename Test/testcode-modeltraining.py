import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
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
OIL_TICKER = 'CL=F'  # Crude Oil futures
VIX_TICKER = '^VIX'
START_DATE = '2020-01-01'
END_DATE = '2025-01-31'
LOOKBACK = 60
FUTURE_STEPS = 10
LSTM_UNITS = 128
EPOCHS = 10
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.05

# ==================== 1. TÉLÉCHARGER LES DONNÉES ====================
print("📊 Téléchargement des données...")
oil_df = yf.download(OIL_TICKER, start=START_DATE, end=END_DATE, progress=False)
vix_df = yf.download(VIX_TICKER, start=START_DATE, end=END_DATE, progress=False)

# Merger VIX avec Oil
vix_close = vix_df[['Close']].rename(columns={'Close': 'VIX_Close'})
df = oil_df.join(vix_close, how='left')
df = df.fillna(method='ffill').dropna()

print(f"✅ Données chargées: {len(df)} jours")

# ==================== 2. AJOUTER INDICATEURS TECHNIQUES (TA-LIB PYTHON) ====================
print("\n📊 Calcul des indicateurs techniques (TA)...")

high = df['High'].squeeze()
low = df['Low'].squeeze()
close = df['Close'].squeeze()
volume = df['Volume'].squeeze()

# RSI - Relative Strength Index (momentum)
rsi = RSIIndicator(close=close, window=14)
df['RSI_14'] = rsi.rsi()

# MACD - Moving Average Convergence Divergence
macd = MACD(close=close, window_fast=12, window_slow=26, window_sign=9)
df['MACD'] = macd.macd()
df['MACD_Signal'] = macd.macd_signal()
df['MACD_Hist'] = macd.macd_diff()

# ATR - Average True Range (volatility)
atr = AverageTrueRange(high=high, low=low, close=close, window=14)
df['ATR_14'] = atr.average_true_range()

# Bollinger Bands
bb = BollingerBands(close=close, window=20, window_dev=2)
df['BB_Upper'] = bb.bollinger_hband()
df['BB_Mid'] = bb.bollinger_mavg()
df['BB_Lower'] = bb.bollinger_lband()

# ADX - Average Directional Index (trend strength)
adx = ADXIndicator(high=high, low=low, close=close, window=14)
df['ADX_14'] = adx.adx()

# Volume Price Trend
df['VROC'] = ta.volume.volume_price_trend(close=close, volume=volume)

# Price Return
df['Price_Return'] = df['Close'].pct_change()

print("✅ Indicateurs techniques ajoutés")

# ==================== 3. MAPPER LES CRISES (2024-2025) ====================
print("\n📍 Mapping des régimes de marché...")

def map_market_regime(date):
    """
    Mappe les régimes de marché basé sur la période
    -1 = Bear market / Crise
    0 = Normal / Transition
    1 = Bull market / Recovery
    """
    date_val = date
    
    # 2024
    if pd.Timestamp('2024-01-01') <= date_val <= pd.Timestamp('2024-12-31'):
        return 1  # Bull market 2024
    
    # 2025
    elif pd.Timestamp('2025-01-01') <= date_val <= pd.Timestamp('2025-01-31'):
        return 1  # Début 2025
    
    else:
        return 0

df['Market_Regime'] = df.index.map(lambda x: map_market_regime(x))

print("✅ Régime de marché mappé")

# ==================== 4. NETTOYER LES DONNÉES ====================
df = df.dropna()
print(f"\n✅ Données après nettoyage: {len(df)} jours")
print(f"   Prix min: ${df['Close'].values.min():.2f}, Prix max: ${df['Close'].values.max():.2f}")
print(f"   VIX min: {df['VIX_Close'].values.min():.2f}, VIX max: {df['VIX_Close'].values.max():.2f}")

# ==================== 5. PRÉPARER LES FEATURES ====================
feature_cols = ['Open', 'High', 'Low', 'Volume', 'VIX_Close', 
                'RSI_14', 'MACD', 'ATR_14', 'ADX_14', 'VROC',
                'BB_Upper', 'BB_Mid', 'BB_Lower', 'Market_Regime']
target_col = 'Close'

X_data = df[feature_cols].values
y_data = df[target_col].values.reshape(-1, 1)

print(f"\n📐 Features count: {len(feature_cols)}")
print(f"   X shape: {X_data.shape}")
print(f"   y shape: {y_data.shape}")

# ==================== 6. NORMALISER ====================
print("\n🔄 Normalisation MinMax...")
scaler_X = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler_X.fit_transform(X_data)

scaler_y = MinMaxScaler(feature_range=(0, 1))
y_scaled = scaler_y.fit_transform(y_data)

print("✅ Données normalisées [0, 1]")

# ==================== 7. CRÉER SÉQUENCES ====================
def create_sequences(X, y, lookback):
    X_seq, y_seq = [], []
    for i in range(lookback, len(X)):
        X_seq.append(X[i-lookback:i])
        y_seq.append(y[i, 0])
    return np.array(X_seq), np.array(y_seq)

X_seq, y_seq = create_sequences(X_scaled, y_scaled, LOOKBACK)

print(f"\n📊 Séquences créées:")
print(f"   Shape: {X_seq.shape} (samples, timesteps, features)")

# ==================== 8. SPLIT TRAIN/TEST ====================
split_idx = int(len(X_seq) * (1 - VALIDATION_SPLIT))
X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

print(f"\n✂️ Train/Test:")
print(f"   Train: {X_train.shape[0]} samples")
print(f"   Test: {X_test.shape[0]} samples")

# ==================== 9. LSTM MODEL ====================
print("\n🧠 Construction LSTM...")

model = Sequential([
    Input(shape=(LOOKBACK, len(feature_cols))),
    LSTM(LSTM_UNITS, activation='relu', return_sequences=True),
    Dropout(0.3),
    LSTM(64, activation='relu', return_sequences=False),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)  # Single output: predicted price
])

model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
print(model.summary())

# ==================== 10. ENTRAÎNEMENT ====================
print("\n⚙️ Entraînement...")
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

history = model.fit(
    X_train, y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.15,
    callbacks=[early_stop],
    verbose=1
)

# ==================== 11. PRÉDICTIONS ====================
print("\n📈 Prédictions...")
y_train_pred = model.predict(X_train, verbose=0)
y_test_pred = model.predict(X_test, verbose=0)

# Denormaliser
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
print(f"Train - RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}")
print(f"Test  - RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, R²: {test_r2:.4f}")

# ==================== 12. CALCULER LES INTERVALLES (Std des résidus) ====================
train_residuals = y_train_actual - y_train_pred_actual
test_residuals = y_test_actual - y_test_pred_actual

train_std = np.std(train_residuals)
test_std = np.std(test_residuals)

# Intervalles à 95% (±1.96 * std)
y_train_pred_lower = y_train_pred_actual - 1.96 * train_std
y_train_pred_upper = y_train_pred_actual + 1.96 * train_std

y_test_pred_lower = y_test_pred_actual - 1.96 * test_std
y_test_pred_upper = y_test_pred_actual + 1.96 * test_std

# Coverage
coverage_test = np.mean((y_test_actual >= y_test_pred_lower) & 
                        (y_test_actual <= y_test_pred_upper))

print(f"\n📊 INTERVALLES DE CONFIANCE (95%):")
print(f"Train Std: ${train_std:.4f}")
print(f"Test Std: ${test_std:.4f}")
print(f"Coverage Test: {coverage_test*100:.2f}%")

# ==================== 13. FORECAST FUTUR ====================
def forecast_future(model, last_sequence, steps, scaler_y, std_val):
    predictions = []
    current_seq = last_sequence.copy()
    
    for _ in range(steps):
        pred_scaled = model.predict(current_seq.reshape(1, LOOKBACK, len(feature_cols)), verbose=0)[0][0]
        pred_actual = scaler_y.inverse_transform([[pred_scaled]])[0][0]
        predictions.append(pred_actual)
        current_seq = np.vstack([current_seq[1:], current_seq[-1:]])
    
    predictions = np.array(predictions)
    lower = predictions - 1.96 * std_val
    upper = predictions + 1.96 * std_val
    
    return predictions, lower, upper

last_sequence = X_scaled[-LOOKBACK:]
future_pred, future_lower, future_upper = forecast_future(model, last_sequence, FUTURE_STEPS, scaler_y, test_std)

print(f"\n🔮 PRÉDICTIONS FUTURES ({FUTURE_STEPS} jours):")
for i, (pred, low, high) in enumerate(zip(future_pred, future_lower, future_upper), 1):
    print(f"  +{i}j: ${pred:.2f} [${low:.2f}, ${high:.2f}]")

# ==================== 14. VISUALISATION ====================
fig, axes = plt.subplots(3, 1, figsize=(18, 14))

# Plot 1: Loss
ax1 = axes[0]
ax1.plot(history.history['loss'], label='Train Loss', alpha=0.7, linewidth=2)
ax1.plot(history.history['val_loss'], label='Val Loss', alpha=0.7, linewidth=2)
ax1.set_title('LSTM Loss Over Epochs', fontsize=13, fontweight='bold')
ax1.set_ylabel('Loss (MSE)')
ax1.set_xlabel('Epoch')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# Plot 2: Train/Test avec intervalles
ax2 = axes[1]
train_dates = df.index[LOOKBACK:LOOKBACK+len(y_train_actual)]
test_dates = df.index[LOOKBACK+len(y_train_actual):LOOKBACK+len(y_train_actual)+len(y_test_actual)]

# Train
ax2.plot(train_dates, y_train_actual, label='Train Actual', alpha=0.6, linewidth=1.5, color='black')
ax2.plot(train_dates, y_train_pred_actual, label='Train Pred', alpha=0.6, linewidth=1.5, color='blue')
ax2.fill_between(train_dates, y_train_pred_lower.flatten(), y_train_pred_upper.flatten(),
                 alpha=0.15, color='blue', label='Train Interval (95%)')

# Separator
ax2.axvline(test_dates[0], color='red', linestyle='--', linewidth=2.5, label='Train/Test Split')

# Test
ax2.plot(test_dates, y_test_actual, label='Test Actual', alpha=0.8, linewidth=2, color='black')
ax2.plot(test_dates, y_test_pred_actual, label='Test Pred', alpha=0.8, linewidth=2, color='green')
ax2.fill_between(test_dates, y_test_pred_lower.flatten(), y_test_pred_upper.flatten(),
                 alpha=0.25, color='green', label='Test Interval (95%)')

ax2.set_title(f'Oil Price Predictions with Confidence Intervals | Coverage: {coverage_test*100:.1f}% | RMSE: {test_rmse:.2f}', 
              fontsize=13, fontweight='bold')
ax2.set_ylabel('Price ($)', fontsize=11)
ax2.legend(fontsize=10, loc='best')
ax2.grid(True, alpha=0.3)

# Plot 3: Future forecast
ax3 = axes[2]
future_dates = pd.date_range(test_dates[-1], periods=FUTURE_STEPS+1, freq='D')[1:]

# Derniers jours réels
ax3.plot(test_dates[-30:], y_test_actual[-30:], label='Recent History', alpha=0.8, linewidth=2.5, color='black', marker='o', markersize=4)

# Forecast
ax3.plot(future_dates, future_pred, 'g-', linewidth=3, marker='o', markersize=8, label='Forecast')
ax3.fill_between(future_dates, future_lower, future_upper, alpha=0.3, color='green', label='95% Interval')
ax3.plot(future_dates, future_lower, 'g--', linewidth=1.5, alpha=0.7)
ax3.plot(future_dates, future_upper, 'g--', linewidth=1.5, alpha=0.7)

ax3.set_title(f'Oil Price Forecast - Next {FUTURE_STEPS} Days', fontsize=13, fontweight='bold')
ax3.set_xlabel('Date', fontsize=11)
ax3.set_ylabel('Price ($)', fontsize=11)
ax3.legend(fontsize=10, loc='best')
ax3.grid(True, alpha=0.3)
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('lstm_oil_forecast.png', dpi=100, bbox_inches='tight')
plt.show()

print("\n✅ Graphiques sauvegardés: 'lstm_oil_forecast.png'")

# ==================== 15. SAUVEGARDER ====================
model.save('lstm_oil_model.h5')
print("💾 Modèle sauvegardé: 'lstm_oil_model.h5'")

print("\n" + "="*60)
print("✅ ENTRAÎNEMENT TERMINÉ - LSTM OIL PRICE FORECASTING")
print("="*60)