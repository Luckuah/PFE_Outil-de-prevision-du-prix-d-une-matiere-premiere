from model_training import prepare_data,forecast_future
from config_param import DataConfig,ModelConfig
import numpy

def predict_lstm(df_recent,model):
    # Préparation manuelle pour l'inférence
    X_scaled, y_scaled, scaler_X, scaler_y = prepare_data(df_recent, DataConfig.FEATURE_COLS)
    last_sequence = X_scaled[-ModelConfig.LOOKBACK:]
    
    # Simulation des résultats pour l'affichage
    preds, low, high = forecast_future(
        model, last_sequence, ModelConfig.FUTURE_STEPS, 
        len(DataConfig.FEATURE_COLS), scaler_y, std_val=1.5, lookback=ModelConfig.LOOKBACK
    )
    return (preds.tolist(), low.tolist(), high.tolist())
