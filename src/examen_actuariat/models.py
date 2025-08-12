import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not installed. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not installed. Install with: pip install lightgbm")

def train_xgboost(X_train, y_train, **kwargs):
    """Train XGBoost model."""
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost is not installed. Please install it with: pip install xgboost")
    
    # Default parameters
    params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42
    }
    params.update(kwargs)
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    
    return model

def train_lightgbm(X_train, y_train, **kwargs):
    """Train LightGBM model."""
    if not LIGHTGBM_AVAILABLE:
        raise ImportError("LightGBM is not installed. Please install it with: pip install lightgbm")
    
    # Default parameters
    params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42,
        'verbose': -1
    }
    params.update(kwargs)
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    
    return model


def evaluate_model(model, X_test, y_test, model_name='Model'):
    """Evaluate a trained model."""
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    results = {
        'model_name': model_name,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2,
        'predictions': y_pred
    }
    
    return results

def save_model(model, filepath):
    """Save a trained model to disk."""
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")