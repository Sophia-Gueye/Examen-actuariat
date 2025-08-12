import pandas as pd
from sklearn.ensemble import RandomForestRegressor  # Correction de l'import

def get_feature_importance(df_encoded, target='claim'):
    """Calculate feature importance using Random Forest."""
    if target not in df_encoded.columns:  # Correction: df_encoded au lieu de df_clean
        raise ValueError(f"Target column '{target}' not found in DataFrame.")
    
    # Separate features and target
    X = df_encoded.drop(columns=[target])
    y = df_encoded[target]
    
    # Initialize Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Fit the model
    model.fit(X, y)
    
    # Get feature importances
    feature_importances = pd.Series(
        model.feature_importances_, 
        index=X.columns
    ).sort_values(ascending=False)
    
    return feature_importances