import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder  # Ajout de LabelEncoder
from imblearn.over_sampling import SMOTE

def clean_data(df):
    """Clean the data by removing duplicates and handling missing values."""
    
    # Drop rows with NaN values in other columns
    df_clean = df.dropna().drop_duplicates()
    
    return df_clean  # Correction: retourner df_clean

def encodage(df):
    """Encode categorical variables using label encoding."""
    df_encoded = df.copy()
    
    # Encode categorical variables
    for col in df_encoded.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])

    return df_encoded

def scale_features(df_encoded, target='claim'):
    """Scale numerical features using StandardScaler."""
    if target not in df_encoded.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")
    
    # Separate features and target
    X = df_encoded.drop(columns=[target])
    y = df_encoded[target]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X),
        columns=X.columns,
        index=X.index
    )
    
    # Combine scaled features with target
    df_scaled = pd.concat([X_scaled, y], axis=1)
    
    return df_scaled, scaler

def apply_smote(X, y, random_state=42):
    """Apply SMOTE for handling imbalanced data."""
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    return X_resampled, y_resampled