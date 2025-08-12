import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

def compare_models(results_list):
    """Compare multiple model results."""
    if not results_list:
        raise ValueError("Results list cannot be empty")
    
    comparison_df = pd.DataFrame(results_list)
    
    # Sort by R² score (descending) then by MAE (ascending)
    comparison_df = comparison_df.sort_values(['r2', 'mae'], ascending=[False, True])
    
    return comparison_df

def cross_validate_model(model, X, y, cv=5, scoring='neg_mean_absolute_error'):
    """Perform cross-validation on a model."""
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    
    # Convert negative MAE to positive
    if scoring == 'neg_mean_absolute_error':
        scores = -scores
    
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    
    return mean_score, std_score

def detailed_cross_validation(model, X, y, cv=5):
    """Perform detailed cross-validation with multiple metrics."""
    scoring = ['neg_mean_absolute_error', 'neg_mean_squared_error', 'r2']
    
    cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, return_train_score=True)
    
    results = {
        'test_mae_mean': -np.mean(cv_results['test_neg_mean_absolute_error']),
        'test_mae_std': np.std(cv_results['test_neg_mean_absolute_error']),
        'test_mse_mean': -np.mean(cv_results['test_neg_mean_squared_error']),
        'test_mse_std': np.std(cv_results['test_neg_mean_squared_error']),
        'test_r2_mean': np.mean(cv_results['test_r2']),
        'test_r2_std': np.std(cv_results['test_r2']),
        'train_mae_mean': -np.mean(cv_results['train_neg_mean_absolute_error']),
        'train_mae_std': np.std(cv_results['train_neg_mean_absolute_error']),
        'train_r2_mean': np.mean(cv_results['train_r2']),
        'train_r2_std': np.std(cv_results['train_r2'])
    }
    
    return results

def plot_model_comparison(comparison_df, metric='mae'):
    """Plot comparison of models based on a specific metric."""
    plt.figure(figsize=(10, 6))
    
    if metric.lower() == 'mae':
        sns.barplot(data=comparison_df, x='model_name', y='mae')
        plt.title('Model Comparison - Mean Absolute Error (Lower is Better)')
        plt.ylabel('MAE')
    elif metric.lower() == 'r2':
        sns.barplot(data=comparison_df, x='model_name', y='r2')
        plt.title('Model Comparison - R² Score (Higher is Better)')
        plt.ylabel('R² Score')
    elif metric.lower() == 'rmse':
        sns.barplot(data=comparison_df, x='model_name', y='rmse')
        plt.title('Model Comparison - Root Mean Squared Error (Lower is Better)')
        plt.ylabel('RMSE')
    
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def plot_predictions_vs_actual(y_true, y_pred, model_name='Model'):
    """Plot predictions vs actual values."""
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 2, 1)
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'{model_name} - Predictions vs Actual')
    
    plt.subplot(2, 2, 2)
    residuals = y_true - y_pred
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(f'{model_name} - Residual Plot')
    
    plt.subplot(2, 2, 3)
    plt.hist(residuals, bins=30, alpha=0.7)
    plt.xlabel('Residuals')
    plt.ylabel('Frequency')
    plt.title(f'{model_name} - Residuals Distribution')
    
    plt.subplot(2, 2, 4)
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title(f'{model_name} - Q-Q Plot')
    
    plt.tight_layout()
    plt.show()

def calculate_model_metrics(y_true, y_pred):
    """Calculate comprehensive model metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Additional metrics
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # Mean Absolute Percentage Error
    
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R²': r2,
        'MAPE (%)': mape
    }

def print_evaluation_report(results):
    """Print a formatted evaluation report."""
    print("=" * 50)
    print("MODEL EVALUATION REPORT")
    print("=" * 50)
    
    for key, value in results.items():
        if isinstance(value, float):
            print(f"{key:15s}: {value:.4f}")
        else:
            print(f"{key:15s}: {value}")
    
    print("=" * 50)