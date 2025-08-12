import sys
from pathlib import Path
from sklearn.model_selection import train_test_split

# Ajouter le répertoire parent au path pour les imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import direct du package installé par Poetry
try:
    from examen_actuariat import (
        data_loading, 
        data_processing, 
        features,  
        exploration,
        models,
        evaluation
    )
except ImportError:
    # Fallback : import depuis src
    try:
        from src.examen_actuariat import (
            data_loading, 
            data_processing, 
            features,  
            exploration,
            models,
            evaluation
        )
    except ImportError as e:
        print(f"Erreur d'import: {e}")
        print("Assurez-vous que le package est installé avec 'poetry install'")
        print("Et que la structure src/examen_actuariat/ existe")
        sys.exit(1)

def main():
    # Configuration
    DATA_PATH = Path('data/insurance-demographic-health.csv')
    TARGET = 'claim'

    try:
        # Load data
        print("Chargement des données...")
        df = data_loading.load_raw(DATA_PATH)
        print(f"Données chargées: {df.shape}")

        # Nettoyage
        print("Nettoyage des données...")
        df_clean = data_processing.clean_data(df)
        print(f"Données nettoyées: {df_clean.shape}")

        # Analyse correlations
        print("\nAnalyse des corrélations...")
        correlations = exploration.analyze_correlation(df_clean)
        for key, value in correlations.items():
            print(f"{key}: {value:.3f}")

        # Analyse par genre
        print("\nAnalyse par genre...")
        gender_stats = exploration.analyze_by_gender(df_clean)
        if gender_stats is not None:
            print("Statistiques par genre:")
            print(gender_stats)

        # Analyse de l'impact du tabagisme
        print("\nAnalyse de l'impact du tabagisme...")
        smoking_stats = exploration.analyze_smoking_impact(df_clean)
        if smoking_stats:
            print("Smoking Impact Analysis")
            print(f"Mean Claim Amount for Smokers: {smoking_stats['mean_smoker']:.2f}")
            print(f"Mean Claim Amount for Non-Smokers: {smoking_stats['mean_non_smoker']:.2f}")
            print(f"Correlation between Smoking and Claim Amount: {smoking_stats['correlation_smoker_claim']:.3f}")

        # Encodage des variables et feature importance
        print("\nEncodage des variables...")
        df_encoded = data_processing.encodage(df_clean)
        
        print("Calcul de l'importance des features...")
        feature_importances = features.get_feature_importance(df_encoded, target=TARGET)
        print("Feature Importances:")
        print(feature_importances.head())

        # Préparation des données pour la modélisation
        print("\nPréparation des données pour la modélisation...")
        X = df_encoded.drop(columns=[TARGET])
        y = df_encoded[TARGET]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Taille d'entraînement: {X_train.shape}")
        print(f"Taille de test: {X_test.shape}")

        # Entraînement des modèles
        print("\nEntraînement des modèles...")
        results = []

        # XGBoost
        try:
            print("Entraînement XGBoost...")
            xgb_model = models.train_xgboost(X_train, y_train)
            xgb_results = models.evaluate_model(xgb_model, X_test, y_test, model_name='XGBoost')
            results.append(xgb_results)
        except ImportError as e:
            print(f"XGBoost non disponible: {e}")
        except Exception as e:
            print(f"Erreur XGBoost: {e}")

        # LightGBM
        try:
            print("Entraînement LightGBM...")
            lgb_model = models.train_lightgbm(X_train, y_train)
            lgb_results = models.evaluate_model(lgb_model, X_test, y_test, model_name='LightGBM')
            results.append(lgb_results)
        except ImportError as e:
            print(f"LightGBM non disponible: {e}")
        except Exception as e:
            print(f"Erreur LightGBM: {e}")

        # Vérifier qu'au moins un modèle a été entraîné
        if not results:
            print("\n❌ Aucun modèle disponible!")
            print("Installation recommandée:")
            print("poetry add xgboost lightgbm")
            return

        # Comparaison des modèles
        print("\nComparaison des modèles...")
        comparison = evaluation.compare_models(results)
        print("Résultats de comparaison:")
        print(comparison[['model_name', 'mae', 'rmse', 'r2']])

        # Validation croisée pour le meilleur modèle
        best_model_name = comparison.iloc[0]['model_name']
        print(f"\nValidation croisée pour le meilleur modèle: {best_model_name}")
        
        # Sélectionner le meilleur modèle
        if best_model_name == 'XGBoost' and 'xgb_model' in locals():
            best_model = xgb_model
        elif best_model_name == 'LightGBM' and 'lgb_model' in locals():
            best_model = lgb_model
        else:
            # Fallback au premier modèle disponible
            if 'xgb_model' in locals():
                best_model = xgb_model
                best_model_name = 'XGBoost'
            elif 'lgb_model' in locals():
                best_model = lgb_model
                best_model_name = 'LightGBM'

        mae_cv, std_cv = evaluation.cross_validate_model(best_model, X, y, cv=5)
        print(f"{best_model_name} Cross-Validation MAE: {mae_cv:.2f} ± {std_cv:.2f}")

        # Sauvegarde du meilleur modèle
        model_path = Path('models') / f'best_model_{best_model_name.lower().replace(" ", "_")}.joblib'
        model_path.parent.mkdir(exist_ok=True)
        models.save_model(best_model, model_path)

        print("\n" + "="*50)
        print("ANALYSE TERMINÉE AVEC SUCCÈS !")
        print(f"Meilleur modèle: {best_model_name}")
        print(f"MAE: {comparison.iloc[0]['mae']:.2f}")
        print(f"R²: {comparison.iloc[0]['r2']:.3f}")
        print("="*50)

    except FileNotFoundError:
        print(f"Erreur: Le fichier {DATA_PATH} n'a pas été trouvé.")
        print("Assurez-vous que le fichier de données est présent dans le répertoire spécifié.")
    except Exception as e:
        print(f"Erreur lors de l'exécution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()