"""
Grid search script for hyperparameter optimization of regression models.
Compares different regression algorithms and finds optimal parameters.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, cross_val_score


def load_processed_data():
    """Load processed training data."""
    processed_dir = os.path.join('data', 'processed')
    
    X_train = pd.read_csv(os.path.join(processed_dir, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(processed_dir, 'y_train.csv')).squeeze()
    
    print(f"Loaded training data: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    return X_train, y_train

def define_model_grid():
    """Define models and hyperparameter grids for search."""
    
    models = {
        'LinearRegression': {
            'model': LinearRegression(),
            'params': {}  # No hyperparameters to tune
        },
        'Ridge': {
            'model': Ridge(random_state=42),
            'params': {
                'alpha': [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
            }
        },
        'Lasso': {
            'model': Lasso(random_state=42, max_iter=2000),
            'params': {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
            }
        },
        'ElasticNet': {
            'model': ElasticNet(random_state=42, max_iter=2000),
            'params': {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            }
        }
    }
    
    return models

def perform_grid_search(X_train, y_train):
    """
    Perform grid search for all models and return results.
    
    Args:
        X_train: Training features
        y_train: Training target
    
    Returns:
        dict: Results for all models
    """
    
    models = define_model_grid()
    results = {}
    
    print("Starting Grid Search for Regression Models")
    print("=" * 50)
    
    for model_name, model_config in models.items():
        print(f"\nGrid Search for {model_name}...")
        
        if model_config['params']:  # If there are parameters to tune
            grid_search = GridSearchCV(
                estimator=model_config['model'],
                param_grid=model_config['params'],
                cv=5,
                scoring='neg_mean_squared_error',
                n_jobs=-1,
                verbose=0
            )
            
            grid_search.fit(X_train, y_train)
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_score = -grid_search.best_score_  # Convert back to positive RMSE
            
        else:  # For LinearRegression (no parameters)
            best_model = model_config['model']
            best_model.fit(X_train, y_train)
            best_params = {}
            
            # Calculate cross-validation score manually
            cv_scores = cross_val_score(
                best_model, X_train, y_train, 
                cv=5, scoring='neg_mean_squared_error'
            )
            best_score = -cv_scores.mean()
        
        # Calculate additional metrics
        y_train_pred = best_model.predict(X_train)
        train_r2 = r2_score(y_train, y_train_pred)
        train_mae = mean_absolute_error(y_train, y_train_pred)
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        
        results[model_name] = {
            'model': best_model,
            'best_params': best_params,
            'cv_rmse': np.sqrt(best_score),
            'train_r2': train_r2,
            'train_mae': train_mae,
            'train_rmse': train_rmse
        }
        
        print(f"  Best parameters: {best_params}")
        print(f"  CV RMSE: {np.sqrt(best_score):.4f}")
        print(f"  Train R2: {train_r2:.4f}")
    
    return results

def select_best_model(results):
    """Select the best model based on cross-validation RMSE."""
    
    print("\n" + "=" * 50)
    print("MODEL COMPARISON RESULTS:")
    print("=" * 50)
    
    # Create comparison DataFrame
    comparison_data = []
    for model_name, result in results.items():
        comparison_data.append({
            'Model': model_name,
            'CV_RMSE': result['cv_rmse'],
            'Train_R2': result['train_r2'],
            'Train_MAE': result['train_mae'],
            'Best_Params': str(result['best_params'])
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df = comparison_df.sort_values('CV_RMSE')
    
    print(comparison_df.to_string(index=False))
    
    # Select best model (lowest CV RMSE)
    best_model_name = comparison_df.iloc[0]['Model']
    best_model_result = results[best_model_name]
    
    print(f"\nBEST MODEL: {best_model_name}")
    print(f"   CV RMSE: {best_model_result['cv_rmse']:.4f}")
    print(f"   Train R2: {best_model_result['train_r2']:.4f}")
    print(f"   Parameters: {best_model_result['best_params']}")
    
    return best_model_name, best_model_result

def save_results(best_model_name, best_result, all_results):
    """Save grid search results and best model."""
    
    # Create models directory if it doesn't exist
    models_dir = 'models'
    os.makedirs(models_dir, exist_ok=True)
    
    # Save best model
    best_model_path = os.path.join(models_dir, 'best_model.pkl')
    joblib.dump(best_result['model'], best_model_path)
    
    # Save grid search results
    results_summary = {
        'best_model': best_model_name,
        'best_params': best_result['best_params'],
        'best_cv_rmse': best_result['cv_rmse'],
        'best_train_r2': best_result['train_r2'],
        'all_results': {
            name: {
                'cv_rmse': result['cv_rmse'],
                'train_r2': result['train_r2'],
                'train_mae': result['train_mae'],
                'train_rmse': result['train_rmse'],
                'best_params': result['best_params']
            }
            for name, result in all_results.items()
        }
    }
    
    results_path = os.path.join(models_dir, 'grid_search_results.json')
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    # Save model metadata for BentoML later
    model_metadata = {
        'model_name': best_model_name,
        'model_type': 'regression',
        'features': ['GRE Score', 'TOEFL Score', 'CGPA'],
        'target': 'Chance of Admit',
        'hyperparameters': best_result['best_params'],
        'performance_metrics': {
            'cv_rmse': best_result['cv_rmse'],
            'train_r2': best_result['train_r2'],
            'train_mae': best_result['train_mae']
        }
    }
    
    metadata_path = os.path.join(models_dir, 'model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(model_metadata, f, indent=2)
    
    print(f"\nResults saved:")
    print(f"   - Best model: {best_model_path}")
    print(f"   - Grid search results: {results_path}")
    print(f"   - Model metadata: {metadata_path}")

def main():
    """Main function for grid search."""
    print("Starting Hyperparameter Grid Search")
    print("=" * 50)
    
    try:
        # Load data
        X_train, y_train = load_processed_data()
        
        # Perform grid search
        results = perform_grid_search(X_train, y_train)
        
        # Select best model
        best_model_name, best_result = select_best_model(results)
        
        # Save results
        save_results(best_model_name, best_result, results)
        
        print("\n" + "=" * 50)
        print("Grid Search completed successfully!")
        print(f"Best model: {best_model_name}")
        print(f"Cross-validation RMSE: {best_result['cv_rmse']:.4f}")
        print(f"Training R2: {best_result['train_r2']:.4f}")
        
    except Exception as e:
        print(f"Error during grid search: {e}")
        raise

if __name__ == "__main__":
    main()