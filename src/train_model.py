"""
Model training script that loads the best model from grid search,
trains it on the full training set, evaluates on test set, 
and saves to BentoML Model Store.
"""

import json
import os

import bentoml
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_processed_data():
    """Load processed training and test data."""
    processed_dir = os.path.join('data', 'processed')
    
    X_train = pd.read_csv(os.path.join(processed_dir, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(processed_dir, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(processed_dir, 'y_train.csv')).squeeze()
    y_test = pd.read_csv(os.path.join(processed_dir, 'y_test.csv')).squeeze()
    
    print(f"Loaded data:")
    print(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  Test: {X_test.shape[0]} samples, {X_test.shape[1]} features")
    
    return X_train, X_test, y_train, y_test

def load_best_model_config():
    """Load best model configuration from grid search results."""
    results_path = os.path.join('models', 'grid_search_results.json')
    
    if not os.path.exists(results_path):
        raise FileNotFoundError("Grid search results not found. Run grid search first.")
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    best_model_name = results['best_model']
    best_params = results['best_params']
    
    print(f"Best model from grid search: {best_model_name}")
    print(f"Best parameters: {best_params}")
    
    return best_model_name, best_params

def create_and_train_model(best_model_name, best_params, X_train, y_train):
    """Create and train the best model with optimal parameters."""
    
    # Create model based on grid search results
    if best_model_name == 'Ridge':
        model = Ridge(random_state=42, **best_params)
    elif best_model_name == 'Lasso':
        from sklearn.linear_model import Lasso
        model = Lasso(random_state=42, max_iter=2000, **best_params)
    elif best_model_name == 'ElasticNet':
        from sklearn.linear_model import ElasticNet
        model = ElasticNet(random_state=42, max_iter=2000, **best_params)
    else:  # LinearRegression
        from sklearn.linear_model import LinearRegression
        model = LinearRegression(**best_params)
    
    print(f"Training {best_model_name} with parameters: {best_params}")
    
    # Train the model
    model.fit(X_train, y_train)
    
    print("Model training completed successfully")
    
    return model

def evaluate_model(model, X_train, X_test, y_train, y_test):
    """Evaluate model performance on train and test sets."""
    
    # Training predictions
    y_train_pred = model.predict(X_train)
    
    # Test predictions
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_metrics = {
        'r2': r2_score(y_train, y_train_pred),
        'rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
        'mae': mean_absolute_error(y_train, y_train_pred)
    }
    
    test_metrics = {
        'r2': r2_score(y_test, y_test_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
        'mae': mean_absolute_error(y_test, y_test_pred)
    }
    
    print("\nModel Performance Evaluation:")
    print("=" * 40)
    print(f"Training Set:")
    print(f"  R2 Score: {train_metrics['r2']:.4f}")
    print(f"  RMSE: {train_metrics['rmse']:.4f}")
    print(f"  MAE: {train_metrics['mae']:.4f}")
    
    print(f"\nTest Set:")
    print(f"  R2 Score: {test_metrics['r2']:.4f}")
    print(f"  RMSE: {test_metrics['rmse']:.4f}")
    print(f"  MAE: {test_metrics['mae']:.4f}")
    
    # Check for overfitting
    r2_diff = train_metrics['r2'] - test_metrics['r2']
    if r2_diff > 0.05:
        print(f"\nWarning: Potential overfitting detected (R2 difference: {r2_diff:.4f})")
    else:
        print(f"\nModel shows good generalization (R2 difference: {r2_diff:.4f})")
    
    return train_metrics, test_metrics, y_test_pred

def save_model_to_bentoml(model, best_model_name, best_params, train_metrics, test_metrics):
    """Save trained model to BentoML Model Store."""
    
    # Create model metadata
    metadata = {
        'model_type': best_model_name,
        'hyperparameters': best_params,
        'features': ['GRE Score', 'TOEFL Score', 'CGPA'],
        'target': 'Chance of Admit',
        'training_metrics': train_metrics,
        'test_metrics': test_metrics,
        'framework': 'scikit-learn'
    }
    
    # Save model to BentoML
    try:
        bento_model = bentoml.sklearn.save_model(
            name="admission_predictor",
            model=model,
            labels={
                "model_type": best_model_name,
                "framework": "scikit-learn",
                "task": "regression"
            },
            metadata=metadata
        )
        
        print(f"\nModel saved to BentoML Model Store:")
        print(f"  Model tag: {bento_model.tag}")
        print(f"  Model path: {bento_model.path}")
        
        return bento_model.tag
        
    except Exception as e:
        print(f"Error saving model to BentoML: {e}")
        raise

def save_training_artifacts(model, best_model_name, best_params, train_metrics, test_metrics, model_tag):
    """Save training artifacts and results."""
    
    models_dir = 'models'
    
    # Save final trained model
    final_model_path = os.path.join(models_dir, 'final_model.pkl')
    joblib.dump(model, final_model_path)
    
    # Save training results
    training_results = {
        'model_name': best_model_name,
        'model_parameters': best_params,
        'bentoml_tag': str(model_tag),
        'training_metrics': train_metrics,
        'test_metrics': test_metrics,
        'model_artifacts': {
            'final_model_path': final_model_path,
            'bentoml_model_store': True
        }
    }
    
    results_path = os.path.join(models_dir, 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(training_results, f, indent=2)
    
    print(f"\nTraining artifacts saved:")
    print(f"  Final model: {final_model_path}")
    print(f"  Training results: {results_path}")
    
    return results_path

def main():
    """Main function for model training."""
    print("Starting Model Training")
    print("=" * 50)
    
    try:
        # Load processed data
        X_train, X_test, y_train, y_test = load_processed_data()
        
        # Load best model configuration from grid search
        best_model_name, best_params = load_best_model_config()
        
        # Create and train model
        model = create_and_train_model(best_model_name, best_params, X_train, y_train)
        
        # Evaluate model performance
        train_metrics, test_metrics, y_test_pred = evaluate_model(
            model, X_train, X_test, y_train, y_test
        )
        
        # Save model to BentoML
        model_tag = save_model_to_bentoml(
            model, best_model_name, best_params, train_metrics, test_metrics
        )
        
        # Save training artifacts
        results_path = save_training_artifacts(
            model, best_model_name, best_params, train_metrics, test_metrics, model_tag
        )
        
        print("\n" + "=" * 50)
        print("Model Training Completed Successfully!")
        print(f"Model: {best_model_name}")
        print(f"Test R2: {test_metrics['r2']:.4f}")
        print(f"Test RMSE: {test_metrics['rmse']:.4f}")
        print(f"BentoML Tag: {model_tag}")

    except Exception as e:
        print(f"Error during model training: {e}")
        raise

if __name__ == "__main__":
    main()