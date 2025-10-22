"""
Data preparation script for admission prediction model.
Loads raw data, selects features based on correlation analysis,
normalizes features and splits into train/test sets.
"""

import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_and_prepare_data():
    """
    Load raw data and prepare it for modeling.
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, scaler)
    """
    
    # Load data
    data_path = os.path.join('data', 'raw', 'admission.csv')
    df = pd.read_csv(data_path)
    
    # Clean column names (remove trailing spaces)
    df.columns = df.columns.str.strip()
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Define target variable
    target_col = 'Chance of Admit'
    
    # Select features based on correlation analysis (correlation > 0.7 with target)
    # Based on correlation analysis from notebook:
    # CGPA: 0.882, GRE Score: 0.810, TOEFL Score: 0.792
    selected_features = ['GRE Score', 'TOEFL Score', 'CGPA']
    
    # Verify features exist in dataset
    missing_features = [f for f in selected_features if f not in df.columns]
    if missing_features:
        raise ValueError(f"Missing features in dataset: {missing_features}")
    
    print(f"Selected features (correlation > 0.7): {selected_features}")
    
    # Prepare feature matrix and target vector
    X = df[selected_features].copy()
    y = df[target_col].copy()
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    
    # Check for missing values
    print(f"\nMissing values in features:\n{X.isnull().sum()}")
    print(f"Missing values in target: {y.isnull().sum()}")
    
    # Split data into train and test sets (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=None  # For regression, we don't stratify
    )
    
    print(f"\nTrain set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    
    # Normalize features using StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrames to maintain feature names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=selected_features, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=selected_features, index=X_test.index)
    
    print(f"\nFeature scaling completed.")
    print(f"Training features - Mean: {X_train_scaled.mean().round(3).to_dict()}")
    print(f"Training features - Std: {X_train_scaled.std().round(3).to_dict()}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def save_processed_data(X_train, X_test, y_train, y_test, scaler):
    """
    Save processed data to the data/processed folder.
    
    Args:
        X_train, X_test, y_train, y_test: Split datasets
        scaler: Fitted StandardScaler object
    """
    
    # Create processed data directory if it doesn't exist
    processed_dir = os.path.join('data', 'processed')
    os.makedirs(processed_dir, exist_ok=True)
    
    # Save datasets
    X_train.to_csv(os.path.join(processed_dir, 'X_train.csv'), index=False)
    X_test.to_csv(os.path.join(processed_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(processed_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(processed_dir, 'y_test.csv'), index=False)
    
    # Save scaler for later use
    import joblib
    joblib.dump(scaler, os.path.join(processed_dir, 'scaler.pkl'))
    
    print(f"\nProcessed data saved to {processed_dir}/")
    print("Files created:")
    print("- X_train.csv")
    print("- X_test.csv") 
    print("- y_train.csv")
    print("- y_test.csv")
    print("- scaler.pkl")

def main():
    """Main function to prepare data."""
    print("Starting data preparation...")
    print("="*50)
    
    try:
        # Prepare data
        X_train, X_test, y_train, y_test, scaler = load_and_prepare_data()
        
        # Save processed data
        save_processed_data(X_train, X_test, y_train, y_test, scaler)
        
        print("="*50)
        print("Data preparation completed successfully!")
        
        # Display basic statistics
        print(f"\nFinal dataset statistics:")
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Number of features: {X_train.shape[1]}")
        print(f"Feature names: {list(X_train.columns)}")
        print(f"Target range: [{y_train.min():.3f}, {y_train.max():.3f}]")
        
    except Exception as e:
        print(f"Error during data preparation: {e}")
        raise

if __name__ == "__main__":
    main()