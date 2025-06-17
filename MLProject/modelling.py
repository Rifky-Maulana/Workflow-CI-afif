"""
Modelling.py - MLflow Implementation for CI/CD
Kriteria 3: Workflow CI dengan MLflow Project
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import argparse
import sys
import os

warnings.filterwarnings('ignore')

def load_data():
    """Load preprocessed data"""
    print("📊 Loading preprocessed data...")
    
    # Try to load the modeling-ready dataset first
    try:
        df = pd.read_csv('automobile_clean.csv')
        print(f"✅ Loaded automobile_clean.csv! Shape: {df.shape}")
    except FileNotFoundError:
        print("❌ automobile_clean.csv not found!")
        print("Please ensure the dataset is in the same directory as this script.")
        sys.exit(1)
        
    # Clean the data for modeling
    print("🧹 Preparing data for modeling...")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    df = df[numeric_cols]
    print(f"✅ Cleaned data shape: {df.shape}")
    
    return df

def prepare_features(df):
    """Prepare features and target variable"""
    print("🎯 Preparing features and target...")
    
    # Check if target column exists
    if 'mpg' not in df.columns:
        print("❌ Target column 'mpg' not found in dataset!")
        print(f"Available columns: {list(df.columns)}")
        # Try to find a suitable target column
        possible_targets = ['mpg', 'price', 'highway-mpg', 'city-mpg']
        target_col = None
        for col in possible_targets:
            if col in df.columns:
                target_col = col
                print(f"✅ Using '{col}' as target variable")
                break
        
        if target_col is None:
            print("❌ No suitable target column found!")
            sys.exit(1)
    else:
        target_col = 'mpg'
    
    # Separate features and target
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    
    print(f"📈 Features shape: {X.shape}")
    print(f"🎯 Target shape: {y.shape}")
    print(f"✅ Feature columns: {list(X.columns)}")
    
    # Verify all features are numeric
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        print(f"⚠️ Non-numeric columns found: {non_numeric}")
        print("🧹 Converting or dropping non-numeric columns...")
        X = X.select_dtypes(include=[np.number])
        print(f"✅ Final features shape: {X.shape}")
    
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into train and test sets"""
    print("✂️ Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    print(f"🚂 Training set: {X_train.shape[0]} samples")
    print(f"🧪 Testing set: {X_test.shape[0]} samples")
    
    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    metrics = {
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'rmse': rmse
    }
    
    return metrics, y_pred

def train_random_forest(X_train, X_test, y_train, y_test):
    """Train Random Forest model with MLflow tracking"""
    print("\n🌲 Training Random Forest Model...")
    
    with mlflow.start_run(run_name="RandomForest_CI"):
        # Enable MLflow autolog
        mlflow.sklearn.autolog()
        
        # Create and train model
        rf_model = RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            max_depth=10
        )
        
        # Train model
        rf_model.fit(X_train, y_train)
        
        # Evaluate model
        metrics, y_pred = evaluate_model(rf_model, X_test, y_test)
        
        # Log additional metrics manually
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("dataset_name", "automobile_dataset")
        mlflow.log_param("training_environment", "GitHub_Actions_CI")
        
        # Print results
        print("📊 Random Forest Results:")
        print(f"   MSE: {metrics['mse']:.4f}")
        print(f"   MAE: {metrics['mae']:.4f}")
        print(f"   R²: {metrics['r2']:.4f}")
        print(f"   RMSE: {metrics['rmse']:.4f}")
        
        return rf_model, metrics

def train_linear_regression(X_train, X_test, y_train, y_test):
    """Train Linear Regression model with MLflow tracking"""
    print("\n📈 Training Linear Regression Model...")
    
    with mlflow.start_run(run_name="LinearRegression_CI"):
        # Enable MLflow autolog
        mlflow.sklearn.autolog()
        
        # Create and train model
        lr_model = LinearRegression()
        
        # Train model
        lr_model.fit(X_train, y_train)
        
        # Evaluate model
        metrics, y_pred = evaluate_model(lr_model, X_test, y_test)
        
        # Log additional metrics manually
        mlflow.log_param("model_type", "LinearRegression")
        mlflow.log_param("dataset_name", "automobile_dataset")
        mlflow.log_param("training_environment", "GitHub_Actions_CI")
        
        # Print results
        print("📊 Linear Regression Results:")
        print(f"   MSE: {metrics['mse']:.4f}")
        print(f"   MAE: {metrics['mae']:.4f}")
        print(f"   R²: {metrics['r2']:.4f}")
        print(f"   RMSE: {metrics['rmse']:.4f}")
        
        return lr_model, metrics

def main():
    """Main function to run the modeling pipeline"""
    parser = argparse.ArgumentParser(description='MLflow Model Training Pipeline')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size (default: 0.2)')
    parser.add_argument('--random_state', type=int, default=42, help='Random state (default: 42)')
    
    args = parser.parse_args()
    
    print("🚀 Starting MLflow Model Training Pipeline (CI/CD)")
    print("=" * 50)
    print(f"⚙️ Parameters: test_size={args.test_size}, random_state={args.random_state}")
    
    # Set MLflow tracking URI (local)
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Auto_Prediction_CI")
    
    try:
        # Load data
        df = load_data()
        
        # Prepare features
        X, y = prepare_features(df)
        
        # Split data
        X_train, X_test, y_train, y_test = split_data(X, y, args.test_size, args.random_state)
        
        # Train models
        print("\n🤖 Training Models with MLflow Tracking...")
        
        # Random Forest
        rf_model, rf_metrics = train_random_forest(X_train, X_test, y_train, y_test)
        
        # Linear Regression
        lr_model, lr_metrics = train_linear_regression(X_train, X_test, y_train, y_test)
        
        # Compare models
        print("\n🏆 Model Comparison:")
        print("=" * 30)
        print(f"Random Forest R²: {rf_metrics['r2']:.4f}")
        print(f"Linear Regression R²: {lr_metrics['r2']:.4f}")
        
        best_model = "Random Forest" if rf_metrics['r2'] > lr_metrics['r2'] else "Linear Regression"
        print(f"🥇 Best Model: {best_model}")
        
        print("\n✅ Training completed successfully!")
        print("📊 MLflow artifacts saved to ./mlruns/")
        print("🤖 CI/CD Pipeline executed successfully!")
        
        # Save summary for CI
        with open('training_summary.txt', 'w') as f:
            f.write("MLflow Model Training Summary\n")
            f.write("=" * 30 + "\n")
            f.write(f"Random Forest R²: {rf_metrics['r2']:.4f}\n")
            f.write(f"Linear Regression R²: {lr_metrics['r2']:.4f}\n")
            f.write(f"Best Model: {best_model}\n")
            f.write(f"Training Environment: GitHub Actions CI\n")
        
        print("📄 Training summary saved to training_summary.txt")
        
    except Exception as e:
        print(f"❌ Error occurred: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()