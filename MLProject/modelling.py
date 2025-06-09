import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import mlflow
import mlflow.sklearn
import logging
import os
import argparse
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log'),
        logging.StreamHandler()
    ]
)

def parse_args():
    """
    Parse command line arguments for MLflow Project
    """
    parser = argparse.ArgumentParser(description='Train Palmer Penguins Classification Models')
    
    # General parameters
    parser.add_argument('--test_size', type=float, default=0.2, help='Test set size')
    parser.add_argument('--random_state', type=int, default=42, help='Random state')
    parser.add_argument('--model', type=str, default='all', 
                       choices=['all', 'random_forest', 'logistic_regression', 'svm'],
                       help='Model to train')
    
    # Random Forest parameters
    parser.add_argument('--n_estimators', type=int, default=100, help='Number of estimators for RF')
    parser.add_argument('--max_depth', type=int, default=10, help='Max depth for RF')
    
    # Logistic Regression parameters
    parser.add_argument('--max_iter', type=int, default=1000, help='Max iterations for LR')
    
    # SVM parameters
    parser.add_argument('--C', type=float, default=1.0, help='Regularization parameter for SVM')
    parser.add_argument('--kernel', type=str, default='rbf', help='Kernel for SVM')
    
    return parser.parse_args()

def load_and_prepare_data():
    """
    Load preprocessed data and prepare for modeling
    """
    try:
        # Load preprocessed data
        df = pd.read_csv('penguins_processed.csv')
        logging.info(f"Data loaded successfully. Shape: {df.shape}")
        
        # Define features and target
        feature_columns = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 
                          'body_mass_g', 'island_encoded', 'sex_encoded']
        
        X = df[feature_columns]
        y = df['species_encoded']
        
        logging.info(f"Features shape: {X.shape}")
        logging.info(f"Target shape: {y.shape}")
        logging.info(f"Target distribution: {y.value_counts().to_dict()}")
        
        return X, y
        
    except Exception as e:
        logging.error(f"Error loading data: {str(e)}")
        raise

def get_model(model_name, args):
    """
    Get model instance based on name and arguments
    """
    if model_name == "random_forest":
        return RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=args.max_depth,
            random_state=args.random_state
        )
    elif model_name == "logistic_regression":
        return LogisticRegression(
            max_iter=args.max_iter,
            random_state=args.random_state
        )
    elif model_name == "svm":
        return SVC(
            C=args.C,
            kernel=args.kernel,
            random_state=args.random_state
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")

def train_model(model, model_name, X_train, X_test, y_train, y_test, args):
    """
    Train model with MLflow tracking
    """
    with mlflow.start_run(run_name=model_name):
        # Enable autolog
        mlflow.sklearn.autolog()
        
        logging.info(f"Training {model_name}...")
        
        # Log parameters from args
        mlflow.log_param("test_size", args.test_size)
        mlflow.log_param("random_state", args.random_state)
        
        if model_name == "random_forest":
            mlflow.log_param("n_estimators", args.n_estimators)
            mlflow.log_param("max_depth", args.max_depth)
        elif model_name == "logistic_regression":
            mlflow.log_param("max_iter", args.max_iter)
        elif model_name == "svm":
            mlflow.log_param("C", args.C)
            mlflow.log_param("kernel", args.kernel)
        
        # Train model
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Log additional metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_param("model_type", model_name)
        
        # Log model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=f"{model_name}_penguins"
        )
        
        logging.info(f"{model_name} - Accuracy: {accuracy:.4f}")
        
        # Print classification report
        print(f"\n{model_name} Classification Report:")
        print(classification_report(y_test, y_pred))
        
        return model, accuracy

def main():
    """
    Main function to execute the modeling pipeline
    """
    # Parse arguments
    args = parse_args()
    
    # Set MLflow tracking URI
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
    mlflow.set_experiment("Palmer_Penguins_Classification_CI")
    
    logging.info("Starting Palmer Penguins Classification Model Training (CI)")
    logging.info(f"Arguments: {vars(args)}")
    
    try:
        # Load and prepare data
        X, y = load_and_prepare_data()
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=args.test_size, random_state=args.random_state, stratify=y
        )
        
        logging.info(f"Train set shape: {X_train.shape}")
        logging.info(f"Test set shape: {X_test.shape}")
        
        # Determine which models to train
        if args.model == 'all':
            models_to_train = ['random_forest', 'logistic_regression', 'svm']
        else:
            models_to_train = [args.model]
        
        # Train models
        results = {}
        best_accuracy = 0
        best_model_name = ""
        
        for model_name in models_to_train:
            model = get_model(model_name, args)
            trained_model, accuracy = train_model(
                model, model_name, X_train, X_test, y_train, y_test, args
            )
            results[model_name] = {
                'model': trained_model,
                'accuracy': accuracy
            }
            
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_name = model_name
        
        # Log best model information
        logging.info(f"\nBest Model: {best_model_name}")
        logging.info(f"Best Accuracy: {best_accuracy:.4f}")
        
        # Print results summary
        print("\n" + "="*50)
        print("MODEL TRAINING RESULTS SUMMARY")
        print("="*50)
        
        for model_name, result in results.items():
            print(f"{model_name}: {result['accuracy']:.4f}")
        
        print(f"\nBest Model: {best_model_name} (Accuracy: {best_accuracy:.4f})")
        print("="*50)
        
        logging.info("Model training completed successfully!")
        
    except Exception as e:
        logging.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main()