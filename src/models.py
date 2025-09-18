"""
Simple ML models for the MLOps assignment.
Trains three different models: Logistic Regression, Random Forest, and SVM.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib
import os


class ModelTrainer:
    """Simple class to train different machine learning models."""
    
    def __init__(self):
        """Initialize the trainer and create models directory."""
        self.models_dir = "models"
        os.makedirs(self.models_dir, exist_ok=True)
    
    def train_logistic_regression(self, X_train, y_train):
        """Train a Logistic Regression model."""
        print("Training Logistic Regression...")
        
        # Create and train the model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        # Evaluate with cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        avg_score = cv_scores.mean()
        
        # Save the model
        model_path = os.path.join(self.models_dir, "logistic_regression.pkl")
        joblib.dump(model, model_path)
        
        print(f"Logistic Regression - CV Score: {avg_score:.4f}")
        return model, avg_score
    
    def train_random_forest(self, X_train, y_train):
        """Train a Random Forest model."""
        print("Training Random Forest...")
        
        # Create and train the model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate with cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        avg_score = cv_scores.mean()
        
        # Save the model
        model_path = os.path.join(self.models_dir, "random_forest.pkl")
        joblib.dump(model, model_path)
        
        print(f"Random Forest - CV Score: {avg_score:.4f}")
        return model, avg_score
    
    def train_svm(self, X_train, y_train):
        """Train an SVM model."""
        print("Training SVM...")
        
        # Create and train the model
        model = SVC(random_state=42)
        model.fit(X_train, y_train)
        
        # Evaluate with cross-validation
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        avg_score = cv_scores.mean()
        
        # Save the model
        model_path = os.path.join(self.models_dir, "svm.pkl")
        joblib.dump(model, model_path)
        
        print(f"SVM - CV Score: {avg_score:.4f}")
        return model, avg_score
    
    def train_all_models(self, X_train, y_train):
        """Train all three models and return them with their scores."""
        print("Training all models...")
        
        # Train each model
        lr_model, lr_score = self.train_logistic_regression(X_train, y_train)
        rf_model, rf_score = self.train_random_forest(X_train, y_train)
        svm_model, svm_score = self.train_svm(X_train, y_train)
        
        # Store results
        results = {
            'logistic_regression': {'model': lr_model, 'cv_score': lr_score},
            'random_forest': {'model': rf_model, 'cv_score': rf_score},
            'svm': {'model': svm_model, 'cv_score': svm_score}
        }
        
        print("All models trained successfully!")
        return results


class ModelEvaluator:
    """Simple class to evaluate trained models."""
    
    def evaluate_model(self, model, X_test, y_test, model_name):
        """Evaluate a model and return performance metrics."""
        print(f"Evaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        print(f"{model_name} Results:")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  F1-Score: {f1:.4f}")
        
        return results
    
    def compare_models(self, evaluation_results):
        """Compare all models and find the best one."""
        print("\nComparing all models:")
        
        best_model = None
        best_accuracy = 0
        
        for model_name, results in evaluation_results.items():
            accuracy = results['accuracy']
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model = model_name
        
        print(f"Best model: {best_model} with accuracy: {best_accuracy:.4f}")
        return best_model, best_accuracy


# Simple test of the models
if __name__ == "__main__":
    from data_loader import DataLoader
    
    # Load data
    loader = DataLoader()
    X_train, X_test, y_train, y_test = loader.get_data()
    
    # Train models
    trainer = ModelTrainer()
    trained_models = trainer.train_all_models(X_train, y_train)
    
    # Evaluate models
    evaluator = ModelEvaluator()
    evaluation_results = {}
    
    for model_name, model_info in trained_models.items():
        model = model_info['model']
        results = evaluator.evaluate_model(model, X_test, y_test, model_name)
        evaluation_results[model_name] = results
    
    # Find best model
    best_model, best_accuracy = evaluator.compare_models(evaluation_results)
    print(f"\nTraining and evaluation completed!")