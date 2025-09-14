"""
Machine Learning models for the MLOps assignment.
Implements training for multiple classifiers with hyperparameter tuning.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib
import os
from typing import Dict, Any, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Class to handle training of multiple ML models."""
    
    def __init__(self, models_dir: str = "models"):
        """
        Initialize the ModelTrainer.
        
        Args:
            models_dir: Directory to save trained models
        """
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        # Define models and their hyperparameter grids
        self.models_config = {
            'logistic_regression': {
                'model': LogisticRegression(random_state=42, max_iter=1000),
                'params': {
                    'C': [0.1, 1.0, 10.0, 100.0],
                    'solver': ['liblinear', 'lbfgs'],
                    'penalty': ['l2']
                }
            },
            'random_forest': {
                'model': RandomForestClassifier(random_state=42),
                'params': {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 5, 7, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
            },
            'svm': {
                'model': SVC(random_state=42, probability=True),
                'params': {
                    'C': [0.1, 1.0, 10.0, 100.0],
                    'kernel': ['linear', 'rbf', 'poly'],
                    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1.0]
                }
            }
        }
    
    def train_model(self, model_name: str, X_train: np.ndarray, y_train: np.ndarray, 
                   cv_folds: int = 5) -> Tuple[Any, Dict[str, Any]]:
        """
        Train a single model with hyperparameter tuning.
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training targets
            cv_folds: Number of cross-validation folds
            
        Returns:
            Tuple of (best_model, training_info)
        """
        if model_name not in self.models_config:
            raise ValueError(f"Model '{model_name}' not supported")
        
        logger.info(f"Training {model_name}...")
        
        config = self.models_config[model_name]
        
        # Perform grid search with cross-validation
        grid_search = GridSearchCV(
            estimator=config['model'],
            param_grid=config['params'],
            cv=cv_folds,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # Get best model and parameters
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        
        # Save the model
        model_path = os.path.join(self.models_dir, f"{model_name}_best.pkl")
        joblib.dump(best_model, model_path)
        
        training_info = {
            'model_name': model_name,
            'best_params': best_params,
            'best_cv_score': best_score,
            'model_path': model_path,
            'cv_folds': cv_folds
        }
        
        logger.info(f"{model_name} training completed. Best CV score: {best_score:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        return best_model, training_info
    
    def train_all_models(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Tuple[Any, Dict[str, Any]]]:
        """
        Train all configured models.
        
        Args:
            X_train: Training features
            y_train: Training targets
            
        Returns:
            Dictionary mapping model names to (model, training_info) tuples
        """
        trained_models = {}
        
        for model_name in self.models_config.keys():
            model, info = self.train_model(model_name, X_train, y_train)
            trained_models[model_name] = (model, info)
        
        return trained_models
    
    def load_model(self, model_name: str) -> Any:
        """
        Load a previously trained model.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            Loaded model
        """
        model_path = os.path.join(self.models_dir, f"{model_name}_best.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        return joblib.load(model_path)


class ModelEvaluator:
    """Class to handle evaluation of trained models."""
    
    @staticmethod
    def evaluate_model(model: Any, X_test: np.ndarray, y_test: np.ndarray, 
                      model_name: str) -> Dict[str, Any]:
        """
        Evaluate a trained model on test data.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            model_name: Name of the model
            
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info(f"Evaluating {model_name}...")
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Generate classification report
        class_report = classification_report(y_test, y_pred, output_dict=True)
        
        evaluation_results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'predictions': y_pred,
            'prediction_probabilities': y_pred_proba,
            'classification_report': class_report
        }
        
        logger.info(f"{model_name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, "
                   f"Recall: {recall:.4f}, F1: {f1:.4f}")
        
        return evaluation_results
    
    @staticmethod
    def compare_models(evaluation_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Compare multiple models based on evaluation metrics.
        
        Args:
            evaluation_results: Dictionary mapping model names to evaluation results
            
        Returns:
            Comparison summary
        """
        logger.info("Comparing models...")
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        comparison = {}
        
        for metric in metrics:
            comparison[metric] = {}
            scores = {name: results[metric] for name, results in evaluation_results.items()}
            
            # Find best model for this metric
            best_model = max(scores.items(), key=lambda x: x[1])
            comparison[metric]['best_model'] = best_model[0]
            comparison[metric]['best_score'] = best_model[1]
            comparison[metric]['all_scores'] = scores
        
        # Overall best model (based on accuracy)
        comparison['overall_best'] = comparison['accuracy']['best_model']
        
        return comparison


def main():
    """Test the model training functionality."""
    from data_loader import DataLoader
    
    # Load data
    loader = DataLoader()
    X_train, X_test, y_train, y_test, metadata = loader.get_data()
    
    # Train models
    trainer = ModelTrainer()
    trained_models = trainer.train_all_models(X_train, y_train)
    
    # Evaluate models
    evaluator = ModelEvaluator()
    evaluation_results = {}
    
    for model_name, (model, training_info) in trained_models.items():
        eval_results = evaluator.evaluate_model(model, X_test, y_test, model_name)
        evaluation_results[model_name] = eval_results
    
    # Compare models
    comparison = evaluator.compare_models(evaluation_results)
    print(f"\nBest overall model: {comparison['overall_best']}")
    print(f"Best accuracy: {comparison['accuracy']['best_score']:.4f}")


if __name__ == "__main__":
    main()