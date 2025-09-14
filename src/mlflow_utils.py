"""
MLflow configuration and tracking utilities.
"""

import mlflow
import mlflow.sklearn
import mlflow.tracking
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import json
from typing import Dict, Any, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLflowTracker:
    """Class to handle MLflow experiment tracking and logging."""
    
    def __init__(self, experiment_name: str = "MLOps-Assignment-1", tracking_uri: str = None):
        """
        Initialize MLflow tracker.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: URI for MLflow tracking server (None for local)
        """
        self.experiment_name = experiment_name
        
        # Set tracking URI (default to local mlruns directory)
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            mlflow.set_tracking_uri("file:./mlruns")
        
        # Set or create experiment
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                logger.info(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
            else:
                experiment_id = experiment.experiment_id
                logger.info(f"Using existing experiment: {experiment_name} (ID: {experiment_id})")
        except Exception as e:
            logger.error(f"Error setting up experiment: {e}")
            raise
        
        mlflow.set_experiment(experiment_name)
        self.experiment_id = experiment_id
    
    def start_run(self, run_name: str = None) -> mlflow.ActiveRun:
        """
        Start a new MLflow run.
        
        Args:
            run_name: Name for the run
            
        Returns:
            Active MLflow run
        """
        return mlflow.start_run(run_name=run_name)
    
    def log_model_info(self, model: Any, model_name: str, model_params: Dict[str, Any]):
        """
        Log model information to MLflow.
        
        Args:
            model: Trained model
            model_name: Name of the model
            model_params: Model hyperparameters
        """
        # Log model type and name
        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("model_class", model.__class__.__name__)
        
        # Log hyperparameters
        for param_name, param_value in model_params.items():
            # Convert numpy types to Python types for MLflow compatibility
            if isinstance(param_value, np.integer):
                param_value = int(param_value)
            elif isinstance(param_value, np.floating):
                param_value = float(param_value)
            elif param_value is None:
                param_value = "None"
            
            mlflow.log_param(param_name, param_value)
        
        # Log the model itself
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name=f"{model_name.replace('_', '-')}-model"
        )
    
    def log_metrics(self, metrics: Dict[str, float]):
        """
        Log evaluation metrics to MLflow.
        
        Args:
            metrics: Dictionary of metric names and values
        """
        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)
    
    def log_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           class_names: List[str], model_name: str):
        """
        Create and log confusion matrix as artifact.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            model_name: Name of the model
        """
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create figure
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Save and log as artifact
        confusion_matrix_path = f"confusion_matrix_{model_name}.png"
        plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
        mlflow.log_artifact(confusion_matrix_path)
        plt.close()
        
        # Clean up temporary file
        if os.path.exists(confusion_matrix_path):
            os.remove(confusion_matrix_path)
    
    def log_feature_importance(self, model: Any, feature_names: List[str], model_name: str):
        """
        Log feature importance plot if model supports it.
        
        Args:
            model: Trained model
            feature_names: Names of features
            model_name: Name of the model
        """
        if hasattr(model, 'feature_importances_'):
            # Create feature importance plot
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(10, 6))
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
            plt.title(f'Feature Importance - {model_name}')
            plt.tight_layout()
            
            # Save and log as artifact
            feature_importance_path = f"feature_importance_{model_name}.png"
            plt.savefig(feature_importance_path, dpi=300, bbox_inches='tight')
            mlflow.log_artifact(feature_importance_path)
            plt.close()
            
            # Clean up temporary file
            if os.path.exists(feature_importance_path):
                os.remove(feature_importance_path)
    
    def log_classification_report(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                class_names: List[str], model_name: str):
        """
        Log classification report as artifact.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            class_names: Names of classes
            model_name: Name of the model
        """
        # Generate classification report
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
        
        # Save as JSON
        report_path = f"classification_report_{model_name}.json"
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        mlflow.log_artifact(report_path)
        
        # Clean up temporary file
        if os.path.exists(report_path):
            os.remove(report_path)
    
    def log_dataset_info(self, metadata: Dict[str, Any]):
        """
        Log dataset information.
        
        Args:
            metadata: Dataset metadata
        """
        mlflow.set_tag("dataset_name", "iris")
        mlflow.log_param("n_samples", metadata['n_samples'])
        mlflow.log_param("n_features", metadata['n_features'])
        mlflow.log_param("n_classes", metadata['n_classes'])
        
        # Log feature names
        for i, feature_name in enumerate(metadata['feature_names']):
            mlflow.set_tag(f"feature_{i}", feature_name)
    
    def end_run(self):
        """End the current MLflow run."""
        mlflow.end_run()
    
    @staticmethod
    def create_comparison_plot(evaluation_results: Dict[str, Dict[str, Any]]):
        """
        Create a comparison plot of all models.
        
        Args:
            evaluation_results: Dictionary mapping model names to evaluation results
        """
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        model_names = list(evaluation_results.keys())
        
        # Prepare data for plotting
        metric_data = {metric: [] for metric in metrics}
        
        for model_name in model_names:
            for metric in metrics:
                metric_data[metric].append(evaluation_results[model_name][metric])
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics):
            axes[i].bar(model_names, metric_data[metric])
            axes[i].set_title(f'{metric.replace("_", " ").title()} Comparison')
            axes[i].set_ylabel(metric.replace("_", " ").title())
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for j, v in enumerate(metric_data[metric]):
                axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save and log as artifact
        comparison_path = "models_comparison.png"
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        mlflow.log_artifact(comparison_path)
        plt.close()
        
        # Clean up temporary file
        if os.path.exists(comparison_path):
            os.remove(comparison_path)


def setup_mlflow_ui():
    """
    Instructions for setting up MLflow UI.
    """
    print("To view MLflow UI, run the following command in your terminal:")
    print("mlflow ui --backend-store-uri file:./mlruns")
    print("Then open http://localhost:5000 in your browser")


def main():
    """Test MLflow setup."""
    tracker = MLflowTracker()
    
    with tracker.start_run(run_name="test_run"):
        # Test logging
        tracker.log_metrics({"test_accuracy": 0.95, "test_precision": 0.94})
        mlflow.log_param("test_param", "test_value")
        logger.info("Test run completed successfully")
    
    setup_mlflow_ui()


if __name__ == "__main__":
    main()