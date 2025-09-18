"""
Simple MLflow tracking for the assignment.
"""

import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np


class MLflowTracker:
    """Simple class to track experiments with MLflow."""
    
    def __init__(self):
        """Initialize MLflow tracking."""
        self.experiment_name = "Iris-Classification-Comparison"
        
        # Set up MLflow to save locally
        mlflow.set_tracking_uri("file:./mlruns")
        
        # Create or get experiment
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            mlflow.create_experiment(self.experiment_name)
            print(f"Created experiment: {self.experiment_name}")
        
        mlflow.set_experiment(self.experiment_name)
    
    def log_model_training(self, model, model_name, cv_score, params):
        """Log model training information to MLflow."""
        with mlflow.start_run(run_name=f"{model_name}_training"):
            # Log parameters
            mlflow.log_params(params)
            
            # Log cross-validation score
            mlflow.log_metric("cv_score", cv_score)
            mlflow.log_metric("cv_folds", 5)
            
            # Log the trained model
            mlflow.sklearn.log_model(model, "model")
            
            print(f"Logged {model_name} training to MLflow")
    
    def log_model_evaluation(self, model, model_name, metrics, y_true, y_pred):
        """Log model evaluation results to MLflow and save to results folder."""
        with mlflow.start_run(run_name=f"{model_name}_evaluation"):
            # Log evaluation metrics
            mlflow.log_metric("test_accuracy", metrics['accuracy'])
            mlflow.log_metric("test_precision", metrics['precision'])
            mlflow.log_metric("test_recall", metrics['recall'])
            mlflow.log_metric("test_f1_score", metrics['f1_score'])
            
            # Save metrics to results folder
            self.save_metrics_to_results(model_name, metrics)
            
            # Create and log confusion matrix
            self.log_confusion_matrix(y_true, y_pred, model_name)
            
            print(f"Logged {model_name} evaluation to MLflow")
    
    def save_metrics_to_results(self, model_name, metrics):
        """Save evaluation metrics to results folder."""
        import os
        import json
        
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        # Save metrics as JSON
        results_file = f"results/{model_name}_metrics.json"
        with open(results_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"Saved metrics to: {results_file}")
    
    def log_confusion_matrix(self, y_true, y_pred, model_name):
        """Create and log confusion matrix plot."""
        import os
        
        # Create confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create plot
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation='nearest', cmap='Blues')
        plt.title(f'Confusion Matrix - {model_name}')
        plt.colorbar()
        
        # Add labels
        classes = ['Setosa', 'Versicolor', 'Virginica']
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        
        # Add text annotations
        for i in range(len(classes)):
            for j in range(len(classes)):
                plt.text(j, i, cm[i, j], ha="center", va="center")
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        # Save to results folder AND log to MLflow
        results_filename = f"results/confusion_matrix_{model_name}.png"
        mlflow_filename = f"confusion_matrix_{model_name}.png"
        
        # Save to results folder
        plt.savefig(results_filename)
        print(f"Saved confusion matrix to: {results_filename}")
        
        # Save temporary file for MLflow
        plt.savefig(mlflow_filename)
        mlflow.log_artifact(mlflow_filename)
        
        plt.close()
        
        # Clean up temporary MLflow file (keep the results folder file)
        if os.path.exists(mlflow_filename):
            os.remove(mlflow_filename)
    
    def log_model_comparison(self, evaluation_results):
        """Log overall model comparison and save to results folder."""
        with mlflow.start_run(run_name="models_comparison"):
            # Log best model metrics
            best_accuracy = 0
            best_model = None
            
            for model_name, results in evaluation_results.items():
                if results['accuracy'] > best_accuracy:
                    best_accuracy = results['accuracy']
                    best_model = model_name
            
            mlflow.log_metric("best_accuracy", best_accuracy)
            mlflow.set_tag("best_model", best_model)
            
            # Save comparison summary to results folder
            self.save_comparison_to_results(evaluation_results, best_model, best_accuracy)
            
            # Create comparison plot
            self.create_comparison_plot(evaluation_results)
            
            print("Logged model comparison to MLflow")
    
    def save_comparison_to_results(self, evaluation_results, best_model, best_accuracy):
        """Save model comparison summary to results folder."""
        import os
        import json
        
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        # Create summary
        summary = {
            "best_model": best_model,
            "best_accuracy": best_accuracy,
            "all_results": evaluation_results,
            "model_ranking": sorted(evaluation_results.items(), 
                                  key=lambda x: x[1]['accuracy'], 
                                  reverse=True)
        }
        
        # Save summary as JSON
        summary_file = "results/model_comparison_summary.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Saved comparison summary to: {summary_file}")
    
    def create_comparison_plot(self, evaluation_results):
        """Create a comparison plot of all models."""
        import os
        
        models = list(evaluation_results.keys())
        accuracies = [evaluation_results[model]['accuracy'] for model in models]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(models, accuracies, color=['skyblue', 'lightgreen', 'lightcoral'])
        plt.title('Model Performance Comparison')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1.1)
        
        # Add value labels on bars
        for bar, accuracy in zip(bars, accuracies):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{accuracy:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Create results directory if it doesn't exist
        os.makedirs('results', exist_ok=True)
        
        # Save to results folder AND log to MLflow
        results_filename = "results/models_comparison.png"
        mlflow_filename = "models_comparison.png"
        
        # Save to results folder
        plt.savefig(results_filename)
        print(f"Saved comparison plot to: {results_filename}")
        
        # Save temporary file for MLflow
        plt.savefig(mlflow_filename)
        mlflow.log_artifact(mlflow_filename)
        
        plt.close()
        
        # Clean up temporary MLflow file
        if os.path.exists(mlflow_filename):
            os.remove(mlflow_filename)


# Simple test
if __name__ == "__main__":
    tracker = MLflowTracker()
    print("MLflow tracker initialized successfully!")
