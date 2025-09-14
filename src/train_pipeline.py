"""
Main training pipeline that integrates data loading, model training, evaluation, and MLflow tracking.
"""

import os
import sys
import logging
from typing import Dict, Any
import warnings
import mlflow

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

from data_loader import DataLoader
from models import ModelTrainer, ModelEvaluator
from mlflow_utils import MLflowTracker
from model_registry import MLflowMonitor

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class MLOpsTrainingPipeline:
    """Complete MLOps training pipeline with MLflow tracking."""
    
    def __init__(self, experiment_name: str = "Iris-Classification-Comparison"):
        """
        Initialize the training pipeline.
        
        Args:
            experiment_name: Name for the MLflow experiment
        """
        self.experiment_name = experiment_name
        self.tracker = MLflowTracker(experiment_name)
        self.data_loader = DataLoader()
        self.model_trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()
        
        logger.info(f"Initialized MLOps pipeline for experiment: {experiment_name}")
    
    def run_complete_pipeline(self):
        """Run the complete training and evaluation pipeline."""
        logger.info("Starting complete MLOps training pipeline...")
        
        # Load and prepare data
        logger.info("Loading and preparing data...")
        X_train, X_test, y_train, y_test, metadata = self.data_loader.get_data()
        
        # Train all models with MLflow tracking
        trained_models = self.train_models_with_tracking(X_train, y_train, metadata)
        
        # Evaluate all models with MLflow tracking
        evaluation_results = self.evaluate_models_with_tracking(
            trained_models, X_test, y_test, metadata
        )
        
        # Create and log comparison artifacts
        self.create_comparison_artifacts(evaluation_results)
        
        # Print summary
        self.print_summary(evaluation_results)
        
        # Monitor experiments and register best model
        self.monitor_and_register_best_model()
        
        logger.info("Complete MLOps pipeline finished successfully!")
        
        return evaluation_results
    
    def train_models_with_tracking(self, X_train, y_train, metadata):
        """Train all models with MLflow tracking."""
        logger.info("Training models with MLflow tracking...")
        
        trained_models = {}
        
        for model_name in self.model_trainer.models_config.keys():
            logger.info(f"Training {model_name} with MLflow tracking...")
            
            with self.tracker.start_run(run_name=f"{model_name}_training"):
                try:
                    # Log dataset information
                    self.tracker.log_dataset_info(metadata)
                    
                    # Train model
                    model, training_info = self.model_trainer.train_model(
                        model_name, X_train, y_train
                    )
                    
                    # Log model information to MLflow
                    self.tracker.log_model_info(
                        model, model_name, training_info['best_params']
                    )
                    
                    # Log training metrics
                    self.tracker.log_metrics({
                        'cv_score': training_info['best_cv_score'],
                        'cv_folds': training_info['cv_folds']
                    })
                    
                    # Log feature importance if available
                    self.tracker.log_feature_importance(
                        model, metadata['feature_names'], model_name
                    )
                    
                    trained_models[model_name] = (model, training_info)
                    
                    logger.info(f"Completed training {model_name} with MLflow tracking")
                    
                except Exception as e:
                    logger.error(f"Error training {model_name}: {e}")
                    raise
                finally:
                    self.tracker.end_run()
        
        return trained_models
    
    def evaluate_models_with_tracking(self, trained_models, X_test, y_test, metadata):
        """Evaluate all models with MLflow tracking."""
        logger.info("Evaluating models with MLflow tracking...")
        
        evaluation_results = {}
        
        for model_name, (model, training_info) in trained_models.items():
            logger.info(f"Evaluating {model_name} with MLflow tracking...")
            
            with self.tracker.start_run(run_name=f"{model_name}_evaluation"):
                try:
                    # Log dataset information
                    self.tracker.log_dataset_info(metadata)
                    
                    # Evaluate model
                    eval_results = self.evaluator.evaluate_model(
                        model, X_test, y_test, model_name
                    )
                    
                    # Log evaluation metrics
                    metrics_to_log = {
                        'test_accuracy': eval_results['accuracy'],
                        'test_precision': eval_results['precision'],
                        'test_recall': eval_results['recall'],
                        'test_f1_score': eval_results['f1_score']
                    }
                    self.tracker.log_metrics(metrics_to_log)
                    
                    # Log model parameters from training
                    for param_name, param_value in training_info['best_params'].items():
                        # Convert numpy types to Python types for MLflow compatibility
                        if hasattr(param_value, 'item'):
                            param_value = param_value.item()
                        elif param_value is None:
                            param_value = "None"
                        self.tracker.log_metrics({f"param_{param_name}": hash(str(param_value)) % 1000 / 1000.0})
                    
                    # Create and log confusion matrix
                    self.tracker.log_confusion_matrix(
                        y_test, eval_results['predictions'], 
                        metadata['target_names'], model_name
                    )
                    
                    # Log classification report
                    self.tracker.log_classification_report(
                        y_test, eval_results['predictions'],
                        metadata['target_names'], model_name
                    )
                    
                    evaluation_results[model_name] = eval_results
                    
                    logger.info(f"Completed evaluation {model_name} with MLflow tracking")
                    
                except Exception as e:
                    logger.error(f"Error evaluating {model_name}: {e}")
                    raise
                finally:
                    self.tracker.end_run()
        
        return evaluation_results
    
    def create_comparison_artifacts(self, evaluation_results):
        """Create and log comparison artifacts."""
        logger.info("Creating comparison artifacts...")
        
        with self.tracker.start_run(run_name="models_comparison"):
            try:
                # Create comparison plot
                self.tracker.create_comparison_plot(evaluation_results)
                
                # Log overall comparison metrics
                comparison = self.evaluator.compare_models(evaluation_results)
                
                # Log best model information
                self.tracker.log_metrics({
                    'best_accuracy': comparison['accuracy']['best_score'],
                    'best_precision': comparison['precision']['best_score'],
                    'best_recall': comparison['recall']['best_score'],
                    'best_f1_score': comparison['f1_score']['best_score']
                })
                
                # Tag the best overall model
                mlflow.set_tag("best_overall_model", comparison['overall_best'])
                
                logger.info("Comparison artifacts created successfully")
                
            except Exception as e:
                logger.error(f"Error creating comparison artifacts: {e}")
                raise
            finally:
                self.tracker.end_run()
    
    def print_summary(self, evaluation_results):
        """Print a summary of all results."""
        print("\n" + "="*80)
        print("MLOPS TRAINING PIPELINE SUMMARY")
        print("="*80)
        
        print(f"\nExperiment: {self.experiment_name}")
        print(f"Models trained: {len(evaluation_results)}")
        
        print("\nModel Performance Summary:")
        print("-" * 80)
        print(f"{'Model':<20} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
        print("-" * 80)
        
        for model_name, results in evaluation_results.items():
            print(f"{model_name:<20} {results['accuracy']:<12.4f} {results['precision']:<12.4f} "
                  f"{results['recall']:<12.4f} {results['f1_score']:<12.4f}")
        
        # Find best model
        best_model = max(evaluation_results.items(), key=lambda x: x[1]['accuracy'])
        print("-" * 80)
        print(f"Best Model: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.4f})")
        
        print("\nMLflow Tracking:")
        print("-" * 40)
        print("To view detailed results and comparisons:")
        print("1. Run: mlflow ui --backend-store-uri file:./mlruns")
        print("2. Open: http://localhost:5000")
        print("3. Navigate to your experiment to see all runs and artifacts")
        
        print("\nFiles saved:")
        print("-" * 40)
        print("- Trained models: ./models/")
        print("- MLflow runs: ./mlruns/")
        print("- Confusion matrices, feature importance plots, and classification reports logged as artifacts")
        
        print("\n" + "="*80)
    
    def monitor_and_register_best_model(self):
        """Monitor experiments and register the best performing model."""
        logger.info("Starting model monitoring and registration...")
        
        try:
            # Initialize monitor
            monitor = MLflowMonitor(self.experiment_name)
            
            # Print monitoring summary
            monitor.print_monitoring_summary()
            
            # Register best model
            logger.info("Registering best performing model in MLflow Model Registry...")
            registration_info = monitor.register_best_model(
                metric='test_accuracy',
                model_name='iris-best-classifier'
            )
            
            # Promote to production
            logger.info("Promoting best model to production stage...")
            monitor.promote_model_to_production(
                registration_info['model_name'],
                registration_info['version']
            )
            
            print(f"\n🎯 MODEL REGISTRATION SUMMARY:")
            print(f"{'='*50}")
            print(f"Model Name: {registration_info['model_name']}")
            print(f"Version: {registration_info['version']}")
            print(f"Model Type: {registration_info['model_type']}")
            print(f"Best Accuracy: {registration_info['best_score']:.4f}")
            print(f"Stage: Production")
            print(f"Registration Date: {registration_info['registration_date']}")
            print(f"{'='*50}")
            
            logger.info("Model monitoring and registration completed successfully!")
            
        except Exception as e:
            logger.error(f"Model monitoring/registration failed: {e}")
            # Don't raise the exception to avoid breaking the pipeline
            print(f"⚠️  Model registration failed: {e}")


def main():
    """Run the complete MLOps training pipeline."""
    try:
        pipeline = MLOpsTrainingPipeline()
        results = pipeline.run_complete_pipeline()
        return results
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()