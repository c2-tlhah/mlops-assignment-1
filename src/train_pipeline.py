"""
Simple training pipeline that puts everything together.
"""

from .data_loader import DataLoader
from .models import ModelTrainer, ModelEvaluator
from .mlflow_utils import MLflowTracker


class SimpleTrainingPipeline:
    """Simple class to run the complete ML pipeline."""
    
    def __init__(self):
        """Initialize all the components we need."""
        print("Setting up MLOps pipeline...")
        self.data_loader = DataLoader()
        self.trainer = ModelTrainer()
        self.evaluator = ModelEvaluator()
        self.mlflow_tracker = MLflowTracker()
        
    def run_pipeline(self):
        """Run the complete pipeline from data loading to MLflow tracking."""
        print("\n=== Starting MLOps Pipeline ===")
        
        # Step 1: Load and prepare data
        print("\nStep 1: Loading data...")
        X_train, X_test, y_train, y_test = self.data_loader.get_data()
        
        # Step 2: Train all models
        print("\nStep 2: Training models...")
        trained_models = self.trainer.train_all_models(X_train, y_train)
        
        # Step 3: Log training to MLflow
        print("\nStep 3: Logging training to MLflow...")
        for model_name, model_info in trained_models.items():
            model = model_info['model']
            cv_score = model_info['cv_score']
            
            # Get model parameters
            params = model.get_params()
            
            # Log to MLflow
            self.mlflow_tracker.log_model_training(model, model_name, cv_score, params)
        
        # Step 4: Evaluate models
        print("\nStep 4: Evaluating models...")
        evaluation_results = {}
        
        for model_name, model_info in trained_models.items():
            model = model_info['model']
            
            # Evaluate the model
            results = self.evaluator.evaluate_model(model, X_test, y_test, model_name)
            evaluation_results[model_name] = results
            
            # Log evaluation to MLflow
            y_pred = model.predict(X_test)
            self.mlflow_tracker.log_model_evaluation(model, model_name, results, y_test, y_pred)
        
        # Step 5: Compare models and log comparison
        print("\nStep 5: Comparing models...")
        best_model, best_accuracy = self.evaluator.compare_models(evaluation_results)
        
        # Log comparison to MLflow
        self.mlflow_tracker.log_model_comparison(evaluation_results)
        
        print(f"\n=== Pipeline Complete ===")
        print(f"Best Model: {best_model}")
        print(f"Best Accuracy: {best_accuracy:.4f}")
        print(f"Check MLflow UI: mlflow ui --backend-store-uri file:./mlruns")
        
        return evaluation_results, best_model


# Simple test of the pipeline
if __name__ == "__main__":
    pipeline = SimpleTrainingPipeline()
    results, best_model = pipeline.run_pipeline()
    print(f"\nPipeline completed! Best model: {best_model}")
