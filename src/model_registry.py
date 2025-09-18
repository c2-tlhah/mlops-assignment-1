"""
Simple model registry to register only the best model.
"""

import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime


class SimpleModelRegistry:
    """Simple class to register the best model."""
    
    def __init__(self):
        """Initialize the model registry."""
        self.experiment_name = "Iris-Classification-Comparison"
        self.client = MlflowClient()
        
        # Get experiment
        self.experiment = self.client.get_experiment_by_name(self.experiment_name)
        if self.experiment is None:
            print(f"Error: Experiment '{self.experiment_name}' not found")
            return
            
        self.experiment_id = self.experiment.experiment_id
        print(f"Connected to experiment: {self.experiment_name}")
    
    def find_best_model(self):
        """Find the best model based on test accuracy."""
        print("Looking for the best model...")
        
        # Get all runs from the experiment
        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=["start_time DESC"]
        )
        
        best_run = None
        best_accuracy = 0
        
        # Look through runs to find highest test accuracy
        for run in runs:
            metrics = run.data.metrics
            
            # Look for test_accuracy metric
            if 'test_accuracy' in metrics:
                accuracy = metrics['test_accuracy']
                print(f"Run {run.info.run_name}: Accuracy = {accuracy:.4f}")
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_run = run
        
        if best_run:
            print(f"Best model found: {best_run.info.run_name} with accuracy {best_accuracy:.4f}")
            return best_run, best_accuracy
        else:
            print("No model with test_accuracy found")
            return None, 0
    
    def register_best_model(self):
        """Register only the best performing model."""
        print("\n=== Registering Best Model ===")
        
        # Find the best model
        best_run, best_accuracy = self.find_best_model()
        
        if best_run is None:
            print("No model to register")
            return
        
        try:
            # Get model URI from the best run
            model_uri = f"runs:/{best_run.info.run_id}/model"
            
            # Register the model with a simple name
            model_name = "iris-best-classifier"
            
            print(f"Registering model: {model_name}")
            
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=model_name,
                tags={
                    "model_type": best_run.info.run_name.split('_')[0],
                    "best_accuracy": str(best_accuracy),
                    "registration_date": str(datetime.now().date())
                }
            )
            
            print(f"Model registered successfully!")
            print(f"Model name: {model_name}")
            print(f"Version: {model_version.version}")
            print(f"Accuracy: {best_accuracy:.4f}")
            
            # Transition to Production stage
            self.client.transition_model_version_stage(
                name=model_name,
                version=model_version.version,
                stage="Production"
            )
            
            print(f"Model promoted to Production stage!")
            
        except Exception as e:
            print(f"Error registering model: {e}")
    
    def show_registered_models(self):
        """Show all registered models."""
        print("\n=== Registered Models ===")
        
        try:
            registered_models = self.client.search_registered_models()
            
            if not registered_models:
                print("No registered models found")
                return
            
            for model in registered_models:
                print(f"\nModel: {model.name}")
                
                # Get latest version
                latest_version = self.client.get_latest_versions(
                    model.name, stages=["Production", "Staging", "None"]
                )
                
                for version in latest_version:
                    print(f"  Version {version.version}: {version.current_stage}")
                    
                    # Show tags if any
                    if version.tags:
                        for key, value in version.tags.items():
                            print(f"    {key}: {value}")
        
        except Exception as e:
            print(f"Error showing registered models: {e}")


# Simple test and execution
if __name__ == "__main__":
    from datetime import datetime
    
    registry = SimpleModelRegistry()
    
    # Register the best model
    registry.register_best_model()
    
    # Show registered models
    registry.show_registered_models()
    
    print("\nModel registration completed!")
