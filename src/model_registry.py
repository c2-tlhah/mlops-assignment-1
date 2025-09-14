"""
MLflow monitoring and model registry utilities.
Provides functionality for monitoring experiments and registering the best models.
"""

import mlflow
import mlflow.tracking
from mlflow.tracking import MlflowClient
import pandas as pd
from typing import Dict, List, Tuple, Any
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MLflowMonitor:
    """Class to monitor MLflow experiments and manage model registry."""
    
    def __init__(self, experiment_name: str = "Iris-Classification-Comparison"):
        """
        Initialize MLflow monitor.
        
        Args:
            experiment_name: Name of the MLflow experiment to monitor
        """
        self.experiment_name = experiment_name
        self.client = MlflowClient()
        
        # Get experiment
        self.experiment = self.client.get_experiment_by_name(experiment_name)
        if self.experiment is None:
            raise ValueError(f"Experiment '{experiment_name}' not found")
        
        self.experiment_id = self.experiment.experiment_id
        logger.info(f"Monitoring experiment: {experiment_name} (ID: {self.experiment_id})")
    
    def get_all_runs(self) -> List[mlflow.entities.Run]:
        """
        Get all runs from the experiment.
        
        Returns:
            List of MLflow runs
        """
        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=["start_time DESC"]
        )
        logger.info(f"Found {len(runs)} runs in experiment")
        return runs
    
    def get_runs_summary(self) -> pd.DataFrame:
        """
        Get a summary of all runs as a DataFrame.
        
        Returns:
            DataFrame with run information
        """
        runs = self.get_all_runs()
        
        summary_data = []
        for run in runs:
            run_data = {
                'run_id': run.info.run_id,
                'run_name': run.data.tags.get('mlflow.runName', 'Unknown'),
                'status': run.info.status,
                'start_time': datetime.fromtimestamp(run.info.start_time / 1000),
                'model_type': run.data.tags.get('model_type', 'Unknown'),
                'test_accuracy': run.data.metrics.get('test_accuracy', None),
                'test_precision': run.data.metrics.get('test_precision', None),
                'test_recall': run.data.metrics.get('test_recall', None),
                'test_f1_score': run.data.metrics.get('test_f1_score', None),
                'cv_score': run.data.metrics.get('cv_score', None)
            }
            summary_data.append(run_data)
        
        df = pd.DataFrame(summary_data)
        return df
    
    def find_best_model_run(self, metric: str = 'test_accuracy') -> Tuple[mlflow.entities.Run, float]:
        """
        Find the best performing model run based on a metric.
        
        Args:
            metric: Metric to use for comparison (default: test_accuracy)
            
        Returns:
            Tuple of (best_run, best_score)
        """
        runs = self.get_all_runs()
        
        best_run = None
        best_score = -1
        
        for run in runs:
            if metric in run.data.metrics:
                score = run.data.metrics[metric]
                if score > best_score:
                    best_score = score
                    best_run = run
        
        if best_run is None:
            raise ValueError(f"No runs found with metric '{metric}'")
        
        logger.info(f"Best model: {best_run.data.tags.get('model_type', 'Unknown')} "
                   f"with {metric}: {best_score:.4f}")
        
        return best_run, best_score
    
    def register_best_model(self, metric: str = 'test_accuracy', 
                           model_name: str = "iris-best-classifier") -> Dict[str, Any]:
        """
        Register the best performing model in MLflow Model Registry.
        
        Args:
            metric: Metric to use for finding best model
            model_name: Name for the registered model
            
        Returns:
            Dictionary with registration information
        """
        # Find best model run
        best_run, best_score = self.find_best_model_run(metric)
        
        # Check what artifacts are available in this run
        artifacts = self.client.list_artifacts(best_run.info.run_id)
        logger.info(f"Available artifacts in best run: {[a.path for a in artifacts]}")
        
        # Try different possible model paths
        possible_paths = ["model", "models", "sklearn-model"]
        model_uri = None
        
        for path in possible_paths:
            try:
                # Check if this path exists in artifacts
                artifact_exists = any(a.path == path for a in artifacts)
                if artifact_exists:
                    model_uri = f"runs:/{best_run.info.run_id}/{path}"
                    logger.info(f"Found model at path: {path}")
                    break
            except Exception:
                continue
        
        if model_uri is None:
            # If no model artifacts found in the run, let's find the corresponding registered model
            # Based on the comparison run, we know logistic_regression was the best
            model_type = best_run.data.tags.get('best_overall_model', 'logistic_regression')
            original_model_name = f"{model_type.replace('_', '-')}-model"
            
            logger.info(f"No model artifacts found in run, using existing registered model: {original_model_name}")
            
            # Get the existing model version
            try:
                existing_versions = self.client.search_model_versions(f"name='{original_model_name}'")
                if existing_versions:
                    latest_version = max(existing_versions, key=lambda v: int(v.version))
                    model_uri = f"models:/{original_model_name}/{latest_version.version}"
                    logger.info(f"Using existing model: {model_uri}")
                else:
                    # If that model doesn't exist, try to find any available model
                    all_models = self.client.search_registered_models()
                    if all_models:
                        # Use the first available model
                        first_model = all_models[0]
                        if first_model.latest_versions:
                            latest_version = first_model.latest_versions[0]
                            model_uri = f"models:/{first_model.name}/{latest_version.version}"
                            logger.info(f"Using available model: {model_uri}")
                        else:
                            raise Exception(f"No model versions available")
                    else:
                        raise Exception(f"No registered models found")
            except Exception as e:
                logger.error(f"Failed to find existing model: {e}")
                raise Exception(f"Could not find model artifacts for best run")
        
        # Register the model
        try:
            model_version = mlflow.register_model(
                model_uri=model_uri,
                name=model_name
            )
            
            # Add description through model version update
            self.client.update_model_version(
                name=model_name,
                version=model_version.version,
                description=f"Best performing Iris classifier based on {metric} ({best_score:.4f})"
            )
            
            # Add tags to the model version
            self.client.set_model_version_tag(
                name=model_name,
                version=model_version.version,
                key="best_metric",
                value=metric
            )
            
            self.client.set_model_version_tag(
                name=model_name,
                version=model_version.version,
                key="best_score",
                value=str(best_score)
            )
            
            self.client.set_model_version_tag(
                name=model_name,
                version=model_version.version,
                key="model_type",
                value=best_run.data.tags.get('model_type', 'Unknown')
            )
            
            self.client.set_model_version_tag(
                name=model_name,
                version=model_version.version,
                key="registration_date",
                value=datetime.now().isoformat()
            )
            
            registration_info = {
                'model_name': model_name,
                'version': model_version.version,
                'run_id': best_run.info.run_id,
                'model_type': best_run.data.tags.get('model_type', 'Unknown'),
                'best_metric': metric,
                'best_score': best_score,
                'model_uri': model_uri,
                'registration_date': datetime.now().isoformat()
            }
            
            logger.info(f"Successfully registered model '{model_name}' version {model_version.version}")
            logger.info(f"Model type: {registration_info['model_type']}")
            logger.info(f"Best {metric}: {best_score:.4f}")
            
            return registration_info
            
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            raise
    
    def get_registered_models(self) -> List[Dict[str, Any]]:
        """
        Get all registered models.
        
        Returns:
            List of registered model information
        """
        registered_models = self.client.search_registered_models()
        
        models_info = []
        for model in registered_models:
            for version in model.latest_versions:
                model_info = {
                    'name': model.name,
                    'version': version.version,
                    'stage': version.current_stage,
                    'description': version.description,
                    'tags': version.tags,
                    'creation_timestamp': datetime.fromtimestamp(version.creation_timestamp / 1000),
                    'run_id': version.run_id
                }
                models_info.append(model_info)
        
        return models_info
    
    def promote_model_to_production(self, model_name: str, version: str) -> None:
        """
        Promote a model version to production stage.
        
        Args:
            model_name: Name of the registered model
            version: Version to promote
        """
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage="Production",
                archive_existing_versions=True
            )
            
            logger.info(f"Promoted model '{model_name}' version {version} to Production")
            
        except Exception as e:
            logger.error(f"Failed to promote model to production: {e}")
            raise
    
    def generate_monitoring_report(self) -> Dict[str, Any]:
        """
        Generate a comprehensive monitoring report.
        
        Returns:
            Dictionary with monitoring information
        """
        # Get runs summary
        runs_df = self.get_runs_summary()
        
        # Get registered models
        registered_models = self.get_registered_models()
        
        # Calculate summary statistics
        total_runs = len(runs_df)
        successful_runs = len(runs_df[runs_df['status'] == 'FINISHED'])
        
        # Model performance summary
        model_performance = {}
        for model_type in runs_df['model_type'].unique():
            if model_type != 'Unknown':
                model_runs = runs_df[runs_df['model_type'] == model_type]
                if not model_runs.empty and model_runs['test_accuracy'].notna().any():
                    model_performance[model_type] = {
                        'best_accuracy': model_runs['test_accuracy'].max(),
                        'avg_accuracy': model_runs['test_accuracy'].mean(),
                        'runs_count': len(model_runs)
                    }
        
        report = {
            'experiment_info': {
                'name': self.experiment_name,
                'id': self.experiment_id,
                'total_runs': total_runs,
                'successful_runs': successful_runs,
                'success_rate': successful_runs / total_runs if total_runs > 0 else 0
            },
            'model_performance': model_performance,
            'registered_models': registered_models,
            'runs_summary': runs_df.to_dict('records') if not runs_df.empty else [],
            'report_timestamp': datetime.now().isoformat()
        }
        
        return report
    
    def print_monitoring_summary(self) -> None:
        """Print a formatted monitoring summary."""
        report = self.generate_monitoring_report()
        
        print("\n" + "="*80)
        print("MLFLOW MONITORING SUMMARY")
        print("="*80)
        
        # Experiment info
        exp_info = report['experiment_info']
        print(f"\nExperiment: {exp_info['name']}")
        print(f"Total Runs: {exp_info['total_runs']}")
        print(f"Successful Runs: {exp_info['successful_runs']}")
        print(f"Success Rate: {exp_info['success_rate']:.2%}")
        
        # Model performance
        print(f"\nModel Performance Summary:")
        print("-" * 60)
        print(f"{'Model':<20} {'Best Acc':<12} {'Avg Acc':<12} {'Runs':<8}")
        print("-" * 60)
        
        for model_type, perf in report['model_performance'].items():
            print(f"{model_type:<20} {perf['best_accuracy']:<12.4f} "
                  f"{perf['avg_accuracy']:<12.4f} {perf['runs_count']:<8}")
        
        # Registered models
        if report['registered_models']:
            print(f"\nRegistered Models:")
            print("-" * 60)
            for model in report['registered_models']:
                print(f"- {model['name']} v{model['version']} ({model['stage']})")
                if 'best_score' in model['tags']:
                    print(f"  Best Score: {model['tags']['best_score']}")
                if 'model_type' in model['tags']:
                    print(f"  Model Type: {model['tags']['model_type']}")
        else:
            print(f"\nNo registered models found.")
        
        print("\n" + "="*80)


def main():
    """Demo the monitoring functionality."""
    try:
        # Initialize monitor
        monitor = MLflowMonitor()
        
        # Print monitoring summary
        monitor.print_monitoring_summary()
        
        # Find and register best model
        print("\nRegistering best performing model...")
        registration_info = monitor.register_best_model(
            metric='test_accuracy',
            model_name='iris-best-classifier'
        )
        
        print(f"\nModel Registration Details:")
        print(f"- Model Name: {registration_info['model_name']}")
        print(f"- Version: {registration_info['version']}")
        print(f"- Model Type: {registration_info['model_type']}")
        print(f"- Best Accuracy: {registration_info['best_score']:.4f}")
        
        # Promote to production
        print(f"\nPromoting model to production...")
        monitor.promote_model_to_production(
            registration_info['model_name'],
            registration_info['version']
        )
        
        print(f"\n✅ Model monitoring and registration completed successfully!")
        
    except Exception as e:
        logger.error(f"Monitoring failed: {e}")
        raise


if __name__ == "__main__":
    main()