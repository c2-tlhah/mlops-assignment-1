"""
Simple script to run the complete MLOps pipeline.
This is the main script students should run.
"""

from src.train_pipeline import SimpleTrainingPipeline
from src.model_registry import SimpleModelRegistry


def main():
    """Run the complete MLOps pipeline."""
    print("🚀 Starting MLOps Assignment Pipeline")
    print("=" * 50)
    
    try:
        # Step 1: Run training pipeline
        print("\n📊 Phase 1: Model Training and Evaluation")
        pipeline = SimpleTrainingPipeline()
        results, best_model = pipeline.run_pipeline()
        
        # Step 2: Register best model
        print("\n📝 Phase 2: Model Registration")
        registry = SimpleModelRegistry()
        registry.register_best_model()
        
        # Step 3: Show final results
        print("\n🎉 Pipeline Completed Successfully!")
        print("=" * 50)
        print(f"✅ Best Model: {best_model}")
        print(f"✅ Model registered in MLflow")
        print("\n📊 To view results:")
        print("   1. Run: mlflow ui --backend-store-uri file:./mlruns")
        print("   2. Open: http://localhost:5000")
        print("   3. Check the 'Models' tab to see registered model")
        
    except Exception as e:
        print(f"\n❌ Error in pipeline: {e}")
        print("Please check your setup and try again.")


if __name__ == "__main__":
    main()
