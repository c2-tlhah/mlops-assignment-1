#!/usr/bin/env python3
"""
Quick start script for MLOps Assignment 1
Runs the complete training pipeline with MLflow tracking.
"""

import os
import sys

def main():
    """Run the complete MLOps pipeline."""
    print("="*60)
    print("MLOps Assignment 1 - Quick Start")
    print("="*60)
    
    # Check if we're in the right directory
    if not os.path.exists("src/train_pipeline.py"):
        print("❌ Error: Please run this script from the project root directory")
        print("   Current directory:", os.getcwd())
        print("   Expected files: src/train_pipeline.py")
        return 1
    
    print("🚀 Starting MLOps training pipeline...")
    print("\n1. Loading dataset (Iris)")
    print("2. Training 3 ML models (Logistic Regression, Random Forest, SVM)")
    print("3. Evaluating models with comprehensive metrics")
    print("4. Tracking experiments with MLflow")
    print("5. Generating visualizations and artifacts")
    
    # Add src to Python path
    sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
    
    try:
        # Import and run the training pipeline
        from train_pipeline import main as run_pipeline
        
        print("\n" + "="*60)
        print("STARTING TRAINING PIPELINE")
        print("="*60)
        
        results = run_pipeline()
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY! 🎉")
        print("="*60)
        
        print("\n📊 To view MLflow UI:")
        print("   1. Open terminal in this directory")
        print("   2. Run: mlflow ui --backend-store-uri file:./mlruns")
        print("   3. Open: http://localhost:5000")
        
        print("\n📁 Files created:")
        print("   - Models saved in: ./models/")
        print("   - MLflow runs in: ./mlruns/")
        print("   - Jupyter notebook: ./notebooks/mlops_training_demo.ipynb")
        
        print("\n🔍 What to check in MLflow UI:")
        print("   - Compare model performances")
        print("   - View confusion matrices")
        print("   - Analyze feature importance")
        print("   - Download trained models")
        
        return 0
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("\n💡 Please install required packages:")
        print("   pip install -r requirements.txt")
        return 1
        
    except Exception as e:
        print(f"❌ Pipeline Error: {e}")
        print("\n📝 Check the logs above for detailed error information")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)