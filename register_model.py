#!/usr/bin/env python3
"""
Model Registry Script for MLOps Assignment 1
Monitors MLflow experiments and registers the best performing model.
"""

import os
import sys

def main():
    """Run model monitoring and registration."""
    print("="*60)
    print("MLOps Assignment 1 - Model Registry")
    print("="*60)
    
    # Check if we're in the right directory
    if not os.path.exists("src/model_registry.py"):
        print("❌ Error: Please run this script from the project root directory")
        print("   Current directory:", os.getcwd())
        print("   Expected files: src/model_registry.py")
        return 1
    
    # Check if MLflow runs exist
    if not os.path.exists("mlruns"):
        print("❌ Error: No MLflow runs found")
        print("   Please run the training pipeline first: python3 run_pipeline.py")
        return 1
    
    print("🔍 Monitoring MLflow experiments...")
    print("📝 Registering best performing model...")
    
    # Add src to Python path
    sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
    
    try:
        # Import and run model registration
        from model_registry import main as run_registration
        
        print("\n" + "="*60)
        print("STARTING MODEL REGISTRATION")
        print("="*60)
        
        run_registration()
        
        print("\n" + "="*60)
        print("MODEL REGISTRATION COMPLETED SUCCESSFULLY! 🎉")
        print("="*60)
        
        print("\n📊 To view registered models in MLflow UI:")
        print("   1. Open terminal in this directory")
        print("   2. Run: mlflow ui --backend-store-uri file:./mlruns")
        print("   3. Open: http://localhost:5000")
        print("   4. Click on 'Models' tab to see registered models")
        
        print("\n🎯 What was registered:")
        print("   - Best performing model identified and registered")
        print("   - Model promoted to 'Production' stage")
        print("   - Version number assigned automatically")
        print("   - Tags added for tracking and metadata")
        
        print("\n📋 Model Registry Benefits:")
        print("   - Centralized model versioning")
        print("   - Stage-based model lifecycle management")
        print("   - Model lineage and provenance tracking")
        print("   - Easy model deployment and rollback")
        
        return 0
        
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        print("\n💡 Please install required packages:")
        print("   pip install -r requirements.txt")
        return 1
        
    except Exception as e:
        print(f"❌ Registration Error: {e}")
        print("\n📝 Check the logs above for detailed error information")
        print("💡 Make sure MLflow experiments exist (run training pipeline first)")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)