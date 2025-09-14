# MLOps Assignment 1: Model Training & Comparison with MLflow Tracking

A comprehensive MLOps project demonstrating machine learning model training, evaluation, comparison, and experiment tracking using MLflow with model registry integration.

## 🎯 Problem Statement

This project addresses the challenge of building a robust machine learning pipeline for **Iris flower classification** that can:
- Train and compare multiple ML models efficiently
- Track experiments and maintain reproducibility  
- Register and manage model versions
- Provide comprehensive evaluation and monitoring

**Dataset**: Classic Iris dataset with 150 samples, 4 features (sepal/petal length/width), and 3 species (setosa, versicolor, virginica).

**Objective**: Build a multi-class classification system with complete MLOps workflow and MLflow tracking.

## 📁 Project Structure

```
mlops-assignment-1/
├── data/                          # Dataset storage
├── notebooks/                     # Jupyter notebooks
│   └── mlops_training_demo.ipynb  # Interactive demo notebook
├── src/                           # Source code
│   ├── data_loader.py            # Data loading utilities
│   ├── models.py                 # Model training and evaluation
│   ├── mlflow_utils.py           # MLflow tracking utilities
│   ├── model_registry.py         # Model monitoring and registry
│   └── train_pipeline.py         # Complete training pipeline
├── models/                        # Trained model storage
│   ├── logistic_regression_best.pkl
│   ├── random_forest_best.pkl
│   └── svm_best.pkl
├── results/                       # Experiment results
├── mlruns/                        # MLflow tracking data
├── requirements.txt               # Python dependencies
├── run_pipeline.py               # Quick start script
└── README.md                     # Project documentation
```

## 🚀 Quick Start Guide

### 1. Prerequisites
- Python 3.7+
- All packages installed globally (as specified)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Complete Pipeline
```bash
# Execute the full MLOps pipeline
python3 run_pipeline.py
```

### 4. Register Best Model
```bash
# Register the best performing model
python3 src/model_registry.py
```

### 5. View MLflow UI
```bash
# Start MLflow UI
mlflow ui --backend-store-uri file:./mlruns

# Open in browser: http://localhost:5000
```

## 📊 Model Selection & Comparison

### Models Implemented

| Model | Hyperparameters Tuned | Cross-Validation | Key Strengths |
|-------|----------------------|------------------|---------------|
| **Logistic Regression** | C, solver, penalty | 5-fold stratified | Simple, interpretable, fast |
| **Random Forest** | n_estimators, max_depth, min_samples_split, min_samples_leaf | 5-fold stratified | Handles non-linearity, feature importance |
| **Support Vector Machine** | C, kernel, gamma | 5-fold stratified | Effective in high dimensions, robust |

### Performance Results

| Model | CV Score | Test Accuracy | Test Precision | Test Recall | Test F1-Score |
|-------|----------|---------------|----------------|-------------|---------------|
| **Logistic Regression** ⭐ | 0.9667 | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| Random Forest | 0.9583 | 0.9667 | 0.9697 | 0.9667 | 0.9666 |
| SVM | 0.9833 | 0.9667 | 0.9697 | 0.9667 | 0.9666 |

**🏆 Best Model**: Logistic Regression achieved perfect 100% accuracy on the test set and is registered as the production model.

## 📸 MLflow UI Screenshots

### Experiment Tracking Dashboard
![MLflow Experiment Tracking](1.png)
*MLflow experiment tracking interface showing all model runs with parameters, metrics, and artifacts*

### Model Registry Interface
![MLflow Model Registry](2.png)
*MLflow Model Registry showing registered models with versions and stages*

## 🔬 MLflow Experiment Tracking

### Part 3 Implementation ✅

Our MLflow integration includes comprehensive tracking of:

#### 1. **Parameters Logged**
- All hyperparameters for each model
- Data preprocessing parameters
- Cross-validation configuration
- Model-specific settings

#### 2. **Metrics Logged**
- Cross-validation scores
- Test accuracy, precision, recall, F1-score
- Training time and performance
- Model comparison metrics

#### 3. **Artifacts Logged**
- **Confusion Matrices**: Visual performance analysis per model
- **Feature Importance Plots**: For tree-based models
- **Classification Reports**: Detailed per-class metrics (JSON format)
- **Model Comparison Charts**: Side-by-side performance visualization
- **Trained Models**: Serialized models for deployment

#### 4. **Experiment Organization**
```
Iris-Classification-Comparison/
├── logistic_regression_training/     # Training run
├── logistic_regression_evaluation/   # Evaluation run
├── random_forest_training/           # Training run
├── random_forest_evaluation/         # Evaluation run
├── svm_training/                     # Training run
├── svm_evaluation/                   # Evaluation run
└── models_comparison/                # Overall comparison
```

## 🏭 Model Registry & Monitoring

### Part 4 Implementation ✅

#### 1. **Monitoring Capabilities**
- **Experiment Monitoring**: Track all runs and their status
- **Performance Monitoring**: Compare metrics across models
- **Success Rate Tracking**: Monitor pipeline health
- **Automated Best Model Selection**: Based on configurable metrics

#### 2. **Model Registration Process**
```python
# Best model identification
best_run, best_score = monitor.find_best_model_run('test_accuracy')

# Model registration
model_version = mlflow.register_model(
    model_uri=f"runs:/{best_run.info.run_id}/model",
    name="iris-best-classifier"
)
```

#### 3. **Model Registry Features**
- ✅ **Version Control**: Automatic version numbering
- ✅ **Stage Management**: None → Staging → Production
- ✅ **Metadata Tracking**: Tags for model lineage
- ✅ **Model Lineage**: Full traceability to training runs

#### 4. **Registered Models**
- `iris-best-classifier`: Production model (Logistic Regression)
- `logistic-regression-model`: Individual model registry
- `random-forest-model`: Individual model registry  
- `svm-model`: Individual model registry

## 📋 Complete Running Instructions

### Step-by-Step Execution

#### 1. **Environment Setup**
```bash
# Clone the repository
git clone https://github.com/c2-tlhah/mlops-assignment-1.git
cd mlops-assignment-1

# Install dependencies (global environment)
pip install -r requirements.txt
```

#### 2. **Run Complete Pipeline**
```bash
# Execute full training and evaluation
python3 run_pipeline.py
```
**Expected Output**: 
- Data loading and preprocessing
- 3 models trained with hyperparameter tuning
- Model evaluation and comparison
- MLflow tracking with all artifacts
- Performance summary

#### 3. **Register Best Model**
```bash
# Register the best performing model
python3 src/model_registry.py
```
**Expected Output**:
- Experiment monitoring summary
- Best model identification
- Model registration in MLflow registry

#### 4. **Interactive Analysis**
```bash
# Launch Jupyter notebook
jupyter notebook notebooks/mlops_training_demo.ipynb
```
**Contains**:
- Step-by-step walkthrough
- Data exploration and visualization
- Model training with explanations
- Results analysis and interpretation

#### 5. **MLflow UI Exploration**
```bash
# Start MLflow UI
mlflow ui --backend-store-uri file:./mlruns
```
**Explore**:
- Experiment runs and metrics
- Model comparisons and artifacts
- Registered models and versions

## 📸 MLflow UI Screenshots Guide

### Accessing MLflow UI
1. **Start MLflow UI**: `mlflow ui --backend-store-uri file:./mlruns`
2. **Open Browser**: http://localhost:5000

### Key Views to Screenshot

#### 1. **Experiments Overview**
- Navigate to "Experiments" tab
- Shows: Iris-Classification-Comparison experiment
- Screenshot shows: All runs with their metrics

#### 2. **Run Comparison**
- Select multiple runs (Ctrl+click)
- Click "Compare" button
- Screenshot shows: Side-by-side metric comparison

#### 3. **Model Registry**
- Click "Models" tab in top navigation
- Screenshot shows: Registered models list
- Click on "iris-best-classifier" for details

## 🎯 Assignment Requirements Checklist
### ✅ Part 1 – GitHub Setup
- Create a new GitHub repository named mlops-assignment-1.
-  Clone the repository to your local machine.
### ✅ Part 2 – Model Training & Comparison (25 marks)
- [x] **Dataset Selection**: Iris dataset for multi-class classification
- [x] **Multiple Models**: 3 ML models (Logistic Regression, Random Forest, SVM)
- [x] **Hyperparameter Tuning**: Grid search with cross-validation
- [x] **Model Comparison**: Comprehensive evaluation with multiple metrics
- [x] **Model Storage**: All trained models saved in `/models` folder

### ✅ Part 3 – MLflow Tracking & Logging (30 marks)
- [x] **MLflow Setup**: Complete configuration and experiment management
- [x] **Parameter Logging**: All hyperparameters tracked
- [x] **Metrics Logging**: Accuracy, precision, recall, F1-score
- [x] **Artifact Logging**: Plots, confusion matrices, classification reports
- [x] **MLflow UI**: Full functionality for run comparison and visualization

### ✅ Part 4 – Monitoring & Model Registration (15 marks)
- [x] **MLflow Monitoring**: Comprehensive experiment and metrics monitoring
- [x] **Best Model Selection**: Automated identification based on performance
- [x] **Model Registration**: MLflow Model Registry with version control
- [x] **Documentation**: Complete registration process documentation

### ✅ Part 5 – Documentation & GitHub Submission (10 marks)
- [x] **Problem Statement**: Clear dataset and objective description
- [x] **Model Selection**: Detailed comparison and analysis
- [x] **MLflow Screenshots**: Comprehensive UI documentation guide
- [x] **Model Registration**: Step-by-step registration documentation
- [x] **Running Instructions**: Complete setup and execution guide
- [x] **GitHub Repository**: All code, logs, and documentation

## 🚀 Advanced Features Implemented

### 1. **Automated Pipeline**
- End-to-end automation from data to deployment
- Error handling and logging throughout
- Reproducible results with fixed random seeds

### 2. **Comprehensive Evaluation**
- Multiple metrics for thorough assessment
- Stratified cross-validation for reliable estimates
- Visual analysis with confusion matrices and plots

### 3. **Production-Ready Code**
- Modular design with clean separation of concerns
- Type hints and comprehensive documentation
- Exception handling and informative logging

### 4. **MLflow Best Practices**
- Structured experiment organization
- Complete artifact management
- Model lineage and provenance tracking

## 📚 Usage Examples

### Running Individual Components

```python
# Load data
from src.data_loader import DataLoader
loader = DataLoader()
X_train, X_test, y_train, y_test, metadata = loader.get_data()

# Train a single model
from src.models import ModelTrainer
trainer = ModelTrainer()
model, info = trainer.train_model('logistic_regression', X_train, y_train)

# Evaluate model
from src.models import ModelEvaluator
evaluator = ModelEvaluator()
results = evaluator.evaluate_model(model, X_test, y_test, 'logistic_regression')

# Monitor experiments
from src.model_registry import MLflowMonitor
monitor = MLflowMonitor()
summary = monitor.get_experiment_summary()
```

## 🛠️ Technical Details

### Dependencies
- **scikit-learn**: Machine learning models and evaluation
- **MLflow**: Experiment tracking and model management
- **pandas/numpy**: Data manipulation and numerical computing
- **matplotlib/seaborn**: Visualization and plotting
- **joblib**: Model serialization

### Environment
- **Python**: 3.7+
- **Platform**: Cross-platform (Linux, Windows, macOS)
- **Package Management**: pip (global environment as specified)

## 🔧 Technical Architecture

### Data Flow
```
Raw Data → Data Loader → Preprocessing → Model Training 
    ↓
MLflow Tracking ← Evaluation ← Trained Models
    ↓
Model Registry ← Best Model Selection ← Monitoring
    ↓
Production Deployment
```

### Key Components
- **DataLoader**: Handles dataset loading and preprocessing
- **ModelTrainer**: Manages training with hyperparameter tuning
- **ModelEvaluator**: Comprehensive model evaluation
- **MLflowTracker**: Experiment tracking and artifact management
- **MLflowMonitor**: Model monitoring and registry management

## 📞 Support & Troubleshooting

### Common Issues

1. **MLflow UI not starting**:
   ```bash
   # Check for existing processes
   pkill -f mlflow
   # Restart MLflow UI
   mlflow ui --backend-store-uri file:./mlruns
   ```

2. **Missing dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Model registration errors**:
   - Ensure training pipeline completed successfully
   - Check that mlruns/ directory exists
   - Verify experiment has evaluation runs

### Getting Help
1. Check the Jupyter notebook for detailed explanations
2. Review MLflow UI for experiment details
3. Examine terminal logs for error messages
4. Ensure all dependencies are correctly installed

## 🚀 Next Steps

1. **Advanced Hyperparameter Optimization**: Implement Bayesian optimization
2. **Model Deployment**: Create REST API for model serving
3. **Monitoring Dashboard**: Real-time model performance monitoring
4. **A/B Testing**: Framework for comparing model versions in production
5. **Data Drift Detection**: Monitor incoming data for distribution changes

---

**Project**: MLOps Assignment 1  
**Author**: M Talha Ramzan  
**Date**: September 2025  
**Repository**: https://github.com/c2-tlhah/mlops-assignment-1  
**MLflow Tracking**: Complete experiment tracking with model registry  
**Status**: ✅ All assignment requirements implemented and documented
