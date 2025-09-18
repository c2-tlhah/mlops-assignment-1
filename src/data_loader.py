"""
Simple data loading for MLOps assignment.
This script loads the Iris dataset and splits it for training and testing.
"""

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class DataLoader:
    """Simple class to load and prepare the Iris dataset."""
    
    def __init__(self):
        """Initialize the data loader with basic settings."""
        self.scaler = StandardScaler()
        
    def load_iris_data(self):
        """Load the Iris dataset and return features and labels."""
        print("Loading Iris dataset...")
        
        # Load the built-in Iris dataset
        iris = load_iris()
        X = iris.data  # Features: sepal/petal length and width
        y = iris.target  # Labels: 0=setosa, 1=versicolor, 2=virginica
        
        print(f"Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
        return X, y
    
    def prepare_data(self, X, y):
        """Split data into training and testing sets and scale the features."""
        print("Splitting data into train (80%) and test (20%) sets...")
        
        # Split the data - 80% for training, 20% for testing
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale the features to have mean=0 and std=1
        print("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Training set: {X_train_scaled.shape[0]} samples")
        print(f"Test set: {X_test_scaled.shape[0]} samples")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def get_data(self):
        """Get the complete prepared dataset for machine learning."""
        # Load the Iris dataset
        X, y = self.load_iris_data()
        
        # Split and scale the data
        X_train, X_test, y_train, y_test = self.prepare_data(X, y)
        
        return X_train, X_test, y_train, y_test


# Simple test of the data loader
if __name__ == "__main__":
    loader = DataLoader()
    X_train, X_test, y_train, y_test = loader.get_data()
    
    print("Data loading completed!")
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print("Ready for model training!")