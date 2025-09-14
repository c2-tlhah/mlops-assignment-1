"""
Data loading utilities for the MLOps assignment.
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Class to handle data loading and preprocessing for ML models."""
    
    def __init__(self, dataset_name: str = "iris", test_size: float = 0.2, random_state: int = 42):
        """
        Initialize the DataLoader.
        
        Args:
            dataset_name: Name of the dataset to load
            test_size: Proportion of dataset to include in test split
            random_state: Random state for reproducibility
        """
        self.dataset_name = dataset_name
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        
    def load_iris_dataset(self) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Load the Iris dataset.
        
        Returns:
            Tuple of (features, targets, metadata)
        """
        logger.info("Loading Iris dataset...")
        
        # Load the dataset
        iris = load_iris()
        X, y = iris.data, iris.target
        
        # Create metadata
        metadata = {
            'feature_names': iris.feature_names,
            'target_names': iris.target_names,
            'n_samples': X.shape[0],
            'n_features': X.shape[1],
            'n_classes': len(iris.target_names),
            'dataset_description': iris.DESCR
        }
        
        logger.info(f"Dataset loaded: {metadata['n_samples']} samples, {metadata['n_features']} features, {metadata['n_classes']} classes")
        
        return X, y, metadata
    
    def split_and_preprocess(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train/test sets and apply preprocessing.
        
        Args:
            X: Feature matrix
            y: Target vector
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        logger.info("Splitting data into train/test sets...")
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state,
            stratify=y  # Ensure balanced splits
        )
        
        # Standardize features
        logger.info("Standardizing features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        logger.info(f"Train set: {X_train_scaled.shape[0]} samples")
        logger.info(f"Test set: {X_test_scaled.shape[0]} samples")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def get_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Complete data loading pipeline.
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, metadata)
        """
        if self.dataset_name.lower() == "iris":
            X, y, metadata = self.load_iris_dataset()
        else:
            raise ValueError(f"Dataset '{self.dataset_name}' not supported")
        
        X_train, X_test, y_train, y_test = self.split_and_preprocess(X, y)
        
        return X_train, X_test, y_train, y_test, metadata


def main():
    """Test the data loader functionality."""
    loader = DataLoader()
    X_train, X_test, y_train, y_test, metadata = loader.get_data()
    
    print("Data loading completed successfully!")
    print(f"Training set shape: {X_train.shape}")
    print(f"Test set shape: {X_test.shape}")
    print(f"Classes: {metadata['target_names']}")


if __name__ == "__main__":
    main()