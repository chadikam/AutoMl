"""
Model training service for AutoML Framework
Supports multiple ML algorithms with automatic training and evaluation
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import Dict, Any, Tuple, Optional
import joblib
import os
from app.models.schemas import ModelType


class ModelTrainingService:
    """
    Service for training and evaluating machine learning models
    """
    
    # Model registry mapping ModelType enum to sklearn classes
    MODEL_REGISTRY = {
        ModelType.LOGISTIC_REGRESSION: LogisticRegression,
        ModelType.DECISION_TREE: DecisionTreeClassifier,
        ModelType.RANDOM_FOREST: RandomForestClassifier,
        ModelType.GRADIENT_BOOSTING: GradientBoostingClassifier,
    }
    
    # Default hyperparameters for each model type
    DEFAULT_PARAMS = {
        ModelType.LOGISTIC_REGRESSION: {
            'max_iter': 1000,
            'random_state': 42
        },
        ModelType.DECISION_TREE: {
            'max_depth': 10,
            'random_state': 42
        },
        ModelType.RANDOM_FOREST: {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42
        },
        ModelType.GRADIENT_BOOSTING: {
            'n_estimators': 100,
            'learning_rate': 0.1,
            'max_depth': 3,
            'random_state': 42
        },
    }
    
    def __init__(self, model_type: ModelType, params: Optional[Dict[str, Any]] = None):
        """
        Initialize the training service with a specific model type
        
        Args:
            model_type: Type of model to train (from ModelType enum)
            params: Optional custom hyperparameters
        """
        self.model_type = model_type
        self.params = params or self.DEFAULT_PARAMS.get(model_type, {})
        self.model = None
        self.feature_names = []
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
    
    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and testing sets
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of data for testing (default: 0.2)
            random_state: Random seed for reproducibility
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state,
            stratify=y if len(np.unique(y)) < max(20, len(y) * 0.05) else None
        )
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """
        Train the model on training data
        
        Args:
            X_train: Training feature matrix
            y_train: Training target vector
        
        Returns:
            Trained model instance
        """
        # Get model class from registry
        model_class = self.MODEL_REGISTRY[self.model_type]
        
        # Initialize model with parameters
        self.model = model_class(**self.params)
        
        # Train model
        self.model.fit(X_train, y_train)
        
        return self.model
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate model performance on test data
        
        Args:
            X_test: Test feature matrix
            y_test: Test target vector
        
        Returns:
            Dictionary containing evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Handle binary and multiclass classification
        average_method = 'binary' if len(np.unique(y_test)) == 2 else 'weighted'
        
        precision = precision_score(y_test, y_pred, average=average_method, zero_division=0)
        recall = recall_score(y_test, y_pred, average=average_method, zero_division=0)
        f1 = f1_score(y_test, y_pred, average=average_method, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'confusion_matrix': cm.tolist()
        }
        
        return metrics
    
    def get_feature_importance(self, feature_names: list) -> Optional[Dict[str, float]]:
        """
        Extract feature importance (if model supports it)
        
        Args:
            feature_names: List of feature names
        
        Returns:
            Dictionary mapping feature names to importance scores, or None
        """
        if self.model is None:
            return None
        
        # Check if model has feature_importances_ attribute
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            feature_importance = {
                name: float(importance) 
                for name, importance in zip(feature_names, importances)
            }
            # Sort by importance
            feature_importance = dict(sorted(
                feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
            return feature_importance
        
        # For Logistic Regression, use coefficients
        elif hasattr(self.model, 'coef_'):
            coef = np.abs(self.model.coef_[0]) if len(self.model.coef_.shape) > 1 else np.abs(self.model.coef_)
            feature_importance = {
                name: float(importance) 
                for name, importance in zip(feature_names, coef)
            }
            feature_importance = dict(sorted(
                feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
            return feature_importance
        
        return None
    
    def train_and_evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list,
        test_size: float = 0.2
    ) -> Dict[str, Any]:
        """
        Complete workflow: split data, train model, evaluate performance
        
        Args:
            X: Feature matrix
            y: Target vector
            feature_names: List of feature names
            test_size: Proportion of data for testing
        
        Returns:
            Dictionary with metrics and feature importance
        """
        self.feature_names = feature_names
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y, test_size=test_size)
        
        # Train model
        self.train(X_train, y_train)
        
        # Evaluate model
        metrics = self.evaluate(X_test, y_test)
        
        # Get feature importance
        feature_importance = self.get_feature_importance(feature_names)
        if feature_importance:
            metrics['feature_importance'] = feature_importance
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data
        
        Args:
            X: Feature matrix
        
        Returns:
            Predicted labels
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        return self.model.predict(X)
    
    def save_model(self, filepath: str):
        """
        Save trained model to disk
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'params': self.params,
            'feature_names': self.feature_names
        }
        
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str):
        """
        Load trained model from disk
        
        Args:
            filepath: Path to the saved model
        """
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.params = model_data['params']
        self.feature_names = model_data['feature_names']
