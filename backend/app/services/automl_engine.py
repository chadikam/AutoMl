"""
AutoML Training Engine with Optuna Hyperparameter Optimization
===============================================================

This engine implements a generalization-focused AutoML system that:
1. Uses Optuna for hyperparameter optimization
2. Applies stratified K-Fold cross-validation
3. Detects and penalizes overfitting
4. Selects best model based on generalization score (NOT raw performance)
5. Generates comprehensive evaluation plots

CRITICAL: Model selection prioritizes generalization over raw scores.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.model_selection import (
    StratifiedKFold, KFold, cross_val_score, cross_validate
)
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)

# Models - Classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Models - Regression
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor
)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

# Models - Unsupervised
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA

import joblib
import os
import gc
import psutil
from datetime import datetime
import multiprocessing
from multiprocessing import Process, Queue
import tempfile
import signal
import traceback


class TaskType(str, Enum):
    """ML task types"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    CLUSTERING = "clustering"


@dataclass
class ModelResult:
    """Results for a single model after training and evaluation"""
    model_name: str
    model_type: str
    task_type: TaskType
    
    # Trained model
    best_model: Any
    best_params: Dict[str, Any]
    
    # Scores
    train_score: float
    cv_score: float
    cv_std: float
    test_score: float
    
    # Overfitting metrics
    overfit_gap: float
    penalty: float
    generalization_score: float
    overfitting: str
    rejected: bool
    rejection_reason: Optional[str]
    
    # Additional info
    detailed_metrics: Dict[str, Any]
    feature_importance: Optional[Dict[str, float]]
    n_trials: int
    optimization_time: float
    
    # Stability info
    stability_status: str = "stable"  # "stable", "unstable", "skipped"
    stability_reason: Optional[str] = None
    generalization_score: float
    
    # Status flags
    overfitting: bool
    rejected: bool
    rejection_reason: Optional[str] = None
    
    # Detailed metrics
    detailed_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Feature importance
    feature_importance: Optional[Dict[str, float]] = None
    
    # Optuna study info
    n_trials: int = 0
    best_trial_number: int = 0
    optimization_time: float = 0.0


@dataclass
class AutoMLResult:
    """Final AutoML results with best model selection"""
    task_type: TaskType
    
    # Best model (based on generalization score)
    best_model: ModelResult
    
    # All models evaluated
    all_models: List[ModelResult]
    
    # Summary statistics
    total_models_evaluated: int
    models_rejected: int
    
    # Plots
    plot_paths: Dict[str, str] = field(default_factory=dict)
    
    # Metadata
    training_timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    feature_names: List[str] = field(default_factory=list)
    target_name: str = ""


def _train_model_subprocess(model_class, params, data_file, result_file, task_type, cv_folds, random_state):
    """
    Train a model in a subprocess with isolation from main process.
    Uses temp files to avoid serialization issues with large sparse matrices.
    
    Args:
        model_class: The model class to instantiate
        params: Hyperparameters dict
        data_file: Path to joblib file containing (X_train, y_train, X_test, y_test)
        result_file: Path to write results
        task_type: TaskType enum value
        cv_folds: Number of CV folds
        random_state: Random seed
    """
    try:
        # Load data from file
        data = joblib.load(data_file)
        X_train = data['X_train']
        y_train = data['y_train']
        X_test = data['X_test']
        y_test = data['y_test']
        
        # Instantiate model
        model = model_class(**params)
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Get predictions and scores
        if task_type == TaskType.CLUSTERING:
            # Unsupervised - silhouette score
            labels = model.predict(X_train) if hasattr(model, 'predict') else model.labels_
            if len(np.unique(labels)) > 1:
                from sklearn.metrics import silhouette_score
                score = silhouette_score(X_train, labels)
            else:
                score = -1.0
            cv_score = score
            cv_std = 0.0
            test_score = score
        else:
            # Supervised - cross-validation
            if task_type == TaskType.CLASSIFICATION:
                from sklearn.model_selection import StratifiedKFold, cross_val_score
                cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
                scoring = 'f1_weighted'
            else:
                from sklearn.model_selection import KFold, cross_val_score
                cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
                scoring = 'r2'
            
            cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring, n_jobs=1)
            cv_score = np.mean(cv_scores)
            cv_std = np.std(cv_scores)
            
            # Test score
            if X_test is not None and y_test is not None:
                if task_type == TaskType.CLASSIFICATION:
                    from sklearn.metrics import f1_score
                    y_pred = model.predict(X_test)
                    test_score = f1_score(y_test, y_pred, average='weighted')
                else:
                    from sklearn.metrics import r2_score
                    y_pred = model.predict(X_test)
                    test_score = r2_score(y_test, y_pred)
            else:
                test_score = cv_score
        
        # Write results to file
        joblib.dump({
            'success': True,
            'cv_score': float(cv_score),
            'cv_std': float(cv_std),
            'test_score': float(test_score)
        }, result_file)
        
    except Exception as e:
        # Write error to file
        joblib.dump({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }, result_file)


class AutoMLEngine:
    """
    Complete AutoML training engine with Optuna optimization.
    
    Key Features:
    - Hyperparameter optimization with Optuna (TPE sampler + Median pruning)
    - Stratified K-Fold for classification, K-Fold for regression
    - Overfitting detection with penalty system
    - Model selection based on generalization score
    - Comprehensive evaluation plots
    """
    
    # Classification models
    CLASSIFICATION_MODELS = {
        'logistic_regression': LogisticRegression,
        'decision_tree': DecisionTreeClassifier,
        'random_forest': RandomForestClassifier,
        'gradient_boosting': GradientBoostingClassifier,
        'xgboost': XGBClassifier,
        'lightgbm': LGBMClassifier,
        'svm': SVC,
        'knn': KNeighborsClassifier,
        'adaboost': AdaBoostClassifier,
    }
    
    # Regression models
    REGRESSION_MODELS = {
        'ridge': Ridge,
        'lasso': Lasso,
        'elastic_net': ElasticNet,
        'decision_tree': DecisionTreeRegressor,
        'random_forest': RandomForestRegressor,
        'gradient_boosting': GradientBoostingRegressor,
        'xgboost': XGBRegressor,
        'lightgbm': LGBMRegressor,
        'svr': SVR,
        'knn': KNeighborsRegressor,
        'adaboost': AdaBoostRegressor,
    }
    
    # Clustering models
    CLUSTERING_MODELS = {
        'kmeans': KMeans,
        'dbscan': DBSCAN,
        'agglomerative': AgglomerativeClustering,
    }
    
    def __init__(
        self,
        task_type: TaskType,
        n_trials: int = 75,
        cv_folds: int = 5,
        test_size: float = 0.2,
        random_state: int = 42,
        penalty_factor: float = 2.0,
        overfit_threshold_reject: float = 0.20,
        overfit_threshold_high: float = 0.10,
        max_cpu_cores: int = 4,
        verbose: bool = True
    ):
        """
        Initialize AutoML engine.
        
        Args:
            task_type: Type of ML task (classification/regression/clustering)
            n_trials: Number of Optuna trials per model
            cv_folds: Number of cross-validation folds
            test_size: Proportion of data for final test set
            random_state: Random seed for reproducibility
            penalty_factor: Base penalty factor for overfitting
            overfit_threshold_reject: Gap threshold for model rejection
            overfit_threshold_high: Gap threshold for high penalty
            max_cpu_cores: Maximum CPU cores to use (limits parallelism)
            verbose: Print progress messages
        """
        self.task_type = task_type
        self.n_trials = n_trials
        self.cv_folds = cv_folds
        self.test_size = test_size
        self.random_state = random_state
        self.penalty_factor = penalty_factor
        self.overfit_threshold_reject = overfit_threshold_reject
        self.overfit_threshold_high = overfit_threshold_high
        self.max_cpu_cores = max_cpu_cores
        self.verbose = verbose
        
        # Configure CPU limits
        self._setup_cpu_limits()
        
        # Select model registry based on task type
        if task_type == TaskType.CLASSIFICATION:
            self.models = self.CLASSIFICATION_MODELS
        elif task_type == TaskType.REGRESSION:
            self.models = self.REGRESSION_MODELS
        else:
            self.models = self.CLUSTERING_MODELS
        
        # Results storage
        self.results: List[ModelResult] = []
        self.best_result: Optional[ModelResult] = None
    
    def _setup_cpu_limits(self):
        """Configure CPU core limits to prevent 100% CPU usage"""
        import os
        import multiprocessing
        
        # Set environment variables to limit CPU cores
        os.environ['OMP_NUM_THREADS'] = str(self.max_cpu_cores)
        os.environ['MKL_NUM_THREADS'] = str(self.max_cpu_cores)
        os.environ['OPENBLAS_NUM_THREADS'] = str(self.max_cpu_cores)
        os.environ['NUMEXPR_NUM_THREADS'] = str(self.max_cpu_cores)
        
        # Force CPU affinity at OS level (Windows/Linux compatible)
        try:
            import psutil
            process = psutil.Process()
            total_cores = psutil.cpu_count(logical=True)
            
            # Limit to specific CPU cores (0, 1, 2, ... up to max_cpu_cores-1)
            cores_to_use = list(range(min(self.max_cpu_cores, total_cores)))
            process.cpu_affinity(cores_to_use)
            
            if self.verbose:
                print(f"✓ CPU core limit set to {self.max_cpu_cores} (of {total_cores} available)")
                print(f"  Process restricted to CPU cores: {cores_to_use}")
        except Exception as e:
            if self.verbose:
                print(f"⚠ Could not set CPU affinity: {e}")
                print(f"  Using environment variables only")
    
    def _check_memory_usage(self):
        """Monitor and manage memory usage during training"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_percent = process.memory_percent()
            
            # If memory usage exceeds threshold, force garbage collection
            if memory_percent > 80:
                gc.collect()
                if self.verbose:
                    print(f"⚠️  High memory usage ({memory_percent:.1f}%), running garbage collection")
            
        except Exception:
            pass
    
    def _check_model_stability(
        self, 
        model: Any, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        predictions: Optional[np.ndarray] = None,
        train_score: Optional[float] = None,
        cv_score: Optional[float] = None
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if a trained model is stable and reliable.
        
        Returns:
            Tuple of (is_stable, instability_reason)
        """
        # 1. Check for convergence warnings in linear models
        if hasattr(model, 'n_iter_'):
            # Check if model hit max iterations
            if hasattr(model, 'max_iter'):
                try:
                    n_iter = model.n_iter_
                    max_iter = model.max_iter
                    # Handle None values and ensure both are valid
                    if n_iter is not None and max_iter is not None:
                        if isinstance(n_iter, np.ndarray):
                            if np.any(n_iter >= max_iter):
                                return False, "convergence_failure_max_iter"
                        elif n_iter >= max_iter:
                            return False, "convergence_failure_max_iter"
                except (TypeError, AttributeError):
                    # Skip if comparison fails
                    pass
        
        # 2. Check for NaN or infinite coefficients/weights
        if hasattr(model, 'coef_'):
            coef = model.coef_
            if np.any(np.isnan(coef)) or np.any(np.isinf(coef)):
                return False, "nan_or_inf_coefficients"
            if np.any(np.abs(coef) > 1e6):
                return False, "extremely_large_coefficients"
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            if np.any(np.isnan(importance)) or np.any(np.isinf(importance)):
                return False, "nan_or_inf_feature_importance"
        
        # 3. Check predictions for NaN or infinite values
        if predictions is not None:
            if np.any(np.isnan(predictions)) or np.any(np.isinf(predictions)):
                return False, "nan_or_inf_predictions"
            
            # For classification, check if only one class is predicted
            if self.task_type == TaskType.CLASSIFICATION:
                unique_predictions = np.unique(predictions)
                if len(unique_predictions) == 1:
                    return False, "single_class_prediction"
                
                # Check for zero variance in predictions
                if np.var(predictions) == 0:
                    return False, "zero_variance_predictions"
        
        # 4. Check for extreme overfitting instability
        if train_score is not None and cv_score is not None:
            overfit_gap = abs(train_score - cv_score)
            if overfit_gap > 0.5:  # Extremely high overfitting
                return False, "extreme_overfitting"
            
            # Check for nonsensical scores
            if cv_score <= 0 and train_score > 0.5:
                return False, "invalid_cv_score"
        
        # 5. Try making a small prediction to ensure model works
        try:
            if X_train.shape[0] > 0:
                test_pred = model.predict(X_train[:min(5, X_train.shape[0])])
                if np.any(np.isnan(test_pred)) or np.any(np.isinf(test_pred)):
                    return False, "unstable_predictions"
        except Exception as e:
            return False, f"prediction_error: {str(e)[:50]}"
        
        return True, None
    
    def _get_cv_splitter(self, y: np.ndarray):
        """Get appropriate cross-validation splitter"""
        if self.task_type == TaskType.CLASSIFICATION:
            return StratifiedKFold(
                n_splits=self.cv_folds, 
                shuffle=True, 
                random_state=self.random_state
            )
        else:
            return KFold(
                n_splits=self.cv_folds, 
                shuffle=True, 
                random_state=self.random_state
            )
    
    def _get_optimization_metric(self) -> str:
        """Get the metric to optimize during training"""
        if self.task_type == TaskType.CLASSIFICATION:
            return 'f1_weighted'
        elif self.task_type == TaskType.REGRESSION:
            return 'r2'
        else:
            return 'silhouette'
    
    def _define_search_space(self, trial: optuna.Trial, model_name: str) -> Dict[str, Any]:
        """
        Define hyperparameter search space for each model.
        Uses Optuna's suggest_* methods for wide parameter ranges.
        """
        
        # ===== CLASSIFICATION MODELS =====
        if model_name == 'logistic_regression':
            return {
                'C': trial.suggest_float('C', 0.001, 100, log=True),
                'max_iter': trial.suggest_int('max_iter', 100, 1000),
                'solver': trial.suggest_categorical('solver', ['lbfgs', 'liblinear', 'saga']),
                'penalty': trial.suggest_categorical('penalty', ['l2', 'l1', None]),
                'random_state': self.random_state
            }
        
        elif model_name == 'decision_tree':
            if self.task_type == TaskType.CLASSIFICATION:
                criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
            else:  # REGRESSION
                criterion = trial.suggest_categorical('criterion', ['squared_error', 'friedman_mse', 'absolute_error'])
            
            return {
                'max_depth': trial.suggest_int('max_depth', 3, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'criterion': criterion,
                'random_state': self.random_state
            }
        
        elif model_name == 'random_forest':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 30),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                'random_state': self.random_state
            }
        
        elif model_name == 'gradient_boosting':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'random_state': self.random_state
            }
        
        elif model_name == 'xgboost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'gamma': trial.suggest_float('gamma', 0.0, 1.0),
                'random_state': self.random_state,
                'eval_metric': 'logloss',
                'use_label_encoder': False,
                'n_jobs': self.max_cpu_cores  # Limit CPU cores
            }
            return params
        
        elif model_name == 'lightgbm':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.3, log=True),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'num_leaves': trial.suggest_int('num_leaves', 20, 150),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
                'random_state': self.random_state,
                'verbose': -1,
                'n_jobs': self.max_cpu_cores  # Limit CPU cores
            }
            return params
        
        elif model_name == 'svm':
            return {
                'C': trial.suggest_float('C', 0.001, 100, log=True),
                'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid']),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                'probability': True,  # Enable probability estimates
                'random_state': self.random_state
            }
        
        elif model_name == 'knn':
            return {
                'n_neighbors': trial.suggest_int('n_neighbors', 3, 30),
                'weights': trial.suggest_categorical('weights', ['uniform', 'distance']),
                'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'minkowski']),
                'p': trial.suggest_int('p', 1, 3)
            }
        
        elif model_name == 'adaboost':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 2.0, log=True),
                'random_state': self.random_state
            }
            # Only add algorithm for classification (not supported in regression)
            if self.task_type == TaskType.CLASSIFICATION:
                params['algorithm'] = 'SAMME'  # Suppress FutureWarning
            return params
        
        # ===== REGRESSION MODELS =====
        elif model_name == 'ridge':
            return {
                'alpha': trial.suggest_float('alpha', 0.001, 100, log=True),
                'random_state': self.random_state
            }
        
        elif model_name == 'lasso':
            return {
                'alpha': trial.suggest_float('alpha', 0.001, 10, log=True),
                'random_state': self.random_state
            }
        
        elif model_name == 'elastic_net':
            return {
                'alpha': trial.suggest_float('alpha', 0.001, 10, log=True),
                'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0),
                'random_state': self.random_state
            }
        
        elif model_name == 'svr':
            return {
                'C': trial.suggest_float('C', 0.001, 100, log=True),
                'kernel': trial.suggest_categorical('kernel', ['rbf', 'poly', 'sigmoid']),
                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
                'epsilon': trial.suggest_float('epsilon', 0.01, 1.0)
            }
        
        # ===== CLUSTERING MODELS =====
        elif model_name == 'kmeans':
            return {
                'n_clusters': trial.suggest_int('n_clusters', 2, 10),
                'max_iter': trial.suggest_int('max_iter', 100, 500),
                'n_init': trial.suggest_int('n_init', 5, 20),
                'random_state': self.random_state
            }
        
        elif model_name == 'dbscan':
            return {
                'eps': trial.suggest_float('eps', 0.1, 2.0),
                'min_samples': trial.suggest_int('min_samples', 2, 20),
                'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan'])
            }
        
        elif model_name == 'agglomerative':
            return {
                'n_clusters': trial.suggest_int('n_clusters', 2, 10),
                'linkage': trial.suggest_categorical('linkage', ['ward', 'complete', 'average'])
            }
        
        else:
            return {}
    
    def _create_objective(
        self, 
        model_name: str, 
        model_class,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        timeout_flag: list,
        cancellation_callback: Optional[callable] = None,
        force_cancel_flag: Optional[list] = None,
        skip_callback: Optional[callable] = None,
        trial_progress_callback: Optional[callable] = None
    ):
        """
        Create Optuna objective function for a specific model.
        """
        if force_cancel_flag is None:
            force_cancel_flag = [False]
            
        def objective(trial: optuna.Trial) -> float:
            # FIRST THING: Check force cancel flag - if True, timeout is 0 seconds
            trial_timeout = 0 if force_cancel_flag[0] else 120
            
            if force_cancel_flag[0] and self.verbose:
                print(f"  [FORCE CANCEL] {model_name}: force_cancel_flag is True - trial will timeout immediately")
            
            # Check for SKIP before doing anything
            if skip_callback and skip_callback():
                if self.verbose:
                    print(f"⏭️  [OBJECTIVE START] Skip detected for {model_name} - returning -9999")
                timeout_flag[0] = True
                return -9999

            # SECOND: Check for cancellation before doing ANYTHING
            if cancellation_callback and cancellation_callback():
                if self.verbose:
                    print(f"❌ [OBJECTIVE START] Cancellation detected - returning -9999 immediately")
                timeout_flag[0] = True  # Set timeout flag to stop study
                return -9999
            
            # Report trial progress
            if trial_progress_callback:
                try:
                    best_val = trial.study.best_value
                except ValueError:
                    best_val = None
                trial_progress_callback(trial.number + 1, self.n_trials, best_val)

            # Start trial timer
            trial_start_time = datetime.now()
            
            # Check cancellation again before getting params
            if cancellation_callback and cancellation_callback():
                if self.verbose:
                    print(f"❌ [OBJECTIVE PARAMS] Cancellation detected - returning -9999")
                timeout_flag[0] = True
                return -9999
            
            # Get hyperparameters
            params = self._define_search_space(trial, model_name)
            
            # Handle incompatible parameter combinations
            if model_name == 'logistic_regression':
                if params['penalty'] is None:
                    params.pop('C', None)
                elif params['penalty'] == 'l1' and params['solver'] == 'lbfgs':
                    params['solver'] = 'liblinear'
            
            # Check cancellation before creating model
            if cancellation_callback and cancellation_callback():
                if self.verbose:
                    print(f"❌ [OBJECTIVE MODEL] Cancellation detected - returning -9999")
                timeout_flag[0] = True
                return -9999
            
            try:
                # Capture warnings during training
                import warnings
                
                # Create model
                model = model_class(**params)
                
                # Check cancellation after model creation
                if cancellation_callback and cancellation_callback():
                    if self.verbose:
                        print(f"❌ [OBJECTIVE FIT] Cancellation detected before fit - returning -9999")
                    timeout_flag[0] = True
                    return -9999
                
                # Cross-validation
                if self.task_type == TaskType.CLUSTERING:
                    # For clustering, train and evaluate
                    model.fit(X_train)
                    if hasattr(model, 'labels_'):
                        labels = model.labels_
                    else:
                        labels = model.predict(X_train)
                    
                    # Calculate silhouette score
                    if len(np.unique(labels)) > 1:
                        score = silhouette_score(X_train, labels)
                    else:
                        score = -1.0
                else:
                    # Check cancellation one more time before expensive subprocess spawn
                    if cancellation_callback and cancellation_callback():
                        if self.verbose:
                            print(f"  [CANCELLED] {model_name}: Skipping trial - cancellation detected")
                        timeout_flag[0] = True
                        return -9999
                    
                    # If force cancel flag is set, immediately return -9999 without spawning subprocess
                    if force_cancel_flag[0]:
                        if self.verbose:
                            print(f"  [FORCE CANCEL] {model_name}: Skipping trial - force cancel active")
                        timeout_flag[0] = True
                        return -9999
                    
                    # Supervised learning - use subprocess with hard timeout and temp files
                    # Create temp files for data and results
                    data_file = tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.joblib')
                    result_file = tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.joblib')
                    data_file.close()
                    result_file.close()
                    
                    try:
                        # Save data to temp file
                        joblib.dump({
                            'X_train': X_train,
                            'y_train': y_train,
                            'X_test': X_test,
                            'y_test': y_test
                        }, data_file.name)
                        
                        # Spawn subprocess for training
                        process = Process(
                            target=_train_model_subprocess,
                            args=(
                                model_class,
                                params,
                                data_file.name,
                                result_file.name,
                                self.task_type,
                                self.cv_folds,
                                self.random_state
                            )
                        )
                        
                        process.start()
                        
                        # Wait with timeout
                        timeout_seconds = trial_timeout if trial_timeout > 0 else 0.1
                        process.join(timeout=timeout_seconds)
                        
                        # Check if process is still alive (timeout occurred)
                        if process.is_alive():
                            # Hard kill the process
                            process.terminate()
                            process.join(timeout=2)  # Give it 2 seconds to terminate gracefully
                            
                            if process.is_alive():
                                # Force kill if still alive
                                process.kill()
                                process.join()
                            
                            if trial_timeout == 0:
                                if self.verbose:
                                    print(f"  [FORCE CANCEL] {model_name}: Trial force cancelled (hard_timeout_kill)")
                            else:
                                print(f"  [TIMEOUT] {model_name}: Hard timeout after {timeout_seconds}s (hard_timeout_kill)")
                            
                            timeout_flag[0] = True
                            return -9999
                        
                        # Process finished - read results from file
                        if os.path.exists(result_file.name) and os.path.getsize(result_file.name) > 0:
                            result = joblib.load(result_file.name)
                            
                            if not result['success']:
                                print(f"  [ERROR] {model_name}: {result['error']}")
                                return -9999
                            
                            # Get score from subprocess
                            score = result['cv_score']
                            
                            # Return CV score for optimization
                            return score
                        else:
                            print(f"  [ERROR] {model_name}: No results from subprocess")
                            return -9999
                            
                    finally:
                        # Clean up temp files
                        try:
                            if os.path.exists(data_file.name):
                                os.unlink(data_file.name)
                            if os.path.exists(result_file.name):
                                os.unlink(result_file.name)
                        except:
                            pass
                
                # For clustering, return score directly
                return score
                
            except Exception as e:
                # Return worst possible score if training fails
                error_msg = str(e).replace('\n', ' ')[:200]
                print(f"  [ERROR] {model_name}: {error_msg}")
                return -9999
        
        return objective
    
    def _calculate_overfitting_metrics(
        self,
        train_score: float,
        cv_score: float
    ) -> Tuple[float, float, float, bool, bool, Optional[str]]:
        """
        Calculate overfitting metrics and determine if model should be rejected.
        
        Returns:
            Tuple of (overfit_gap, penalty, generalization_score, 
                     overfitting_flag, rejected, rejection_reason)
        """
        overfit_gap = abs(train_score - cv_score)
        
        # Determine rejection and penalty
        rejected = False
        rejection_reason = None
        
        if overfit_gap > self.overfit_threshold_reject:
            rejected = True
            rejection_reason = f"Overfit gap {overfit_gap:.4f} exceeds threshold {self.overfit_threshold_reject}"
            penalty_multiplier = 5.0  # Severe penalty
        elif overfit_gap > self.overfit_threshold_high:
            penalty_multiplier = 3.0  # High penalty
        else:
            penalty_multiplier = self.penalty_factor  # Normal penalty
        
        penalty = overfit_gap * penalty_multiplier
        generalization_score = cv_score - penalty
        
        overfitting = overfit_gap > 0.05  # Flag if gap > 5%
        
        return overfit_gap, penalty, generalization_score, overfitting, rejected, rejection_reason
    
    def _train_single_model(
        self,
        model_name: str,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        feature_names: List[str],
        cancellation_callback: Optional[callable] = None,
        force_cancel_flag: Optional[list] = None,
        skip_callback: Optional[callable] = None,
        trial_progress_callback: Optional[callable] = None
    ) -> ModelResult:
        """
        Train a single model with Optuna optimization and evaluate it.
        """
        if force_cancel_flag is None:
            force_cancel_flag = [False]
            
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Training: {model_name.upper()}")
            print(f"{'='*60}")
        
        model_class = self.models[model_name]
        start_time = datetime.now()
        
        # Create Optuna study
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state),
            pruner=MedianPruner()
        )
        
        # Early stopping callback for unstable models
        consecutive_failures = [0]  # Use list to allow modification in nested function
        max_consecutive_failures = 5  # Stop after 5 consecutive unstable trials
        stopped_early = [False]  # Track if we stopped early
        timeout_occurred = [False]  # Track if timeout occurred - shared with objective
        
        def check_early_stop(study: optuna.Study, trial: optuna.Trial):
            """Stop optimization if too many consecutive unstable trials or timeout"""
            # Check for user cancellation FIRST - highest priority
            if cancellation_callback and cancellation_callback():
                if self.verbose:
                    print(f"\n❌ [CALLBACK] Cancellation detected - stopping {model_name} study immediately")
                stopped_early[0] = True
                study.stop()
                return
            
            # Check for SKIP request
            if skip_callback and skip_callback():
                if self.verbose:
                    print(f"\n⏭️  [CALLBACK] Skip detected - stopping {model_name} study")
                stopped_early[0] = True
                study.stop()
                return

            # Check if timeout flag was set by objective function
            if timeout_occurred[0]:
                if self.verbose:
                    print(f"\n⏱️  Trial timeout - skipping remaining trials for {model_name}")
                    print(f"   Moving to next model...\n")
                stopped_early[0] = True
                study.stop()
                return
            
            # Check if trial was pruned
            if trial.state == optuna.trial.TrialState.PRUNED:
                return
            
            if trial.value == -9999:
                consecutive_failures[0] += 1
                if consecutive_failures[0] >= max_consecutive_failures:
                    if self.verbose:
                        print(f"\n❌ Stopping early: {consecutive_failures[0]} consecutive unstable trials")
                        print(f"   Moving to next model...\n")
                    stopped_early[0] = True
                    study.stop()
            else:
                if consecutive_failures[0] > 0 and self.verbose:
                    print(f"   ✓ Trial {trial.number} succeeded (score: {trial.value:.4f})")
                consecutive_failures[0] = 0  # Reset on success
        
        # Optimize
        objective = self._create_objective(model_name, model_class, X_train, y_train, X_test, y_test, timeout_occurred, cancellation_callback, force_cancel_flag, skip_callback, trial_progress_callback)
        
        # Check if cancelled before starting optimization
        if cancellation_callback and cancellation_callback():
            if self.verbose:
                print(f"\n❌ [PRE-OPTIMIZE] Cancellation detected - skipping {model_name} optimization")
            # Return immediately with -9999
            return ModelResult(
                model_name=model_name,
                model_type=model_class.__name__,
                task_type=self.task_type,
                best_model=None,
                best_params={},
                cv_score=-9999,
                test_score=-9999,
                generalization_score=-9999,
                overfit_gap=0.0,
                training_time=0.0,
                feature_importances={},
                predictions=np.array([]),
                cv_scores=[],
                stability_status="cancelled",
                stability_reason="Training cancelled by user"
            )
        
        try:
            # If cancelled, only run 1 trial (which will immediately return -9999)
            trials_to_run = 1 if (cancellation_callback and cancellation_callback()) else self.n_trials
            
            study.optimize(
                objective, 
                n_trials=trials_to_run,
                show_progress_bar=False,
                n_jobs=1,  # Avoid conflicts with CV parallelization
                callbacks=[check_early_stop]
            )
        except optuna.exceptions.OptunaError:
            # Study was stopped early - this is expected behavior
            pass
        
        optimization_time = (datetime.now() - start_time).total_seconds()
        
        # Check if skipped during training
        if skip_callback and skip_callback():
            if self.verbose:
                print(f"\n⏭️  Model {model_name} was skipped by user")
            return ModelResult(
                model_name=model_name,
                model_type=model_class.__name__,
                task_type=self.task_type,
                best_model=None,
                best_params={},
                cv_score=-9999,
                test_score=-9999,
                generalization_score=-9999,
                overfit_gap=0.0,
                training_time=optimization_time,
                feature_importances={},
                predictions=np.array([]),
                cv_scores=[],
                stability_status="skipped",
                stability_reason="Model skipped by user"
            )

        # Check if cancelled during training
        if cancellation_callback and cancellation_callback():
            if self.verbose:
                print(f"\n❌ Training cancelled during {model_name} - returning -9999")
            return ModelResult(
                model_name=model_name,
                model_type=model_class.__name__,
                task_type=self.task_type,
                best_model=None,
                best_params={},
                cv_score=-9999,
                test_score=-9999,
                generalization_score=-9999,
                overfit_gap=0.0,
                training_time=optimization_time,
                feature_importances={},
                predictions=np.array([]),
                cv_scores=[],
                stability_status="cancelled",
                stability_reason="Training cancelled by user"
            )
        
        # Check if stopped early with no good trials OR all trials failed OR timeout
        best_value_invalid = (not study.best_trial or 
                             study.best_value is None or 
                             study.best_value <= -9999)
        
        if best_value_invalid or timeout_occurred[0] or (stopped_early[0] and len(study.trials) < 5):
            if self.verbose and not timeout_occurred[0]:
                print(f"\n❌ All trials unstable - skipping {model_name}\n")
            # Return a dummy failed result
            rejection_reason = "Trial timeout exceeded 120s" if timeout_occurred[0] else "All trials unstable"
            return ModelResult(
                model_name=model_name,
                model_type=model_class.__name__,
                task_type=self.task_type,
                best_model=None,
                best_params={},
                train_score=-1.0,
                cv_score=-1.0,
                cv_std=0.0,
                test_score=-1.0,
                overfit_gap=0.0,
                penalty=0.0,
                generalization_score=-1.0,
                overfitting=False,
                rejected=True,
                rejection_reason=rejection_reason,
                stability_status="unstable",
                stability_reason="No stable configuration found",
                detailed_metrics={},
                feature_importance=None,
                n_trials=len(study.trials),
                best_trial_number=-1,
                optimization_time=optimization_time
            )
        
        # Get best parameters
        best_params = study.best_params
        best_trial = study.best_trial
        
        if self.verbose:
            print(f"Best trial: {best_trial.number}")
            print(f"Best CV score: {best_trial.value:.4f}")
            print(f"Best params: {best_params}")
        
        # Handle parameter compatibility
        if model_name == 'logistic_regression':
            if best_params.get('penalty') == 'none':
                best_params.pop('C', None)
            elif best_params.get('penalty') == 'l1' and best_params.get('solver') == 'lbfgs':
                best_params['solver'] = 'liblinear'
        
        # Train final model with best parameters
        best_model = model_class(**best_params)
        best_model.fit(X_train, y_train)
        
        # ===== CALCULATE SCORES =====
        if self.task_type == TaskType.CLASSIFICATION:
            # Train score
            train_pred = best_model.predict(X_train)
            train_score = f1_score(y_train, train_pred, average='weighted', zero_division=0)
            
            # CV score
            cv = self._get_cv_splitter(y_train)
            cv_scores = cross_val_score(
                best_model, X_train, y_train,
                cv=cv, scoring='f1_weighted', n_jobs=-1
            )
            cv_score = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Test score
            test_pred = best_model.predict(X_test)
            test_score = f1_score(y_test, test_pred, average='weighted', zero_division=0)
            
            # Detailed metrics
            detailed_metrics = {
                'accuracy': accuracy_score(y_test, test_pred),
                'precision': precision_score(y_test, test_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_test, test_pred, average='weighted', zero_division=0),
                'f1_score': test_score,
                'confusion_matrix': confusion_matrix(y_test, test_pred).tolist(),
                'classification_report': classification_report(y_test, test_pred, zero_division=0, output_dict=True)
            }
        
        elif self.task_type == TaskType.REGRESSION:
            # Train score
            train_pred = best_model.predict(X_train)
            train_score = r2_score(y_train, train_pred)
            
            # CV score
            cv = self._get_cv_splitter(y_train)
            cv_scores = cross_val_score(
                best_model, X_train, y_train,
                cv=cv, scoring='r2', n_jobs=-1
            )
            cv_score = cv_scores.mean()
            cv_std = cv_scores.std()
            
            # Test score
            test_pred = best_model.predict(X_test)
            test_score = r2_score(y_test, test_pred)
            
            # Detailed metrics
            detailed_metrics = {
                'r2': test_score,
                'mae': mean_absolute_error(y_test, test_pred),
                'mse': mean_squared_error(y_test, test_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
                'predictions': test_pred.tolist(),
                'actuals': y_test.tolist()
            }
        
        else:  # CLUSTERING
            if hasattr(best_model, 'labels_'):
                labels = best_model.labels_
            else:
                labels = best_model.predict(X_train)
            
            n_clusters = len(np.unique(labels))
            
            if n_clusters > 1:
                train_score = silhouette_score(X_train, labels)
                cv_score = train_score  # No CV for clustering
                cv_std = 0.0
                test_score = train_score
                train_pred = labels  # For stability check
                
                detailed_metrics = {
                    'silhouette_score': train_score,
                    'calinski_harabasz_score': calinski_harabasz_score(X_train, labels),
                    'davies_bouldin_score': davies_bouldin_score(X_train, labels),
                    'n_clusters': n_clusters,
                    'labels': labels.tolist()
                }
            else:
                train_score = -1.0
                cv_score = -1.0
                cv_std = 0.0
                test_score = -1.0
                train_pred = labels
                detailed_metrics = {'error': 'Single cluster found'}
        
        # ===== STABILITY CHECK =====
        stability_status = "stable"
        stability_reason = None
        
        if self.task_type != TaskType.CLUSTERING:
            is_stable, reason = self._check_model_stability(
                best_model, X_train, y_train, train_pred, train_score, cv_score
            )
            if not is_stable:
                stability_status = "unstable"
                stability_reason = reason
                if self.verbose:
                    print(f"\n⚠️  Stability Warning ({model_name}): {reason}")
        
        # ===== OVERFITTING DETECTION =====
        overfit_gap, penalty, generalization_score, overfitting, rejected, rejection_reason = \
            self._calculate_overfitting_metrics(train_score, cv_score)
        
        if self.verbose:
            print(f"\n📊 Scores:")
            print(f"  Train: {train_score:.4f}")
            print(f"  CV:    {cv_score:.4f} (±{cv_std:.4f})")
            print(f"  Test:  {test_score:.4f}")
            print(f"\n🎯 Generalization Metrics:")
            print(f"  Overfit Gap:          {overfit_gap:.4f}")
            print(f"  Penalty:              {penalty:.4f}")
            print(f"  Generalization Score: {generalization_score:.4f}")
            print(f"  Overfitting:          {overfitting}")
            print(f"  Rejected:             {rejected}")
            if rejected:
                print(f"  Reason:               {rejection_reason}")
        
        # ===== FEATURE IMPORTANCE =====
        feature_importance = None
        if hasattr(best_model, 'feature_importances_'):
            importances = best_model.feature_importances_
            feature_importance = {
                name: float(imp) 
                for name, imp in zip(feature_names, importances)
            }
            feature_importance = dict(sorted(
                feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
        elif hasattr(best_model, 'coef_'):
            coef = np.abs(best_model.coef_[0]) if len(best_model.coef_.shape) > 1 else np.abs(best_model.coef_)
            feature_importance = {
                name: float(imp) 
                for name, imp in zip(feature_names, coef)
            }
            feature_importance = dict(sorted(
                feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            ))
        
        # Create result object
        result = ModelResult(
            model_name=model_name,
            model_type=model_class.__name__,
            task_type=self.task_type,
            best_model=best_model,
            best_params=best_params,
            train_score=train_score,
            cv_score=cv_score,
            cv_std=cv_std,
            test_score=test_score,
            overfit_gap=overfit_gap,
            penalty=penalty,
            generalization_score=generalization_score,
            overfitting=overfitting,
            rejected=rejected,
            rejection_reason=rejection_reason,
            stability_status=stability_status,
            stability_reason=stability_reason,
            detailed_metrics=detailed_metrics,
            feature_importance=feature_importance,
            n_trials=self.n_trials,
            best_trial_number=best_trial.number,
            optimization_time=optimization_time
        )
        
        return result
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[List[str]] = None,
        model_subset: Optional[List[str]] = None,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None,
        cancellation_callback: Optional[callable] = None,
        force_cancel_flag: Optional[list] = None,
        skip_callback: Optional[callable] = None,
        on_model_start: Optional[callable] = None,
        on_model_complete: Optional[callable] = None,
        trial_progress_callback: Optional[callable] = None
    ) -> AutoMLResult:
        """
        Train all models and select the best one based on generalization score.
        
        Args:
            X: Feature matrix (training data if X_test provided, otherwise will be split)
            y: Target vector
            feature_names: List of feature names
            model_subset: Optional list of model names to train (train all if None)
            X_test: Optional pre-split test features
            y_test: Optional pre-split test targets
            cancellation_callback: Optional function that returns True if training should be cancelled
            force_cancel_flag: Optional list [bool] that when True makes all trials timeout immediately (0 seconds)
        
        Returns:
            AutoMLResult with best model and all results
        """
        if force_cancel_flag is None:
            force_cancel_flag = [False]
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        # Store callbacks for use in training loop
        self._skip_callback = skip_callback
        self._on_model_start = on_model_start
        self._on_model_complete = on_model_complete
        self._trial_progress_callback = trial_progress_callback
        
        # Use pre-split data if provided, otherwise split the data
        if X_test is not None and y_test is not None:
            if self.verbose:
                print(f"✅ Using provided pre-split test data")
            X_train, y_train = X, y
        else:
            if self.verbose:
                print(f"📊 Splitting data (test_size={self.test_size})")
            # Split data
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.test_size, 
                random_state=self.random_state,
                stratify=y if self.task_type == TaskType.CLASSIFICATION else None
            )
        
        if self.verbose:
            print(f"\n🚀 AutoML Training Started")
            print(f"Task Type: {self.task_type.value}")
            print(f"Training samples: {len(X_train)}")
            print(f"Test samples: {len(X_test)}")
            print(f"Features: {len(feature_names)}")
            print(f"Optuna trials per model: {self.n_trials}")
            print(f"CV folds: {self.cv_folds}")
        
        # Determine which models to train
        models_to_train = model_subset if model_subset else list(self.models.keys())
        
        # Train all models
        for model_name in models_to_train:
            try:
                # Check if training was cancelled - return -9999 for all remaining models
                if cancellation_callback and cancellation_callback():
                    if self.verbose:
                        print(f"\n❌ [LOOP] Training cancelled - returning -9999 for {model_name}")
                    
                    # Create a fake result with -9999 score to skip this model
                    model_class = self.models[model_name]
                    result = ModelResult(
                        model_name=model_name,
                        model_type=model_class.__name__,
                        cv_score=-9999,
                        test_score=-9999,
                        generalization_score=-9999,
                        overfit_gap=0.0,
                        best_params={},
                        training_time=0.0,
                        feature_importances={},
                        predictions=np.array([]),
                        cv_scores=[],
                        stability_status="cancelled",
                        stability_reason="Training cancelled by user"
                    )
                    self.results.append(result)
                    if self._on_model_complete:
                        self._on_model_complete(model_name, None, "cancelled")
                    continue
                
                # Notify model start
                if self._on_model_start:
                    self._on_model_start(model_name)

                # Check memory usage before training
                self._check_memory_usage()
                
                # Clear skip flag before each model
                if skip_callback and hasattr(skip_callback, '__self__'):
                    pass  # skip_callback is a bound method, clearing handled externally
                
                result = self._train_single_model(
                    model_name, X_train, y_train, X_test, y_test, feature_names,
                    cancellation_callback, force_cancel_flag,
                    skip_callback, trial_progress_callback
                )
                self.results.append(result)
                
                # Notify model complete
                if self._on_model_complete:
                    score = result.generalization_score if result.generalization_score != -9999 else None
                    status = "skipped" if (skip_callback and skip_callback()) else (
                        "cancelled" if result.stability_status == "cancelled" else "completed"
                    )
                    self._on_model_complete(model_name, score, status)
                
                # Force garbage collection after each model to free memory
                gc.collect()
                
            except Exception as e:
                if self.verbose:
                    print(f"❌ Error training {model_name}: {str(e)}")
                # Notify model failed so progress tracking stays accurate
                if self._on_model_complete:
                    self._on_model_complete(model_name, None, "failed")
                continue
        
        # Select best model based on generalization score
        # Filter out failed models (where best_model is None)
        valid_results = [r for r in self.results if not r.rejected and r.best_model is not None]
        
        # Check if training was cancelled - if so, all models will have -9999
        cancelled_results = [r for r in self.results if r.stability_status == "cancelled"]
        if cancelled_results and len(cancelled_results) == len(self.results):
            if self.verbose:
                print(f"\n❌ All models cancelled - returning cancelled result")
            # Return a dummy result for the first model
            best_result = self.results[0] if self.results else None
            if best_result is None:
                raise ValueError("Training was cancelled before any models could be processed")
            
            return AutoMLResult(
                best_model=best_result,
                all_models=self.results,
                total_models_evaluated=0,
                models_rejected=len(self.results)
            )
        
        if not valid_results:
            # If all models rejected, try to find any non-None model
            non_none_results = [r for r in self.results if r.best_model is not None]
            
            if not non_none_results:
                # All models completely failed
                raise ValueError(
                    "All models failed to train. This might be due to:\n"
                    "  - Data quality issues (too many missing values, outliers)\n"
                    "  - Insufficient samples for the number of features\n"
                    "  - Convergence issues (try increasing max_iter)\n"
                    "  - Memory constraints"
                )
            
            # If all models rejected, select the one with best generalization score anyway
            # but issue a warning
            if self.verbose:
                print(f"\n⚠️  WARNING: All {len(non_none_results)} models were rejected due to overfitting.")
                print(f"   Selecting the model with the smallest overfit gap as a fallback.")
                print(f"   Consider:")
                print(f"   - Increasing overfit_threshold_reject (currently {self.overfit_threshold_reject})")
                print(f"   - Reducing penalty_factor (currently {self.penalty_factor})")
                print(f"   - Collecting more training data")
                print(f"   - Simplifying your model selection\n")
            
            # Select model with smallest overfit gap (least overfit)
            self.best_result = min(non_none_results, key=lambda x: x.overfit_gap)
            self.best_result.rejected = False  # Override rejection for fallback
            all_rejected = True
        else:
            self.best_result = max(valid_results, key=lambda x: x.generalization_score)
            all_rejected = False
        
        if self.verbose:
            print(f"\n{'='*60}")
            if all_rejected:
                print(f"⚠️  FALLBACK MODEL SELECTED (All models rejected)")
            else:
                print(f"🏆 BEST MODEL SELECTED")
            print(f"{'='*60}")
            print(f"Model: {self.best_result.model_name}")
            print(f"Generalization Score: {self.best_result.generalization_score:.4f}")
            print(f"CV Score: {self.best_result.cv_score:.4f}")
            print(f"Test Score: {self.best_result.test_score:.4f}")
            print(f"Overfit Gap: {self.best_result.overfit_gap:.4f}")
            if all_rejected:
                print(f"⚠️  Note: This model was rejected but selected as best available")
            print(f"{'='*60}")
        
        # Create AutoML result
        automl_result = AutoMLResult(
            task_type=self.task_type,
            best_model=self.best_result,
            all_models=self.results,
            total_models_evaluated=len(self.results),
            models_rejected=len([r for r in self.results if r.rejected]),
            feature_names=feature_names
        )
        
        return automl_result
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using the best model"""
        if self.best_result is None:
            raise ValueError("No model trained. Call fit() first.")
        
        return self.best_result.best_model.predict(X)
    
    def save(self, filepath: str, **kwargs):
        """Save the best model and results"""
        if self.best_result is None:
            raise ValueError("No model trained. Call fit() first.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        save_data = {
            'best_model': self.best_result.best_model,
            'best_result': self.best_result,
            'all_results': self.results,
            'task_type': self.task_type,
            'label_encoder': kwargs.get('label_encoder'),
            'label_mapping': kwargs.get('label_mapping'),
            'preprocessing_pipeline': kwargs.get('preprocessing_pipeline'),  # Save preprocessor!
            'config': {
                'n_trials': self.n_trials,
                'cv_folds': self.cv_folds,
                'penalty_factor': self.penalty_factor,
                'overfit_threshold_reject': self.overfit_threshold_reject,
                'overfit_threshold_high': self.overfit_threshold_high
            }
        }
        
        joblib.dump(save_data, filepath)
    
    @staticmethod
    def load(filepath: str) -> 'AutoMLEngine':
        """Load a saved AutoML engine"""
        data = joblib.load(filepath)
        
        engine = AutoMLEngine(
            task_type=data['task_type'],
            n_trials=data['config']['n_trials'],
            cv_folds=data['config']['cv_folds'],
            penalty_factor=data['config']['penalty_factor'],
            overfit_threshold_reject=data['config']['overfit_threshold_reject'],
            overfit_threshold_high=data['config']['overfit_threshold_high']
        )
        
        engine.best_result = data['best_result']
        engine.results = data['all_results']
        
        return engine
