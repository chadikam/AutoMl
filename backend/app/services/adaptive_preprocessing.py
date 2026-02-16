"""
Adaptive Preprocessing Pipeline
Automatically prepares datasets for ML based on EDA insights and model context.

This service intelligently adapts preprocessing strategies by:
1. Analyzing EDA metadata to understand data characteristics
2. Detecting task type (classification, regression, unsupervised)
3. Choosing preprocessing strategies based on model type and data patterns
4. Handling all data issues automatically while logging decisions
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    OneHotEncoder, OrdinalEncoder, LabelEncoder,
    PowerTransformer, FunctionTransformer
)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import hstack, issparse
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN
from typing import Tuple, List, Dict, Any, Optional
from datetime import datetime
from enum import Enum
import joblib
import os
import re

# Optional: VIF calculation for advanced multicollinearity diagnostics
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False


class OutlierCapper(BaseEstimator, TransformerMixin):
    """
    Cap outliers using IQR method (Winsorization)
    Only caps values, doesn't remove rows
    Uses 2.5×IQR for more conservative outlier detection (not just 1.5×IQR)
    """
    def __init__(self, iqr_multiplier=2.5):
        self.iqr_multiplier = iqr_multiplier
        self.lower_bounds_ = None
        self.upper_bounds_ = None
        self.capped_columns_ = []
    
    def fit(self, X, y=None):
        """Calculate bounds for each column"""
        print(f"\n🔍 OutlierCapper.fit() called:")
        print(f"   Input type: {type(X)}")
        print(f"   Input shape: {X.shape if hasattr(X, 'shape') else 'N/A'}")
        if isinstance(X, pd.DataFrame):
            print(f"   Columns: {X.columns.tolist()}")
            X_array = X.values
            columns = X.columns
        else:
            X_array = X
            columns = [f"col_{i}" for i in range(X.shape[1])]
        
        print(f"   X_array shape: {X_array.shape}")
        print(f"   Sample data (first row): {X_array[0] if len(X_array) > 0 else 'N/A'}")
        
        self.lower_bounds_ = []
        self.upper_bounds_ = []
        self.capped_columns_ = []
        
        for i in range(X_array.shape[1]):
            col_data = X_array[:, i]
            q1 = np.percentile(col_data, 25)
            q3 = np.percentile(col_data, 75)
            iqr = q3 - q1
            
            # Fallback to standard deviation method if IQR is too small
            # This prevents marking everything as outlier when IQR ≈ 0
            if iqr < 1e-10:  # IQR essentially zero
                mean = np.mean(col_data)
                std = np.std(col_data)
                # Use mean ± 3 standard deviations (99.7% rule)
                lower_bound = mean - 3 * std
                upper_bound = mean + 3 * std
            else:
                lower_bound = q1 - (self.iqr_multiplier * iqr)
                upper_bound = q3 + (self.iqr_multiplier * iqr)
            
            self.lower_bounds_.append(lower_bound)
            self.upper_bounds_.append(upper_bound)
            
            # Check if any values will be capped
            if np.any(col_data < lower_bound) or np.any(col_data > upper_bound):
                self.capped_columns_.append(columns[i] if i < len(columns) else f"col_{i}")
        
        return self
    
    def transform(self, X):
        """Cap outliers to bounds"""
        print(f"\n🔍 OutlierCapper.transform() called:")
        print(f"   Input type: {type(X)}")
        print(f"   Input shape: {X.shape if hasattr(X, 'shape') else 'N/A'}")
        
        if isinstance(X, pd.DataFrame):
            print(f"   Columns: {X.columns.tolist()}")
            X_array = X.values.copy()
        else:
            X_array = X.copy()
        
        print(f"   X_array shape before clipping: {X_array.shape}")
        print(f"   Sample data (first row): {X_array[0] if len(X_array) > 0 else 'N/A'}")
        print(f"   Bounds: lower={self.lower_bounds_[:3]}, upper={self.upper_bounds_[:3]}")
        
        for i in range(X_array.shape[1]):
            X_array[:, i] = np.clip(
                X_array[:, i],
                self.lower_bounds_[i],
                self.upper_bounds_[i]
            )
        
        print(f"   Output shape: {X_array.shape}")
        print(f"   Sample output (first row): {X_array[0] if len(X_array) > 0 else 'N/A'}")
        
        return X_array


class TextVectorizer(BaseEstimator, TransformerMixin):
    """
    Custom text vectorizer for text columns.
    Handles multiple text columns and concatenates their TF-IDF features.
    """
    def __init__(self, max_features=5000, min_df=2, max_df=0.95):
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        self.vectorizers_ = {}
        self.feature_names_ = []
    
    def fit(self, X, y=None):
        """Fit TF-IDF vectorizers for each text column"""
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        for col in X_df.columns:
            # Fill NaN with empty string
            X_col = X_df[col].fillna('').astype(str)
            
            # Create vectorizer for this column
            vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                min_df=self.min_df,
                max_df=self.max_df,
                strip_accents='unicode',
                lowercase=True,
                ngram_range=(1, 2),
                stop_words='english'
            )
            
            try:
                vectorizer.fit(X_col)
                self.vectorizers_[col] = vectorizer
                
                # Store feature names
                feature_names = [f"{col}_tfidf_{i}" for i in range(len(vectorizer.get_feature_names_out()))]
                self.feature_names_.extend(feature_names)
                
                print(f"   ✓ TF-IDF fitted for '{col}': {len(feature_names)} features")
            except Exception as e:
                print(f"   ⚠️ Warning: Could not vectorize '{col}': {str(e)}")
        
        return self
    
    def transform(self, X):
        """Transform text columns to TF-IDF features"""
        X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        
        sparse_matrices = []
        for col in X_df.columns:
            if col in self.vectorizers_:
                X_col = X_df[col].fillna('').astype(str)
                X_transformed = self.vectorizers_[col].transform(X_col)
                sparse_matrices.append(X_transformed)
        
        if sparse_matrices:
            # Concatenate all sparse matrices
            return hstack(sparse_matrices)
        else:
            # Return empty sparse matrix if no text columns
            from scipy.sparse import csr_matrix
            return csr_matrix((len(X_df), 0))


class TaskType(str, Enum):
    """Machine learning task types"""
    CLASSIFICATION = "classification"
    REGRESSION = "regression"
    UNSUPERVISED = "unsupervised"


class ModelFamily(str, Enum):
    """Model family categories for preprocessing decisions"""
    LINEAR = "linear"  # Logistic Regression, Linear Regression
    TREE_BASED = "tree_based"  # Decision Trees, Random Forest, Gradient Boosting
    DISTANCE_BASED = "distance_based"  # KNN, SVM
    NEURAL_NETWORK = "neural_network"  # Neural Networks
    UNKNOWN = "unknown"  # Unsupervised or unspecified


class AdaptivePreprocessor:
    """
    Intelligent preprocessing pipeline that adapts to data and model context
    """
    
    # Model type to family mapping
    MODEL_FAMILY_MAP = {
        "logistic_regression": ModelFamily.LINEAR,
        "linear_regression": ModelFamily.LINEAR,
        "decision_tree": ModelFamily.TREE_BASED,
        "random_forest": ModelFamily.TREE_BASED,
        "gradient_boosting": ModelFamily.TREE_BASED,
        "xgboost": ModelFamily.TREE_BASED,
        "knn": ModelFamily.DISTANCE_BASED,
        "svm": ModelFamily.DISTANCE_BASED,
        "neural_network": ModelFamily.NEURAL_NETWORK,
    }
    
    # Thresholds for decision making
    MISSING_DROP_THRESHOLD = 0.6  # Drop columns with >60% missing
    MISSING_HIGH_THRESHOLD = 0.3  # High missing if >30%
    OUTLIER_SIGNIFICANT_THRESHOLD = 0.1  # Significant outliers if >10%
    HIGH_CARDINALITY_THRESHOLD = 50  # High cardinality if >50 unique values (use ordinal encoding)
    CONSTANT_THRESHOLD = 1  # Drop if ≤1 unique value
    LOW_VARIANCE_THRESHOLD = 0.95  # Drop if most common value appears >95% of time
    ID_UNIQUENESS_THRESHOLD = 0.95  # ID-like if >95% unique values
    
    def __init__(self, eda_results: Optional[Dict[str, Any]] = None, outlier_preferences: Optional[Dict[str, str]] = None):
        """
        Initialize adaptive preprocessor
        
        Args:
            eda_results: Optional EDA analysis results for intelligent decisions
            outlier_preferences: Optional dict of {column_name: 'keep'|'cap'|'remove'}
        """
        self.eda_results = eda_results or {}
        self.outlier_preferences = outlier_preferences or {}
        self.task_type = None
        self.model_family = ModelFamily.UNKNOWN
        self.target_column = None
        
        # Preprocessing artifacts
        self.pipeline = None
        self.label_encoder = None  # For target encoding
        self.feature_names = []
        self.numerical_features = []
        self.categorical_features = []
        self.text_features = []  # NEW: Text columns for NLP
        self.datetime_features = []
        self.dropped_columns = []
        
        # Columns to drop
        self.id_columns = []
        self.constant_columns = []
        self.low_variance_columns = []
        self.constant_column_details = {}  # Store details: {col: {value, percentage, count}}
        self.high_missing_columns = []
        self.high_cardinality_columns = []
        
        # Outlier handling tracking
        self.outliers_handled_count = 0
        self.outlier_columns = []
        self.outlier_details = []  # Detailed outlier information
        self.outlier_rows_removed = 0  # Count of rows removed due to outliers
        
        # Imputation tracking
        self.imputation_details = []
        
        # Encoding tracking
        self.encoding_details = []
        
        # Class imbalance tracking
        self.class_imbalance_handled = False
        self.imbalance_strategy = None
        self.original_class_distribution = {}
        self.resampled_class_distribution = {}
        
        # Duplicate removal tracking
        self.duplicate_removal_details = []
        self.duplicates_removed_count = 0
        
        # Target handling tracking
        self.nan_target_rows_removed = 0
        
        # Skewness handling tracking
        self.skewness_details = []
        self.skewed_features_transformed = []
        
        # Decision log
        self.decision_log = []
        self.preprocessing_metadata = {}
        
    def _log_decision(self, category: str, decision: str, reason: str, impact: str = "info"):
        """
        Log a preprocessing decision with reasoning
        
        Args:
            category: Decision category (e.g., "feature_removal", "imputation")
            decision: The decision made
            reason: Why this decision was made
            impact: Impact level (info, warning, critical)
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "category": category,
            "decision": decision,
            "reason": reason,
            "impact": impact
        }
        self.decision_log.append(log_entry)
        print(f"[{impact.upper()}] {category}: {decision} - {reason}")
    
    def detect_task_type(self, df: pd.DataFrame, target_column: Optional[str] = None) -> TaskType:
        """
        Detect ML task type based on target column
        
        Args:
            df: Input DataFrame
            target_column: Target column name (None for unsupervised)
        
        Returns:
            Detected task type
        """
        if target_column is None:
            self.task_type = TaskType.UNSUPERVISED
            self._log_decision(
                "task_detection",
                "Unsupervised learning",
                "No target column specified",
                "info"
            )
            return TaskType.UNSUPERVISED
        
        if target_column not in df.columns:
            self.task_type = TaskType.UNSUPERVISED
            self._log_decision(
                "task_detection",
                "Unsupervised learning (fallback)",
                f"Target column '{target_column}' not found in dataset. Available columns: {df.columns.tolist()[:10]}...",
                "warning"
            )
            print(f"⚠️ WARNING: Target column '{target_column}' not in DataFrame!")
            print(f"   Available columns ({len(df.columns)}): {df.columns.tolist()}")
            return TaskType.UNSUPERVISED
        
        target_data = df[target_column].dropna()
        unique_count = target_data.nunique()
        total_count = len(target_data)
        
        # Check if datetime - always treat as regression (predicting time/dates)
        if pd.api.types.is_datetime64_any_dtype(target_data):
            self.task_type = TaskType.REGRESSION
            self._log_decision(
                "task_detection",
                "Regression task (datetime)",
                f"Target '{target_column}' is datetime - predicting time/dates ({unique_count} unique values)",
                "info"
            )
            return TaskType.REGRESSION
        
        # Check if numerical
        if pd.api.types.is_numeric_dtype(target_data):
            # If unique values / total > 0.05, likely regression
            unique_ratio = unique_count / total_count
            print(f"🔍 TASK DETECTION: Target '{target_column}'")
            print(f"   dtype: {target_data.dtype}")
            print(f"   unique_count: {unique_count}")
            print(f"   total_count: {total_count}")
            print(f"   unique_ratio: {unique_ratio:.4f} (threshold: 0.05)")
            
            if unique_ratio > 0.05:
                self.task_type = TaskType.REGRESSION
                self._log_decision(
                    "task_detection",
                    "Regression task",
                    f"Target '{target_column}' is continuous ({unique_count} unique values, ratio: {unique_ratio:.3f})",
                    "info"
                )
                return TaskType.REGRESSION
            else:
                self.task_type = TaskType.CLASSIFICATION
                self._log_decision(
                    "task_detection",
                    "Classification task",
                    f"Target '{target_column}' is discrete ({unique_count} classes, ratio: {unique_ratio:.3f})",
                    "info"
                )
                return TaskType.CLASSIFICATION
        else:
            self.task_type = TaskType.CLASSIFICATION
            self._log_decision(
                "task_detection",
                "Classification task",
                f"Target '{target_column}' is categorical ({unique_count} classes)",
                "info"
            )
            return TaskType.CLASSIFICATION
    
    def set_model_family(self, model_type: Optional[str] = None):
        """
        Set model family for preprocessing optimization
        
        Args:
            model_type: Model type string (e.g., "random_forest")
        """
        if model_type:
            model_type_lower = model_type.lower()
            self.model_family = self.MODEL_FAMILY_MAP.get(model_type_lower, ModelFamily.UNKNOWN)
            self._log_decision(
                "model_context",
                f"Model family: {self.model_family.value}",
                f"Optimizing preprocessing for {model_type}",
                "info"
            )
        else:
            # Auto-detect model family based on task type
            if self.task_type == TaskType.CLASSIFICATION:
                self.model_family = ModelFamily.TREE_BASED  # Default to tree-based for classification
                self._log_decision(
                    "model_context",
                    f"Model family: {self.model_family.value} (auto-detected)",
                    "No model specified - defaulting to tree-based models for classification (Random Forest, XGBoost)",
                    "info"
                )
            elif self.task_type == TaskType.REGRESSION:
                self.model_family = ModelFamily.LINEAR  # Default to linear for regression
                self._log_decision(
                    "model_context",
                    f"Model family: {self.model_family.value} (auto-detected)",
                    "No model specified - defaulting to linear models for regression (Ridge, Lasso)",
                    "info"
                )
            elif self.task_type == TaskType.UNSUPERVISED:
                self.model_family = ModelFamily.DISTANCE_BASED  # Default to distance-based for clustering
                self._log_decision(
                    "model_context",
                    f"Model family: {self.model_family.value} (auto-detected)",
                    "No model specified - defaulting to distance-based models for clustering (K-Means, DBSCAN)",
                    "info"
                )
            else:
                self.model_family = ModelFamily.UNKNOWN
                self._log_decision(
                    "model_context",
                    "Model family: unknown",
                    "Using general-purpose preprocessing",
                    "info"
                )
    
    def detect_and_remove_problematic_columns(
        self, 
        df: pd.DataFrame,
        target_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Intelligently detect and remove problematic columns using EDA insights
        
        Args:
            df: Input DataFrame
            target_column: Target column to preserve
        
        Returns:
            Cleaned DataFrame
        """
        self._log_decision("column_analysis", "Starting column analysis", "Detecting problematic columns", "info")
        
        print(f"\n🔍 COLUMN ANALYSIS START - Total columns: {len(df.columns)}")
        
        columns_to_drop = []
        
        # Reset tracking lists
        self.id_columns = []
        self.constant_columns = []
        self.low_variance_columns = []
        self.high_missing_columns = []
        self.constant_column_details = {}
        
        for col in df.columns:
            # Skip target column
            if col == target_column:
                continue
            
            # 1. ID columns detection
            is_id = self._is_id_column(df, col)
            if is_id:
                columns_to_drop.append(col)
                self.id_columns.append(col)
                self._log_decision(
                    "id_detection",
                    f"Removing ID column: '{col}'",
                    self._get_id_reason(df, col),
                    "info"
                )
                continue
            
            # 2. High missing rate columns (CHECK THIS FIRST before constant/low-variance)
            # Also detect string 'nan' as missing
            missing_ratio = self._get_missing_ratio(df, col)
            
            # For object/string columns, also count string 'nan' as missing
            additional_string_nan_count = 0
            if df[col].dtype == 'object':
                additional_string_nan_count = (df[col] == 'nan').sum()
            
            total_missing_ratio = missing_ratio + (additional_string_nan_count / len(df))
            
            if total_missing_ratio > self.MISSING_DROP_THRESHOLD:
                columns_to_drop.append(col)
                self.high_missing_columns.append(col)
                
                # Store detailed info for high missing columns
                missing_count = df[col].isna().sum() + additional_string_nan_count
                total_count = len(df)
                missing_percentage = (missing_count / total_count * 100) if total_count > 0 else 0
                
                self.constant_column_details[col] = {
                    'missing_count': int(missing_count),
                    'total_count': int(total_count),
                    'percentage': float(missing_percentage),
                    'type': 'high_missing'
                }
                
                self._log_decision(
                    "missing_data",
                    f"Removing high-missing column: '{col}'",
                    f"{missing_percentage:.1f}% missing (threshold: {self.MISSING_DROP_THRESHOLD*100}%)",
                    "warning"
                )
                continue
            
            # 3. Constant columns (from EDA or direct check)
            if self._is_constant_column(df, col):
                columns_to_drop.append(col)
                self.constant_columns.append(col)
                
                # Store detailed info
                value_counts = df[col].value_counts(dropna=False)  # Include NaN/None values
                most_common_value = value_counts.index[0] if len(value_counts) > 0 else None  # Get the actual value
                value_count = int(value_counts.iloc[0]) if len(value_counts) > 0 else 0  # Get the count
                total_count = len(df)
                percentage = (value_count / total_count * 100) if total_count > 0 else 0.0
                
                # Convert value to string, handling NaN and empty strings specially
                if pd.isna(most_common_value):
                    value_str = 'nan'
                elif most_common_value == '':
                    value_str = '(empty string)'
                else:
                    value_str = str(most_common_value)
                
                # Calculate actual variance for numerical columns
                actual_variance = None
                if pd.api.types.is_numeric_dtype(df[col]):
                    try:
                        actual_variance = float(df[col].var(ddof=1))
                    except Exception:
                        actual_variance = None
                
                self.constant_column_details[col] = {
                    'value': value_str,
                    'count': int(value_count),
                    'percentage': float(percentage),
                    'variance': actual_variance,
                    'type': 'constant'
                }
                
                self._log_decision(
                    "constant_detection",
                    f"Removing constant column: '{col}'",
                    f"Value '{most_common_value}' appears {percentage:.1f}% ({value_count:,}/{len(df):,} rows) - no predictive power",
                    "info"
                )
                continue
            
            # 4. Low-variance columns (most values are the same)
            if self._is_low_variance_column(df, col):
                columns_to_drop.append(col)
                self.low_variance_columns.append(col)
                
                # Store detailed info
                value_counts = df[col].value_counts(dropna=False)  # Include NaN/None values
                most_common_value = value_counts.index[0] if len(value_counts) > 0 else None  # Get the actual value
                value_count = int(value_counts.iloc[0]) if len(value_counts) > 0 else 0  # Get the count
                total_count = len(df)
                percentage = (value_count / total_count * 100) if total_count > 0 else 0.0
                
                # Convert value to string, handling NaN and empty strings specially
                if pd.isna(most_common_value):
                    value_str = 'nan'
                elif most_common_value == '':
                    value_str = '(empty string)'
                else:
                    value_str = str(most_common_value)
                
                # Calculate actual variance for numerical columns
                actual_variance = None
                if pd.api.types.is_numeric_dtype(df[col]):
                    try:
                        actual_variance = float(df[col].var(ddof=1))
                    except Exception:
                        actual_variance = None
                
                self.constant_column_details[col] = {
                    'value': value_str,
                    'count': int(value_count),
                    'percentage': float(percentage),
                    'variance': actual_variance,
                    'type': 'low_variance'
                }
                
                self._log_decision(
                    "low_variance_detection",
                    f"Removing low-variance column: '{col}'",
                    f"Value '{most_common_value}' appears {percentage:.1f}% ({value_count:,}/{len(df):,} rows) - insufficient variation (threshold: {self.LOW_VARIANCE_THRESHOLD*100}%)",
                    "info"
                )
                continue
            
            # (Note: High missing check was moved to position 2, before constant/low-variance checks)
        
        # Drop all problematic columns
        
        # Safety check: ensure we're not removing ALL columns
        remaining_columns = [col for col in df.columns if col not in columns_to_drop]
        if target_column and target_column in remaining_columns:
            remaining_features = [col for col in remaining_columns if col != target_column]
        else:
            remaining_features = remaining_columns
        
        print(f"🔍 Safety check:")
        print(f"   Total columns: {len(df.columns)} - {df.columns.tolist()}")
        print(f"   Target column: {target_column}")
        print(f"   Marked for removal: {len(columns_to_drop)} - {columns_to_drop}")
        print(f"   Remaining after removal: {len(remaining_columns)} - {remaining_columns}")
        print(f"   Remaining features (excluding target): {len(remaining_features)} - {remaining_features}")
        
        # If we would remove all features, don't remove ANY columns
        if len(remaining_features) == 0:
            print(f"⚠️ SAFETY TRIGGERED: Would remove all features!")
            self._log_decision(
                "column_removal",
                "Safety: Keeping all columns",
                f"Would have removed all {len(columns_to_drop)} columns, but this would leave no features. Keeping all columns for preprocessing.",
                "warning"
            )
            # Don't drop anything
            columns_to_drop = []

            self.id_columns = []
            self.constant_columns = []
            self.high_missing_columns = []
        # If we would have very few features left (< 2), be less aggressive
        elif len(remaining_features) < 2:
            print(f"⚠️ SAFETY TRIGGERED: Only {len(remaining_features)} feature(s) would remain!")
            self._log_decision(
                "column_removal",
                "Safety: Keeping more columns",
                f"Only {len(remaining_features)} feature(s) would remain. Keeping all columns to preserve data.",
                "warning"
            )
            # Don't drop anything to keep at least 2 features
            columns_to_drop = []

            # Reset the tracking lists too
            self.id_columns = []
            self.constant_columns = []
            self.high_missing_columns = []
        
        df_cleaned = df.drop(columns=columns_to_drop)
        
        if columns_to_drop:
            self._log_decision(
                "column_removal",
                f"Removed {len(columns_to_drop)} problematic columns",
                f"ID: {len(self.id_columns)}, Constant: {len(self.constant_columns)}, "
                f"Low variance: {len(self.low_variance_columns)}, "
                f"High missing: {len(self.high_missing_columns)}",
                "info"
            )
        else:
            self._log_decision(
                "column_removal",
                "No problematic columns detected",
                "All columns passed quality checks",
                "info"
            )
        
        return df_cleaned
    
    def detect_and_remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and remove duplicate rows with detailed tracking
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with duplicates removed
        """
        initial_count = len(df)
        
        # Find duplicates
        duplicate_mask = df.duplicated(keep=False)  # Mark all duplicates
        duplicate_count = df.duplicated().sum()  # Count excluding first occurrence
        
        if duplicate_count == 0:
            self._log_decision(
                "duplicate_removal",
                "No duplicate rows detected",
                "All rows are unique",
                "info"
            )
            return df
        
        # Track duplicate details before removal
        duplicate_details = []
        df_with_index = df.copy()
        df_with_index['_original_index'] = range(len(df))
        
        # Find groups of duplicates
        duplicate_groups = df_with_index[duplicate_mask].groupby(
            list(df.columns), dropna=False
        )['_original_index'].apply(list).reset_index(drop=True)
        
        # Limit to top 20 duplicate groups for performance
        for i, (_, group) in enumerate(duplicate_groups.items()):
            if i >= 20:
                break
            
            indices = group
            if len(indices) > 1:
                first_idx = indices[0]
                sample_row = df.iloc[first_idx].to_dict()
                
                # Convert to JSON-serializable types
                sample_row_clean = {}
                for k, v in sample_row.items():
                    if pd.isna(v):
                        sample_row_clean[k] = None
                    elif isinstance(v, (np.integer, np.int64, np.int32)):
                        sample_row_clean[k] = int(v)
                    elif isinstance(v, (np.floating, np.float64, np.float32)):
                        sample_row_clean[k] = float(v)
                    else:
                        sample_row_clean[k] = str(v)
                
                duplicate_details.append({
                    'occurrence_count': len(indices),
                    'row_indices': [int(idx) for idx in indices[:10]],
                    'sample_data': sample_row_clean,
                    'total_indices': len(indices)
                })
        
        # Store duplicate details
        self.duplicate_removal_details = duplicate_details
        self.duplicates_removed_count = duplicate_count
        
        # Remove duplicates (keep first occurrence)
        df_cleaned = df.drop_duplicates(keep='first')
        final_count = len(df_cleaned)
        
        # Log the decision
        percentage = (duplicate_count / initial_count * 100) if initial_count > 0 else 0
        self._log_decision(
            "duplicate_removal",
            f"Removed {duplicate_count} duplicate rows ({percentage:.1f}%)",
            f"Found {len(duplicate_details)} unique duplicate patterns. "
            f"Kept first occurrence of each duplicate set. {initial_count} → {final_count} rows",
            "info"
        )
        
        return df_cleaned
    
    def remove_outlier_rows(self, df: pd.DataFrame, target_column: Optional[str] = None) -> pd.DataFrame:
        """
        Remove rows with outliers based on user preferences
        
        Args:
            df: Input DataFrame
            target_column: Target column to preserve
        
        Returns:
            DataFrame with outlier rows removed
        """
        initial_count = len(df)
        rows_to_remove = set()
        
        # Get numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Skip target column
        if target_column and target_column in numerical_cols:
            numerical_cols = [col for col in numerical_cols if col != target_column]
        
        columns_with_removals = []
        outlier_details_for_removal = []
        
        for col in numerical_cols:
            # Check if user wants to remove outliers for this column
            preference = self.outlier_preferences.get(col, 'keep')
            
            if preference == 'remove':
                # Calculate outlier bounds using IQR method
                col_data = df[col].dropna()
                if len(col_data) > 0:
                    Q1 = col_data.quantile(0.25)
                    Q3 = col_data.quantile(0.75)
                    IQR = Q3 - Q1
                    
                    if IQR < 1e-10:
                        # Use standard deviation if IQR is too small
                        mean = col_data.mean()
                        std = col_data.std()
                        lower_bound = mean - 3 * std
                        upper_bound = mean + 3 * std
                    else:
                        lower_bound = Q1 - 2.5 * IQR
                        upper_bound = Q3 + 2.5 * IQR
                    
                    # Find outlier rows
                    outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                    outlier_indices = df[outlier_mask].index.tolist()
                    outlier_count = len(outlier_indices)
                    
                    if outlier_indices:
                        rows_to_remove.update(outlier_indices)
                        columns_with_removals.append(col)
                        
                        # Track outlier details
                        outlier_details_for_removal.append({
                            'column': col,
                            'count': outlier_count,
                            'percentage': (outlier_count / len(df) * 100) if len(df) > 0 else 0,
                            'lower_bound': float(lower_bound),
                            'upper_bound': float(upper_bound),
                            'action': 'removed'
                        })
        
        if rows_to_remove:
            # Remove outlier rows
            df_cleaned = df.drop(index=list(rows_to_remove))
            final_count = len(df_cleaned)
            removed_count = initial_count - final_count
            
            # Store removal details
            self.outlier_rows_removed = removed_count
            self.outlier_details.extend(outlier_details_for_removal)
            
            self._log_decision(
                "outlier_removal",
                f"Removed {removed_count} rows with outliers ({(removed_count/initial_count*100):.1f}%)",
                f"Affected columns: {', '.join(columns_with_removals)}. "
                f"{initial_count} → {final_count} rows",
                "warning"
            )
            
            return df_cleaned.reset_index(drop=True)
        else:
            if any(pref == 'remove' for pref in self.outlier_preferences.values()):
                self._log_decision(
                    "outlier_removal",
                    "No outliers detected for removal",
                    "User requested outlier removal but no significant outliers found",
                    "info"
                )
            
            return df
    
    def _is_id_column(self, df: pd.DataFrame, col: str) -> bool:
        """Check if column is an ID column - requires multiple indicators"""
        # Skip datetime columns - they may be highly unique but are not IDs
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return False
        
        # Try to parse as datetime if it's object/string type
        if df[col].dtype == 'object':
            try:
                # Sample first non-null values to check if datetime
                sample = df[col].dropna().head(100)
                if len(sample) > 0:
                    pd.to_datetime(sample, errors='raise')
                    # If no error, it's a datetime column - not an ID
                    return False
            except:
                pass  # Not a datetime, continue with ID checks
        
        # Check 1: Name-based detection
        col_lower = col.lower()
        has_id_name = (col_lower == 'id' or col_lower.endswith('_id') or 
                       col_lower.startswith('id_') or 'identifier' in col_lower)
        
        # Check 2: High uniqueness
        uniqueness_ratio = df[col].nunique() / len(df) if len(df) > 0 else 0
        is_highly_unique = uniqueness_ratio > self.ID_UNIQUENESS_THRESHOLD
        
        # Only consider it an ID if it has BOTH id-like name AND high uniqueness
        # OR if it's a perfect sequential index
        if has_id_name and is_highly_unique:
            return True
        
        # Check 3: Sequential integers (0,1,2,3... or 1,2,3,4...)
        if pd.api.types.is_numeric_dtype(df[col]):
            col_values = df[col].dropna()
            if len(col_values) > 0:
                try:
                    int_values = col_values.astype(int)
                    sorted_vals = sorted(int_values)
                    # Check if it's 0,1,2,3... or 1,2,3,4...
                    is_sequential = (sorted_vals == list(range(len(sorted_vals))) or 
                                   sorted_vals == list(range(1, len(sorted_vals) + 1)))
                    if is_sequential and has_id_name:
                        return True
                except:
                    pass
        
        return False
    
    def _get_id_reason(self, df: pd.DataFrame, col: str) -> str:
        """Get reason why column is detected as ID"""
        col_lower = col.lower()
        if col_lower == 'id' or col_lower.endswith('_id') or col_lower.startswith('id_'):
            return "Column name indicates ID"
        
        uniqueness_ratio = df[col].nunique() / len(df) if len(df) > 0 else 0
        if uniqueness_ratio > self.ID_UNIQUENESS_THRESHOLD:
            return f"{uniqueness_ratio*100:.1f}% unique values"
        
        return "Sequential integer pattern detected"
    
    def _is_constant_column(self, df: pd.DataFrame, col: str) -> bool:
        """Check if column is constant using EDA or direct check"""
        # Use EDA results if available
        if self.eda_results:
            basic_info = self.eda_results.get('basic_info', {})
            # Could add more sophisticated checks here
        
        # Direct check
        return df[col].nunique() <= self.CONSTANT_THRESHOLD
    
    def _is_low_variance_column(self, df: pd.DataFrame, col: str) -> bool:
        """
        Check if column has low variance (most values are the same)
        Returns True if the most common value appears more than LOW_VARIANCE_THRESHOLD
        
        For numerical columns, also checks statistical variance to avoid removing
        columns like "Pool Area" where 99% are 0 but the non-zero values have high variance
        """
        if len(df[col]) == 0:
            return False
        
        # Calculate frequency of most common value
        value_counts = df[col].value_counts(normalize=True, dropna=False)
        if len(value_counts) == 0:
            return False
        
        most_common_freq = value_counts.iloc[0]
        
        # If frequency is below threshold, not low variance
        if most_common_freq <= self.LOW_VARIANCE_THRESHOLD:
            return False
        
        # For numerical columns, also check statistical variance
        if pd.api.types.is_numeric_dtype(df[col]):
            try:
                variance = float(df[col].var(ddof=1))
                # If variance is high (> 100), keep the column even if most values are the same
                # This preserves columns like "Pool Area" where rare non-zero values matter
                if variance > 100:
                    return False
            except Exception:
                pass  # If variance calculation fails, fall through to frequency-based check
        
        return True
    
    def _get_missing_ratio(self, df: pd.DataFrame, col: str) -> float:
        """Get missing value ratio for a column"""
        # Use EDA results if available
        if self.eda_results:
            missing_values = self.eda_results.get('missing_values', {})
            columns_with_missing = missing_values.get('columns_with_missing', {})
            if col in columns_with_missing:
                return columns_with_missing[col]['percentage'] / 100
        
        # Direct calculation
        return df[col].isna().sum() / len(df) if len(df) > 0 else 0
    
    def categorize_features(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None
    ):
        """
        Categorize features into numerical, categorical, and datetime
        
        NOTE: Preserves datetime-extracted features that were already added to numerical_features
        
        Args:
            df: Input DataFrame
            target_column: Target column to exclude
        """
        self._log_decision("feature_categorization", "Categorizing features", "Analyzing column types", "info")
        
        feature_cols = [col for col in df.columns if col != target_column]
        
        print(f"\n🔍 CATEGORIZE_FEATURES DEBUG:")
        print(f"   feature_cols ({len(feature_cols)}): {feature_cols}")
        print(f"   target_column: '{target_column}'")
        
        # Store existing datetime-extracted features before resetting
        existing_numerical = self.numerical_features.copy()
        existing_categorical = self.categorical_features.copy()
        
        print(f"   existing_numerical: {existing_numerical}")
        print(f"   existing_categorical: {existing_categorical}")
        
        # Reset lists to rebuild them
        self.numerical_features = []
        self.categorical_features = []
        self.datetime_features = []
        
        for col in feature_cols:
            # Skip if this column was already categorized (e.g., datetime-extracted features)
            if col in existing_numerical:
                self.numerical_features.append(col)
                print(f"   ✓ '{col}' -> numerical (from existing)")
                continue
            elif col in existing_categorical:
                self.categorical_features.append(col)
                print(f"   ✓ '{col}' -> categorical (from existing)")
                continue
            
            # Datetime detection
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                self.datetime_features.append(col)
                print(f"   ✓ '{col}' -> datetime (dtype detection)")
            # Numerical
            elif pd.api.types.is_numeric_dtype(df[col]):
                # All numerical columns stay numerical (including binary 0/1)
                self.numerical_features.append(col)
                print(f"   ✓ '{col}' -> numerical (dtype: {df[col].dtype})")
            # Categorical
            else:
                # Try datetime parsing
                is_datetime = False
                non_null = df[col].dropna()
                if len(non_null) > 0:
                    try:
                        pd.to_datetime(non_null.head(100), errors='raise')
                        is_datetime = True
                    except:
                        pass
                
                if is_datetime:
                    self.datetime_features.append(col)
                    print(f"   ✓ '{col}' -> datetime (parsing)")
                else:
                    # Check if this is a text column (long strings, not categorical)
                    is_text = False
                    if df[col].dtype == 'object' or pd.api.types.is_string_dtype(df[col]):
                        non_null_vals = df[col].dropna()
                        if len(non_null_vals) > 0:
                            # Calculate text characteristics
                            str_vals = non_null_vals.astype(str)
                            avg_length = str_vals.str.len().mean()
                            unique_count = df[col].nunique()
                            unique_ratio = unique_count / len(df[col])
                            
                            # Check for sentence structure (spaces, punctuation)
                            has_spaces = str_vals.str.contains(r'\s').mean() > 0.5
                            has_punctuation = str_vals.str.contains(r'[.,!?;:]').mean() > 0.3
                            
                            # Text column criteria:
                            # 1. Average length > 30 chars
                            # 2. Unique values > 50
                            # 3. Unique ratio > 0.5
                            # 4. Has whitespace or punctuation (sentence structure)
                            if (avg_length > 30 and unique_count > 50 and unique_ratio > 0.5) or \
                               (avg_length > 50 and (has_spaces or has_punctuation)):
                                is_text = True
                                self.text_features.append(col)
                                print(f"   ✓ '{col}' -> TEXT (avg_len={avg_length:.0f}, unique={unique_count}, ratio={unique_ratio:.2f})")
                                print(f"      Will apply TF-IDF vectorization with {5000} features")
                    
                    if not is_text:
                        self.categorical_features.append(col)
                        print(f"   ✓ '{col}' -> categorical (dtype: {df[col].dtype})")
        
        print(f"\n   FINAL RESULT:")
        print(f"   numerical_features: {self.numerical_features}")
        print(f"   categorical_features: {self.categorical_features}")
        print(f"   text_features: {self.text_features}")
        print(f"   datetime_features: {self.datetime_features}\n")
        
        # Build feature summary message
        feature_summary_parts = []
        if self.numerical_features:
            feature_summary_parts.append(f"{len(self.numerical_features)} numerical")
        if self.categorical_features:
            feature_summary_parts.append(f"{len(self.categorical_features)} categorical")
        if self.text_features:
            feature_summary_parts.append(f"{len(self.text_features)} text")
        if self.datetime_features:
            feature_summary_parts.append(f"{len(self.datetime_features)} datetime")
        
        feature_summary = ", ".join(feature_summary_parts) if feature_summary_parts else "0 features"
        
        self._log_decision(
            "feature_categorization",
            f"Categorized {len(feature_cols)} features",
            feature_summary,
            "info"
        )
    
    def detect_and_parse_datetime(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect and parse datetime columns with comprehensive format support
        
        Supports:
        - ISO formats: 2024-01-15, 2024/01/15
        - US formats: 01-15-2024, 01/15/2024, 1/15/24
        - UK formats: 15-01-2024, 15/01/2024, 15.01.2024
        - With time: 2024-01-15 14:30:00, 01/15/2024 2:30 PM
        - Unix timestamps
        
        Args:
            df: DataFrame
        
        Returns:
            DataFrame with datetime columns parsed
        """
        # First, detect datetime columns if not already categorized
        if not self.datetime_features:
            for col in df.columns:
                # Check if already datetime type
                if pd.api.types.is_datetime64_any_dtype(df[col]):
                    if col not in self.datetime_features:
                        self.datetime_features.append(col)
                # Try to detect datetime in object/string columns
                elif df[col].dtype == 'object':
                    try:
                        sample = df[col].dropna().head(100)
                        if len(sample) > 0:
                            # Try parsing - if successful, it's datetime
                            parsed = pd.to_datetime(sample, errors='raise', infer_datetime_format=True)
                            # If parsing worked, add to datetime features
                            if col not in self.datetime_features:
                                self.datetime_features.append(col)
                    except:
                        pass  # Not a datetime column
        
        # Parse datetime columns
        for col in self.datetime_features:
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                try:
                    # Try automatic parsing with dayfirst=False (US format default)
                    df[col] = pd.to_datetime(df[col], errors='coerce', infer_datetime_format=True)
                    
                    # Check if conversion was successful
                    if df[col].isna().sum() > len(df) * 0.5:
                        # Try UK format (dayfirst=True)
                        df[col] = pd.to_datetime(df[col], errors='coerce', dayfirst=True)
                    
                    self._log_decision(
                        "datetime_parsing",
                        f"Parsed datetime column: '{col}'",
                        f"Format detected automatically. {df[col].isna().sum()} values could not be parsed",
                        "info"
                    )
                except Exception as e:
                    self._log_decision(
                        "datetime_parsing",
                        f"Failed to parse datetime column: '{col}'",
                        f"Error: {str(e)}. Column will be treated as categorical",
                        "warning"
                    )
                    # Move to categorical if parsing fails
                    self.datetime_features.remove(col)
                    self.categorical_features.append(col)
        
        return df
    
    def extract_datetime_features(
        self, 
        df: pd.DataFrame, 
        target_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Extract datetime features based on model family strategy
        
        Strategies:
        - Tree-based: Keep all numeric and categorical parts (year, month, day, weekday, hour)
        - Linear: Normalized numeric parts + one-hot weekday, drop raw timestamps
        - Unsupervised: Cyclic encoding for periodic features, durations from reference
        
        Args:
            df: DataFrame
            target_column: Target column (for reference date calculation)
        
        Returns:
            DataFrame with datetime features extracted
        """
        if not self.datetime_features:
            return df
        
        datetime_metadata = {
            "original_columns": [],
            "features_created": [],
            "columns_dropped": []
        }
        
        for col in self.datetime_features.copy():
            if col not in df.columns:
                continue
            
            print(f"   🔍 Checking datetime column '{col}' - target_column='{target_column}' - match={col == target_column}")
            
            # Skip target column - keep it as-is for prediction
            if col == target_column:
                self._log_decision(
                    "datetime_features",
                    f"Skipping datetime feature extraction for target column '{col}'",
                    "Target column preserved for prediction",
                    "info"
                )
                continue
            
            original_col = col
            datetime_metadata["original_columns"].append(col)
            
            # Ensure column is datetime
            if not pd.api.types.is_datetime64_any_dtype(df[col]):
                df[col] = pd.to_datetime(df[col], errors='coerce')
            
            # Detect if this is a date-only, time-only, or full datetime column
            has_date_variation = df[col].dt.date.nunique() > 1
            has_time_variation = (df[col].dt.hour.nunique() > 1) or (df[col].dt.minute.nunique() > 1)
            is_time_only = not has_date_variation and has_time_variation
            is_date_only = has_date_variation and not has_time_variation
            
            # Extract components based on model family
            if self.model_family == ModelFamily.TREE_BASED:
                # Tree-based: Extract all useful components directly
                if not is_time_only:
                    # Extract date components
                    df[f'{col}_year'] = df[col].dt.year
                    df[f'{col}_month'] = df[col].dt.month
                    df[f'{col}_day'] = df[col].dt.day
                    df[f'{col}_dayofweek'] = df[col].dt.dayofweek  # 0=Monday, 6=Sunday
                    df[f'{col}_quarter'] = df[col].dt.quarter
                    df[f'{col}_is_weekend'] = df[col].dt.dayofweek.isin([5, 6]).astype(int)
                    
                    datetime_metadata["features_created"].extend([
                        f'{col}_year', f'{col}_month', f'{col}_day',
                        f'{col}_dayofweek', f'{col}_quarter', f'{col}_is_weekend'
                    ])
                
                # Extract time components if they exist
                if not is_date_only and (df[col].dt.hour.any() or is_time_only):
                    df[f'{col}_hour'] = df[col].dt.hour
                    df[f'{col}_is_business_hours'] = df[col].dt.hour.between(9, 17).astype(int)
                    datetime_metadata["features_created"].extend([
                        f'{col}_hour', f'{col}_is_business_hours'
                    ])
                
                # Add to numerical features
                for new_col in datetime_metadata["features_created"]:
                    if new_col.startswith(f'{col}_') and new_col in df.columns:
                        if new_col not in self.numerical_features:
                            self.numerical_features.append(new_col)
                
                feature_type = "time-only" if is_time_only else ("date-only" if is_date_only else "datetime")
                self._log_decision(
                    "datetime_features",
                    f"Extracted datetime features from '{col}' ({feature_type})",
                    f"Tree-based model - created {len([f for f in datetime_metadata['features_created'] if f.startswith(col)])} numeric features",
                    "info"
                )
            
            elif self.model_family == ModelFamily.LINEAR:
                # Linear: Normalized components + categorical weekday
                if not is_time_only:
                    # Extract date components
                    min_year = df[col].dt.year.min()
                    df[f'{col}_years_since_{min_year}'] = df[col].dt.year - min_year
                    df[f'{col}_month'] = df[col].dt.month  # Already 1-12
                    df[f'{col}_day'] = df[col].dt.day  # Already 1-31
                    
                    # Weekday as categorical (for one-hot encoding)
                    df[f'{col}_weekday'] = df[col].dt.day_name()
                    
                    datetime_metadata["features_created"].extend([
                        f'{col}_years_since_{min_year}', f'{col}_month', 
                        f'{col}_day', f'{col}_weekday'
                    ])
                    
                    # Add numerical features
                    for feat in [f'{col}_years_since_{min_year}', f'{col}_month', f'{col}_day']:
                        if feat not in self.numerical_features:
                            self.numerical_features.append(feat)
                    # Add categorical feature
                    if f'{col}_weekday' not in self.categorical_features:
                        self.categorical_features.append(f'{col}_weekday')
                
                # Add time components if they exist
                if not is_date_only and (df[col].dt.hour.any() or is_time_only):
                    df[f'{col}_hour'] = df[col].dt.hour
                    datetime_metadata["features_created"].append(f'{col}_hour')
                    if f'{col}_hour' not in self.numerical_features:
                        self.numerical_features.append(f'{col}_hour')
                
                feature_type = "time-only" if is_time_only else ("date-only" if is_date_only else "datetime")
                self._log_decision(
                    "datetime_features",
                    f"Extracted datetime features from '{col}' ({feature_type})",
                    f"Linear model - created normalized features + weekday categorical",
                    "info"
                )
            
            else:
                # Unsupervised/Distance: Cyclic encoding + durations
                # Only extract features that actually vary in the data
                if not is_time_only:
                    # Cyclic encoding for date periodic features
                    df[f'{col}_month_sin'] = np.sin(2 * np.pi * df[col].dt.month / 12)
                    df[f'{col}_month_cos'] = np.cos(2 * np.pi * df[col].dt.month / 12)
                    df[f'{col}_day_sin'] = np.sin(2 * np.pi * df[col].dt.day / 31)
                    df[f'{col}_day_cos'] = np.cos(2 * np.pi * df[col].dt.day / 31)
                    df[f'{col}_weekday_sin'] = np.sin(2 * np.pi * df[col].dt.dayofweek / 7)
                    df[f'{col}_weekday_cos'] = np.cos(2 * np.pi * df[col].dt.dayofweek / 7)
                    
                    # Duration from reference (first date)
                    reference_date = df[col].min()
                    df[f'{col}_days_since_start'] = (df[col] - reference_date).dt.days
                    
                    datetime_metadata["features_created"].extend([
                        f'{col}_month_sin', f'{col}_month_cos',
                        f'{col}_day_sin', f'{col}_day_cos',
                        f'{col}_weekday_sin', f'{col}_weekday_cos',
                        f'{col}_days_since_start'
                    ])
                    
                    # Add to numerical features
                    for new_col in [f'{col}_month_sin', f'{col}_month_cos', f'{col}_day_sin', 
                                   f'{col}_day_cos', f'{col}_weekday_sin', f'{col}_weekday_cos',
                                   f'{col}_days_since_start']:
                        if new_col in df.columns:
                            self.numerical_features.append(new_col)
                
                # Add time cyclic encoding if time component exists
                if not is_date_only and (df[col].dt.hour.any() or is_time_only):
                    df[f'{col}_hour_sin'] = np.sin(2 * np.pi * df[col].dt.hour / 24)
                    df[f'{col}_hour_cos'] = np.cos(2 * np.pi * df[col].dt.hour / 24)
                    datetime_metadata["features_created"].extend([
                        f'{col}_hour_sin', f'{col}_hour_cos'
                    ])
                    for feat in [f'{col}_hour_sin', f'{col}_hour_cos']:
                        if feat not in self.numerical_features:
                            self.numerical_features.append(feat)
                
                feature_type = "time-only" if is_time_only else ("date-only" if is_date_only else "datetime")
                self._log_decision(
                    "datetime_features",
                    f"Extracted datetime features from '{col}' ({feature_type})",
                    f"Distance/Unsupervised model - created cyclic encodings + duration features",
                    "info"
                )
            
            # Drop original datetime column
            df = df.drop(columns=[col])
            datetime_metadata["columns_dropped"].append(col)

        
        # Remove low-variance datetime features (e.g., month_cos=0 when all dates in same month)
        datetime_features_to_check = [col for col in datetime_metadata["features_created"] if col in df.columns]
        low_variance_datetime_features = []
        
        for feat_col in datetime_features_to_check:
            if feat_col in df.columns and self._is_low_variance_column(df, feat_col):
                low_variance_datetime_features.append(feat_col)
                df = df.drop(columns=[feat_col])
                datetime_metadata["columns_dropped"].append(feat_col)
                
                # Remove from feature lists




                
                self._log_decision(
                    "datetime_features",
                    f"Removed low-variance datetime feature '{feat_col}'",
                    f"Feature has >95% same values (constant/near-constant)",
                    "info"
                )
        
        if low_variance_datetime_features:
            datetime_metadata["low_variance_removed"] = low_variance_datetime_features
        
        # Store metadata
        self.preprocessing_metadata["datetime_features"] = datetime_metadata
        
        return df
    
    def choose_imputation_strategy(
        self,
        df: pd.DataFrame,
        feature_type: str,
        column: str
    ) -> dict:
        """
        Choose optimal imputation strategy based on EDA insights, missing ratio, and model type
        
        Strategy Selection Logic:
        - Missing > 80%: 'drop' (remove column entirely)
        - Missing 50-80%: 'constant' with indicator (too much missing for reliable imputation)
        - Missing 20-50%: 
            * Numerical: 'median' (robust) or 'knn' if <30% and dataset small
            * Categorical: 'most_frequent' with indicator
        - Missing 5-20%:
            * Numerical: 'mean' or 'median' (based on outliers)
            * Categorical: 'most_frequent'
        - Missing 1-5%:
            * Numerical: 'knn' if dataset small (<1000 rows), else mean/median
            * Categorical: 'most_frequent'
        - Missing <1%: 'mean'/'median' or 'most_frequent'
        
        Args:
            df: DataFrame
            feature_type: 'numerical' or 'categorical'
            column: Column name
        
        Returns:
            Dict with 'strategy', 'add_indicator', 'drop' keys
        """
        missing_ratio = self._get_missing_ratio(df, column)
        n_rows = len(df)
        
        result = {
            'strategy': 'none',
            'add_indicator': False,
            'drop': False
        }
        
        if missing_ratio == 0:
            return result
        
        # CRITICAL: Drop columns with >80% missing
        if missing_ratio > 0.80:
            result['drop'] = True
            result['strategy'] = 'drop'
            self._log_decision(
                "imputation_drop",
                f"Column '{column}' will be dropped",
                f"{missing_ratio*100:.1f}% missing values (>80% threshold)",
                "warning"
            )
            return result
        
        if feature_type == "numerical":
            has_outliers = self._has_significant_outliers(column)
            
            # 50-80% missing: constant fill + indicator
            if missing_ratio > 0.50:
                result['strategy'] = 'constant'
                result['add_indicator'] = True
                self._log_decision(
                    "imputation",
                    f"Constant imputation for '{column}'",
                    f"{missing_ratio*100:.1f}% missing (50-80%) - using -999 sentinel + indicator",
                    "info"
                )
            
            # 20-50% missing: median (robust) or KNN for small datasets
            elif missing_ratio > 0.20:
                if n_rows < 500 and missing_ratio < 0.30:
                    result['strategy'] = 'knn'
                    result['add_indicator'] = True
                    self._log_decision(
                        "imputation",
                        f"KNN imputation for '{column}'",
                        f"{missing_ratio*100:.1f}% missing, dataset small ({n_rows} rows)",
                        "info"
                    )
                else:
                    result['strategy'] = 'median'
                    result['add_indicator'] = True
                    self._log_decision(
                        "imputation",
                        f"Median imputation for '{column}'",
                        f"{missing_ratio*100:.1f}% missing (20-50%) - robust strategy + indicator",
                        "info"
                    )
            
            # 5-20% missing: mean/median based on outliers
            elif missing_ratio > 0.05:
                if has_outliers:
                    result['strategy'] = 'median'
                else:
                    result['strategy'] = 'mean'
                self._log_decision(
                    "imputation",
                    f"{'Median' if has_outliers else 'Mean'} imputation for '{column}'",
                    f"{missing_ratio*100:.1f}% missing - {'outliers detected' if has_outliers else 'normal distribution'}",
                    "info"
                )
            
            # 1-5% missing: KNN for small datasets, else mean/median
            elif missing_ratio > 0.01:
                if n_rows < 1000:
                    result['strategy'] = 'knn'
                    self._log_decision(
                        "imputation",
                        f"KNN imputation for '{column}'",
                        f"{missing_ratio*100:.1f}% missing, small dataset ({n_rows} rows)",
                        "info"
                    )
                else:
                    result['strategy'] = 'median' if has_outliers else 'mean'
                    self._log_decision(
                        "imputation",
                        f"{'Median' if has_outliers else 'Mean'} imputation for '{column}'",
                        f"{missing_ratio*100:.1f}% missing",
                        "info"
                    )
            
            # <1% missing: simple mean/median
            else:
                result['strategy'] = 'median' if has_outliers else 'mean'
        
        else:  # categorical
            # 50-80% missing: constant fill + indicator
            if missing_ratio > 0.50:
                result['strategy'] = 'constant'
                result['add_indicator'] = True
                self._log_decision(
                    "imputation",
                    f"Constant imputation for '{column}'",
                    f"{missing_ratio*100:.1f}% missing (50-80%) - using 'MISSING' category + indicator",
                    "info"
                )
            
            # 20-50% missing: most_frequent + indicator
            elif missing_ratio > 0.20:
                result['strategy'] = 'most_frequent'
                result['add_indicator'] = True
                self._log_decision(
                    "imputation",
                    f"Mode imputation for '{column}' + indicator",
                    f"{missing_ratio*100:.1f}% missing (20-50%)",
                    "info"
                )
            
            # <20% missing: most_frequent (mode)
            else:
                result['strategy'] = 'most_frequent'
        
        return result
    
    def _has_significant_outliers(self, column: str) -> bool:
        """Check if column has significant outliers using EDA results"""
        if not self.eda_results:
            return False
        
        outliers = self.eda_results.get('outliers', {})
        outliers_by_col = outliers.get('outliers_by_column', {})
        
        if column in outliers_by_col:
            outlier_pct = outliers_by_col[column].get('percentage', 0)
            return outlier_pct > (self.OUTLIER_SIGNIFICANT_THRESHOLD * 100)
        
        return False
    
    def detect_skewness(self, df: pd.DataFrame, column: str) -> Dict[str, Any]:
        """
        Detect skewness in a numerical column
        
        Args:
            df: DataFrame
            column: Column name
        
        Returns:
            Dictionary with skewness information
        """
        col_data = df[column].dropna()
        
        if len(col_data) == 0:
            return {"skewness": None, "needs_transformation": False}
        
        skewness = col_data.skew()
        
        # Check from EDA if available
        if self.eda_results:
            num_analysis = self.eda_results.get('numerical_analysis', {})
            num_features = num_analysis.get('numerical_features', {})
            if column in num_features:
                eda_skew = num_features[column].get('skewness')
                if eda_skew is not None:
                    skewness = eda_skew
        
        # Consider transformation if |skewness| > 0.75
        needs_transformation = abs(skewness) > 0.75
        
        # Determine best transformation method
        has_zero = (col_data == 0).any()
        has_negative = (col_data < 0).any()
        
        if needs_transformation:
            if has_negative or has_zero:
                method = "yeo-johnson"  # Handles any data
            elif col_data.min() > 0:
                method = "box-cox"  # For strictly positive data
            else:
                method = "yeo-johnson"  # Fallback
        else:
            method = None
        
        return {
            "skewness": float(skewness),
            "needs_transformation": needs_transformation,
            "method": method,
            "has_zero": bool(has_zero),
            "has_negative": bool(has_negative),
            "min_value": float(col_data.min()),
            "max_value": float(col_data.max())
        }
    
    def detect_high_correlation(
        self, 
        df: pd.DataFrame, 
        threshold: float = 0.9,
        target_column: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect highly correlated feature pairs that may cause multicollinearity
        
        Uses SMART clustering approach:
        1. Group features that are highly correlated (>threshold)
        2. Keep ONE representative per group based on:
           - Highest correlation with target (if supervised)
           - Highest variance (if unsupervised)
        
        Args:
            df: DataFrame with numerical features
            threshold: Correlation threshold (default: 0.9 for very high correlation)
            target_column: Target column to use for selecting best representative
        
        Returns:
            List of correlated pairs with metadata
        """
        # Get numerical columns only
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target from feature list if it exists
        if target_column and target_column in numerical_cols:
            feature_cols = [col for col in numerical_cols if col != target_column]
            has_target = True
        else:
            feature_cols = numerical_cols
            has_target = False
        
        if len(feature_cols) < 2:
            return []
        
        # Calculate correlation matrix
        corr_matrix = df[feature_cols].corr().abs()
        
        # Find correlated groups using clustering approach
        correlated_groups = self._find_correlated_groups(corr_matrix, threshold)
        
        if not correlated_groups:
            return []
        
        # For each group, select the best representative
        correlated_pairs = []
        features_to_drop = set()
        
        for group in correlated_groups:
            if len(group) < 2:
                continue
            
            # Select best feature to KEEP from this group
            if has_target and target_column in df.columns:
                # SMART: Keep feature most correlated with target
                best_feature = self._select_by_target_correlation(
                    df, group, target_column
                )
                selection_reason = f"highest correlation with target '{target_column}'"
            else:
                # Fallback: Keep feature with highest variance (most information)
                best_feature = self._select_by_variance(df, group)
                selection_reason = "highest variance (most information)"
            
            # Mark others for removal
            to_drop = [f for f in group if f != best_feature]
            features_to_drop.update(to_drop)
            
            # Create pair metadata for logging
            avg_corr = corr_matrix.loc[group, group].values.mean()
            for drop_feature in to_drop:
                pair_corr = corr_matrix.loc[best_feature, drop_feature]
                
                correlated_pairs.append({
                    "feature1": best_feature,
                    "feature2": drop_feature,
                    "correlation": float(pair_corr),
                    "keep_feature": best_feature,
                    "drop_feature": drop_feature,
                    "group_size": len(group),
                    "avg_group_correlation": float(avg_corr),
                    "selection_reason": selection_reason,
                    "semantic_relationship": self._detect_semantic_relationship(best_feature, drop_feature),
                    "can_combine": False  # Don't combine when using group clustering
                })
        
        self._log_decision(
            "multicollinearity",
            f"Found {len(correlated_groups)} correlated groups",
            f"Total pairs: {len(correlated_pairs)}, Features to drop: {len(features_to_drop)}. "
            f"Selection method: {'target correlation' if has_target else 'variance'}",
            "info"
        )
        
        return correlated_pairs
    
    def _find_correlated_groups(
        self, 
        corr_matrix: pd.DataFrame, 
        threshold: float
    ) -> List[List[str]]:
        """
        Find groups of mutually correlated features using graph clustering
        
        Args:
            corr_matrix: Correlation matrix
            threshold: Correlation threshold
        
        Returns:
            List of feature groups (each group is a list of column names)
        """
        # Build adjacency list (which features are correlated with which)
        n_features = len(corr_matrix)
        adjacency = {col: set() for col in corr_matrix.columns}
        
        for i in range(n_features):
            for j in range(i + 1, n_features):
                if corr_matrix.iloc[i, j] >= threshold:
                    col_i = corr_matrix.columns[i]
                    col_j = corr_matrix.columns[j]
                    adjacency[col_i].add(col_j)
                    adjacency[col_j].add(col_i)
        
        # Find connected components (groups of correlated features)
        visited = set()
        groups = []
        
        for feature in corr_matrix.columns:
            if feature not in visited:
                # BFS to find all features in this group
                group = self._bfs_find_group(feature, adjacency, visited)
                if len(group) > 1:  # Only keep groups with 2+ features
                    groups.append(sorted(group))
        
        return groups
    
    def _bfs_find_group(
        self, 
        start: str, 
        adjacency: dict, 
        visited: set
    ) -> List[str]:
        """
        Find all features connected to start feature using BFS
        
        Args:
            start: Starting feature
            adjacency: Adjacency list
            visited: Set of visited features
        
        Returns:
            List of features in the same group
        """
        queue = [start]
        group = []
        
        while queue:
            feature = queue.pop(0)
            if feature in visited:
                continue
            
            visited.add(feature)
            group.append(feature)
            
            # Add neighbors to queue
            for neighbor in adjacency[feature]:
                if neighbor not in visited:
                    queue.append(neighbor)
        
        return group
    
    def _select_by_target_correlation(
        self, 
        df: pd.DataFrame, 
        group: List[str], 
        target_column: str
    ) -> str:
        """
        Select feature from group with highest correlation to target
        
        Args:
            df: DataFrame
            group: List of feature names
            target_column: Target column
        
        Returns:
            Best feature name
        """
        target_correlations = {}
        
        for feature in group:
            try:
                corr = df[feature].corr(df[target_column])
                target_correlations[feature] = abs(corr)  # Use absolute correlation
            except:
                target_correlations[feature] = 0.0
        
        best_feature = max(target_correlations, key=target_correlations.get)
        
        self._log_decision(
            "multicollinearity",
            f"Selected '{best_feature}' from group of {len(group)}",
            f"Correlation with target: {target_correlations[best_feature]:.3f}. "
            f"Group: {group}",
            "info"
        )
        
        return best_feature
    
    def _select_by_variance(
        self, 
        df: pd.DataFrame, 
        group: List[str]
    ) -> str:
        """
        Select feature from group with highest coefficient of variation
        
        Args:
            df: DataFrame
            group: List of feature names
        
        Returns:
            Best feature name
        """
        cv_scores = {}
        
        for feature in group:
            mean_val = df[feature].mean()
            std_val = df[feature].std()
            # Coefficient of variation (scale-independent)
            cv = abs(std_val / mean_val) if abs(mean_val) > 1e-10 else std_val
            cv_scores[feature] = cv
        
        best_feature = max(cv_scores, key=cv_scores.get)
        
        self._log_decision(
            "multicollinearity",
            f"Selected '{best_feature}' from group of {len(group)}",
            f"Coefficient of variation: {cv_scores[best_feature]:.3f}. "
            f"Group: {group}",
            "info"
        )
        
        return best_feature
    
    def _extract_feature_base_name(self, col_name: str) -> str:
        """
        Extract the base measurement name from a feature
        E.g., 'perimeter_mean' -> 'perimeter', 'area_worst' -> 'area'
        
        Args:
            col_name: Column name
        
        Returns:
            Base name without suffix
        """
        # Common suffixes in features
        suffixes = ['_mean', '_se', '_worst', '_std', '_min', '_max', '_median']
        col_lower = col_name.lower()
        
        for suffix in suffixes:
            if col_lower.endswith(suffix):
                return col_lower.replace(suffix, '')
        
        return col_lower
    
    def _detect_semantic_relationship(self, col1: str, col2: str) -> Optional[str]:
        """
        Detect if two columns have a semantic relationship (e.g., mean/worst of same measure)
        
        Args:
            col1: First column name
            col2: Second column name
        
        Returns:
            Relationship type or None
        """
        base1 = self._extract_feature_base_name(col1)
        base2 = self._extract_feature_base_name(col2)
        
        if base1 == base2:
            return f"same_measure_{base1}"
        
        return None
    
    def _decide_feature_to_keep(
        self, 
        col1: str, 
        col2: str, 
        cv1: float, 
        cv2: float,
        base1: str,
        base2: str
    ) -> tuple:
        """
        Intelligently decide which feature to keep when two are highly correlated
        
        Strategy:
        1. Preserve feature diversity (keep different measurement types)
        2. Use coefficient of variation as tiebreaker
        3. Prefer 'mean' over 'se' over 'worst' (generally more stable)
        
        Args:
            col1, col2: Column names
            cv1, cv2: Coefficients of variation
            base1, base2: Base feature names
        
        Returns:
            (keep_feature, drop_feature) tuple
        """
        # Count how many times each base feature appears
        # This helps preserve diversity
        # (We'll track this globally in the actual implementation)
        
        # Priority ranking for suffixes (most important to least)
        suffix_priority = {
            'mean': 3,
            'median': 3,
            'worst': 2,
            'max': 2,
            'se': 1,
            'std': 1,
            'error': 1
        }
        
        # Get priorities
        priority1 = 0
        priority2 = 0
        for suffix, pri in suffix_priority.items():
            if suffix in col1.lower():
                priority1 = pri
            if suffix in col2.lower():
                priority2 = pri
        
        # Decision logic:
        # 1. If different base names AND similar priority -> keep both by choosing higher CV
        # 2. If same base name -> keep higher priority suffix, then higher CV
        # 3. If different priorities -> keep higher priority
        
        if base1 != base2:
            # Different measurements - prefer to keep both, so use CV
            if cv1 >= cv2:
                return (col1, col2)
            else:
                return (col2, col1)
        else:
            # Same base measurement - use priority then CV
            if priority1 > priority2:
                return (col1, col2)
            elif priority2 > priority1:
                return (col2, col1)
            else:
                # Same priority - use CV
                if cv1 >= cv2:
                    return (col1, col2)
                else:
                    return (col2, col1)
    
    def _detect_semantic_relationship(self, col1: str, col2: str) -> Optional[str]:
        """
        Detect if two column names have a semantic relationship
        that suggests they could be combined intelligently
        
        Args:
            col1: First column name
            col2: Second column name
        
        Returns:
            Relationship type or None
        """
        col1_lower = col1.lower()
        col2_lower = col2.lower()
        
        # Time relationships (year built, year remodeled, etc.)
        if ('year' in col1_lower and 'year' in col2_lower):
            return "temporal_difference"  # Can create "years_since" feature
        
        # Area relationships (garage_area, garage_cars)
        if (('area' in col1_lower and 'cars' in col2_lower) or 
            ('area' in col2_lower and 'cars' in col1_lower)):
            return "area_per_unit"  # Can create area/count ratio
        
        # Size relationships (lot_area, lot_frontage)
        if ('area' in col1_lower and 'frontage' in col2_lower) or \
           ('area' in col2_lower and 'frontage' in col1_lower):
            return "area_ratio"
        
        # Count relationships (rooms_above, bedrooms)
        if ('room' in col1_lower and 'room' in col2_lower) or \
           ('bed' in col1_lower and 'room' in col2_lower) or \
           ('room' in col1_lower and 'bed' in col2_lower):
            return "count_difference"
        
        # Similar prefixes (suggests related measurements)
        prefix1 = col1_lower.split('_')[0] if '_' in col1_lower else col1_lower[:4]
        prefix2 = col2_lower.split('_')[0] if '_' in col2_lower else col2_lower[:4]
        
        if prefix1 == prefix2 and len(prefix1) > 2:
            return "same_category"
        
        return None
    
    def handle_multicollinearity(
        self, 
        df: pd.DataFrame, 
        correlated_pairs: List[Dict[str, Any]], 
        target_column: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Handle multicollinearity based on model family and correlation patterns
        
        CONSERVATIVE STRATEGY: Only remove truly redundant features
        - Linear/Logistic: Remove only exact duplicates or combine semantically related
        - Tree-based: Keep all features (trees handle correlation naturally)
        - Distance-based: Minimal removal (only extreme cases r > 0.95)
        
        Args:
            df: DataFrame
            correlated_pairs: List of correlated feature pairs
            target_column: Target column to preserve
        
        Returns:
            DataFrame with handled multicollinearity
        """
        if not correlated_pairs:
            return df
        
        columns_to_drop = set()
        features_combined = []
        features_to_keep = set()  # Track what we explicitly want to keep
        
        # CRITICAL: Track base feature types to ensure we keep at least one of each
        base_features = {}  # {base_name: [col1, col2, ...]}
        for col in df.select_dtypes(include=[np.number]).columns:
            base = self._extract_feature_base_name(col)
            if base not in base_features:
                base_features[base] = []
            base_features[base].append(col)
        
        for pair in correlated_pairs:
            col1 = pair["feature1"]
            col2 = pair["feature2"]
            correlation = pair["correlation"]
            relationship = pair["semantic_relationship"]
            
            # CRITICAL: Never drop or combine target column
            if col1 == target_column or col2 == target_column:
                self._log_decision(
                    "multicollinearity",
                    f"Skipping multicollinearity handling for target column",
                    f"Detected correlation between '{col1}' and '{col2}' (r={correlation:.3f}), " +
                    f"but one is the target column. Keeping both to preserve target.",
                    "warning"
                )
                continue
            
            # CRITICAL: Check if we would eliminate all features of a base type
            base1 = self._extract_feature_base_name(col1)
            base2 = self._extract_feature_base_name(col2)
            
            # Count how many features of each base type would remain
            remaining_base1 = [f for f in base_features.get(base1, []) 
                              if f not in columns_to_drop and f != pair["drop_feature"]]
            remaining_base2 = [f for f in base_features.get(base2, []) 
                              if f not in columns_to_drop and f != pair["drop_feature"]]
            
            # SAFETY: Don't drop if it would eliminate the last feature of a base type
            drop_candidate = pair["drop_feature"]
            drop_base = self._extract_feature_base_name(drop_candidate)
            remaining_of_drop_type = [f for f in base_features.get(drop_base, []) 
                                     if f not in columns_to_drop and f != drop_candidate]
            
            if len(remaining_of_drop_type) == 0:
                self._log_decision(
                    "multicollinearity",
                    f"Preserving '{drop_candidate}' despite high correlation",
                    f"Would eliminate all '{drop_base}' features. Keeping to preserve feature diversity. " +
                    f"Correlated with '{pair['keep_feature']}' (r={correlation:.3f})",
                    "info"
                )
                continue
            
            # Strategy based on model family
            if self.model_family == ModelFamily.LINEAR:
                # Linear models: CONSERVATIVE - only remove truly redundant features
                # Only combine if there's a clear semantic relationship
                if relationship == "temporal_difference":
                    # Create difference feature (e.g., years_since_remodel)
                    new_col = f"{col1}_{col2}_diff"
                    df[new_col] = df[col1] - df[col2]
                    columns_to_drop.add(col1)
                    columns_to_drop.add(col2)
                    features_combined.append({
                        "new_feature": new_col,
                        "original_features": [col1, col2],
                        "method": "difference",
                        "reason": f"Temporal difference (r={correlation:.3f})"
                    })
                    self._log_decision(
                        "multicollinearity",
                        f"Combined '{col1}' and '{col2}' into '{new_col}'",
                        f"Linear model - temporal relationship detected (r={correlation:.3f})",
                        "warning"
                    )
                    
                elif relationship == "area_per_unit":
                    # Create ratio feature (e.g., area_per_car)
                    area_col = col1 if 'area' in col1.lower() else col2
                    count_col = col2 if 'area' in col1.lower() else col1
                    new_col = f"{area_col}_per_{count_col}"
                    
                    # Avoid division by zero
                    df[new_col] = df[area_col] / (df[count_col].replace(0, 1))
                    columns_to_drop.add(col1)
                    columns_to_drop.add(col2)
                    features_combined.append({
                        "new_feature": new_col,
                        "original_features": [col1, col2],
                        "method": "ratio",
                        "reason": f"Area per unit (r={correlation:.3f})"
                    })
                    self._log_decision(
                        "multicollinearity",
                        f"Combined '{col1}' and '{col2}' into '{new_col}'",
                        f"Linear model - area/count relationship detected (r={correlation:.3f})",
                        "warning"
                    )
                    
                elif base1 == base2 and correlation > 0.98:
                    # ONLY drop if SAME base feature AND extremely high correlation (>0.98)
                    # E.g., radius_mean vs radius_se - keep the more stable one
                    drop_col = pair["drop_feature"]
                    columns_to_drop.add(drop_col)
                    
                    self._log_decision(
                        "multicollinearity",
                        f"Dropped '{drop_col}' (same feature type as '{pair['keep_feature']}')",
                        f"Linear model - extremely high correlation (r={correlation:.3f}) between same measurement type",
                        "warning"
                    )
                else:
                    # Different base features OR correlation < 0.98 -> KEEP BOTH
                    self._log_decision(
                        "multicollinearity",
                        f"Keeping both '{col1}' and '{col2}' despite correlation",
                        f"Linear model - preserving feature diversity (r={correlation:.3f}, bases: {base1} vs {base2})",
                        "info"
                    )
                
            elif self.model_family == ModelFamily.DISTANCE_BASED:
                # Distance-based: CONSERVATIVE - only drop if SAME base AND very high correlation
                if base1 == base2 and correlation > 0.95:
                    drop_col = pair["drop_feature"]
                    columns_to_drop.add(drop_col)
                    
                    self._log_decision(
                        "multicollinearity",
                        f"Removed '{drop_col}' (correlated with '{pair['keep_feature']}')",
                        f"Distance-based model - extremely high correlation (r={correlation:.3f}) of same feature type",
                        "warning"
                    )
                else:
                    self._log_decision(
                        "multicollinearity",
                        f"Keeping both '{col1}' and '{col2}' despite correlation",
                        f"Distance-based model - preserving feature diversity (r={correlation:.3f})",
                        "info"
                    )
                
            elif self.model_family == ModelFamily.TREE_BASED:
                # Tree-based: Moderate impact - log but keep
                self._log_decision(
                    "multicollinearity",
                    f"Detected correlation between '{col1}' and '{col2}'",
                    f"Tree-based model - keeping both features (r={correlation:.3f}). Trees handle correlation naturally.",
                    "info"
                )
                # Don't drop anything for tree-based models
                
            else:
                # Unknown/Neural Network: Moderate handling
                drop_col = pair["drop_feature"]
                columns_to_drop.add(drop_col)
                
                self._log_decision(
                    "multicollinearity",
                    f"Removed '{drop_col}' (correlated with '{pair['keep_feature']}')",
                    f"High correlation detected (r={correlation:.3f})",
                    "info"
                )
        
        # Drop columns
        if columns_to_drop:
            df = df.drop(columns=list(columns_to_drop))

            
            # Track in metadata
            self.preprocessing_metadata["multicollinearity_handled"] = {
                "pairs_detected": len(correlated_pairs),
                "features_dropped": list(columns_to_drop),
                "features_combined": features_combined,
                "model_family": self.model_family.value
            }
        
        return df
    
    def choose_scaling_strategy(self) -> str:
        """
        Choose scaling strategy based on model family and data characteristics
        
        Returns:
            Scaling strategy name
        """
        # Tree-based models don't need scaling
        if self.model_family == ModelFamily.TREE_BASED:
            self._log_decision(
                "scaling",
                "No scaling for tree-based models",
                "Decision trees are scale-invariant",
                "info"
            )
            return "none"
        
        # Check for outliers in numerical features
        has_outliers = any(
            self._has_significant_outliers(col) 
            for col in self.numerical_features if col in self.eda_results.get('outliers', {}).get('outliers_by_column', {})
        )
        
        # Linear and distance-based models need scaling
        if self.model_family in [ModelFamily.LINEAR, ModelFamily.DISTANCE_BASED]:
            if has_outliers:
                self._log_decision(
                    "scaling",
                    "Using RobustScaler",
                    f"Outliers detected in numerical features - robust scaling recommended",
                    "info"
                )
                return "robust"
            else:
                self._log_decision(
                    "scaling",
                    "Using StandardScaler",
                    "Standard scaling for linear/distance-based models",
                    "info"
                )
                return "standard"
        
        # Neural networks typically benefit from normalization
        if self.model_family == ModelFamily.NEURAL_NETWORK:
            self._log_decision(
                "scaling",
                "Using MinMaxScaler",
                "Min-max normalization for neural networks",
                "info"
            )
            return "minmax"
        
        # Default: standard scaling
        self._log_decision(
            "scaling",
            "Using StandardScaler (default)",
            "General-purpose scaling",
            "info"
        )
        return "standard"
    
    def choose_encoding_strategy(self, column: str) -> str:
        """
        Choose encoding strategy based on model family and cardinality
        
        Args:
            column: Categorical column name
        
        Returns:
            Encoding strategy name
        """
        # Get cardinality from EDA or direct count
        if self.eda_results:
            cat_analysis = self.eda_results.get('categorical_analysis', {})
            cat_features = cat_analysis.get('categorical_features', {})
            if column in cat_features:
                unique_count = cat_features[column]['unique_count']
            else:
                return "onehot"  # Default
        else:
            return "onehot"  # Default
        
        # Decision logic:
        # - Tree-based models: prefer ordinal (faster, handles high cardinality)
        # - Linear models: prefer one-hot (captures non-linear relationships)
        # - High cardinality: ordinal or target encoding
        
        if self.model_family == ModelFamily.TREE_BASED:
            return "ordinal"
        elif unique_count > self.HIGH_CARDINALITY_THRESHOLD:
            self._log_decision(
                "encoding",
                f"Using ordinal encoding for '{column}'",
                f"High cardinality ({unique_count} categories)",
                "warning"
            )
            return "ordinal"
        else:
            return "onehot"
    
    def create_preprocessing_pipeline(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        model_type: Optional[str] = None
    ) -> Pipeline:
        """
        Create adaptive preprocessing pipeline
        
        Args:
            df: Input DataFrame
            target_column: Optional target column
            model_type: Optional model type for optimization
        
        Returns:
            Sklearn preprocessing pipeline
        """
        self._log_decision("pipeline_creation", "Creating adaptive preprocessing pipeline", "", "info")
        
        # Set context
        self.target_column = target_column
        self.detect_task_type(df, target_column)
        self.set_model_family(model_type)
        
        # NOTE: Problematic columns are already removed in fit_transform() before this is called
        # Don't remove them again here - just categorize the remaining features
        
        # Categorize features
        self.categorize_features(df, target_column)
        
        print(f"\n{'='*80}")
        print(f"🔍 DEBUG: After categorize_features in create_preprocessing_pipeline")
        print(f"   Numerical features: {self.numerical_features}")
        print(f"   Categorical features: {self.categorical_features}")
        print(f"{'='*80}\n")
        
        # Choose strategies
        scaling_strategy = self.choose_scaling_strategy()
        
        # Build transformers
        transformers = []
        
        # Numerical pipeline
        if self.numerical_features:
            numerical_steps = []
            
            # Imputation - PER-COLUMN strategy
            # MUST come BEFORE outlier capping (so percentile calculations work on complete data)
            # Analyze each numerical column and determine best imputation strategy
            columns_to_drop = []
            columns_with_knn = []
            columns_with_indicators = {}
            
            for col in self.numerical_features:
                # Check if column has missing values
                missing_count = df[col].isnull().sum()
                if missing_count == 0:
                    continue  # Skip columns without missing values
                
                imp_config = self.choose_imputation_strategy(df, "numerical", col)
                
                # Track columns to drop (>80% missing)
                if imp_config['drop']:
                    columns_to_drop.append(col)
                    continue
                
                # Track KNN columns
                if imp_config['strategy'] == 'knn':
                    columns_with_knn.append(col)
                
                # Track columns needing indicators
                if imp_config['add_indicator']:
                    columns_with_indicators[col] = imp_config['strategy']
            
            # Remove columns with excessive missing values
            if columns_to_drop:
                df = df.drop(columns=columns_to_drop)
                self.numerical_features = [c for c in self.numerical_features if c not in columns_to_drop]
                # Also add to dropped columns and high_missing_columns for tracking
                self.dropped_columns.extend(columns_to_drop)
                self.high_missing_columns.extend(columns_to_drop)
                self._log_decision(
                    "drop_high_missing",
                    f"Dropped {len(columns_to_drop)} numerical columns",
                    f"Columns with >80% missing: {', '.join(columns_to_drop)}",
                    "warning"
                )
            
            # If we still have numerical features after dropping, add imputation
            if self.numerical_features:
                # Use KNN imputer if any columns need it (applies to all)
                if columns_with_knn:
                    imputer = KNNImputer(n_neighbors=5, weights='distance')
                    numerical_steps.append(('imputer', imputer))
                    
                    # Track imputation details
                    for col in columns_with_knn:
                        missing_count = df[col].isnull().sum()
                        self.imputation_details.append({
                            'column': col,
                            'strategy': 'KNN (k=5)',
                            'missing_count': int(missing_count),
                            'type': 'numerical'
                        })
                    
                    self._log_decision(
                        "imputation",
                        f"KNN imputation for {len(columns_with_knn)} columns",
                        f"Columns: {', '.join(columns_with_knn)}",
                        "info"
                    )
                else:
                    # Use SimpleImputer with median as default (most robust)
                    # Check if most columns have outliers
                    outlier_count = sum(1 for col in self.numerical_features 
                                      if col in df.columns and self._has_significant_outliers(col))
                    
                    if outlier_count > len(self.numerical_features) / 2:
                        imputer = SimpleImputer(strategy='median', add_indicator=len(columns_with_indicators) > 0)
                        strategy_name = 'median'
                    else:
                        imputer = SimpleImputer(strategy='mean', add_indicator=len(columns_with_indicators) > 0)
                        strategy_name = 'mean'
                    
                    numerical_steps.append(('imputer', imputer))
                    
                    # Track imputation details for columns with missing values
                    for col in self.numerical_features:
                        missing_count = df[col].isnull().sum()
                        if missing_count > 0:
                            self.imputation_details.append({
                                'column': col,
                                'strategy': strategy_name.capitalize(),
                                'missing_count': int(missing_count),
                                'type': 'numerical'
                            })
                    
                    indicator_note = f" + {len(columns_with_indicators)} indicator features" if columns_with_indicators else ""
                    self._log_decision(
                        "imputation",
                        f"Numerical imputation: {strategy_name}{indicator_note}",
                        f"Applies to {len([c for c in self.numerical_features if df[c].isnull().sum() > 0])} columns",
                        "info"
                    )
            
            # Outlier handling - AFTER imputation (so percentile calculations work on complete data)
            # Combine automatic detection with user preferences
            columns_to_cap = []
            columns_to_remove_outliers = []
            
            # First, check user preferences
            for col in self.numerical_features:
                preference = self.outlier_preferences.get(col, 'keep')  # Default: keep

                preference = self.outlier_preferences.get(col, 'keep')  # Default: keep
                
                if preference == 'cap':
                    columns_to_cap.append(col)
                elif preference == 'remove':
                    columns_to_remove_outliers.append(col)
            
            # If no user preferences and model is LINEAR/DISTANCE_BASED, auto-detect outliers
            if not columns_to_cap and self.model_family in [ModelFamily.LINEAR, ModelFamily.DISTANCE_BASED]:
                columns_with_outliers = [
                    col for col in self.numerical_features
                    if col in df.columns and self._has_significant_outliers(col)
                ]
                if columns_with_outliers:
                    columns_to_cap = columns_with_outliers
            
            # Add outlier capping transformer if needed
            if columns_to_cap:
                outlier_capper = OutlierCapper(iqr_multiplier=2.5)
                numerical_steps.append(('outlier_capper', outlier_capper))
                
                # Track outlier handling with details from EDA
                self.outliers_handled_count = len(columns_to_cap)
                self.outlier_columns = columns_to_cap
                
                # Get detailed outlier information from EDA results
                outliers_by_column = self.eda_results.get('outliers', {}).get('outliers_by_column', {})
                for col in columns_to_cap:
                    if col in outliers_by_column:
                        outlier_info = outliers_by_column[col]
                        self.outlier_details.append({
                            'column': col,
                            'count': outlier_info.get('count', 0),
                            'percentage': outlier_info.get('percentage', 0),
                            'lower_bound': outlier_info.get('lower_bound', 0),
                            'upper_bound': outlier_info.get('upper_bound', 0),
                            'action': 'capped'
                        })
                
                reason = "user-selected" if any(self.outlier_preferences.values()) else f"{self.model_family.value} models"
                self._log_decision(
                    "outlier_handling",
                    f"Capping outliers in {len(columns_to_cap)} columns",
                    f"Using IQR method (2.5×IQR) - {reason}. "
                    f"Outliers capped to bounds (not removed). "
                    f"Columns: {', '.join(columns_to_cap[:5])}{'...' if len(columns_to_cap) > 5 else ''}",
                    "info"
                )
            
            # Skewness transformation (before scaling)
            # Check which columns are highly skewed and would benefit from transformation
            skewed_columns = []
            skewness_info = {}
            
            for col in self.numerical_features:
                skew_data = self.detect_skewness(df, col)
                if skew_data['needs_transformation']:
                    skewed_columns.append(col)
                    skewness_info[col] = skew_data
                    self.skewness_details.append({
                        'column': col,
                        'skewness': skew_data['skewness'],
                        'method': skew_data['method'],
                        'transformed': True
                    })
            
            # Apply transformation if needed
            # Linear and distance-based models benefit most from skewness correction
            if skewed_columns and self.model_family in [ModelFamily.LINEAR, ModelFamily.DISTANCE_BASED, ModelFamily.NEURAL_NETWORK]:
                # Use Yeo-Johnson (most versatile - handles any data)
                power_transformer = PowerTransformer(method='yeo-johnson', standardize=False)
                numerical_steps.append(('skewness_transform', power_transformer))
                
                self.skewed_features_transformed = skewed_columns
                
                self._log_decision(
                    "skewness_handling",
                    f"Yeo-Johnson transformation for {len(skewed_columns)} skewed features",
                    f"Correcting skewness (|skew| > 0.75) for {self.model_family.value} models. "
                    f"Columns: {', '.join(skewed_columns[:5])}{'...' if len(skewed_columns) > 5 else ''}",
                    "info"
                )
            elif skewed_columns:
                # For tree-based models, log skewed columns but don't transform
                for col, skew_data in skewness_info.items():
                    self.skewness_details.append({
                        'column': col,
                        'skewness': skew_data['skewness'],
                        'method': None,
                        'transformed': False
                    })
                
                self._log_decision(
                    "skewness_handling",
                    f"Skewness detected in {len(skewed_columns)} features",
                    f"No transformation applied - {self.model_family.value} models are robust to skewness",
                    "info"
                )
            
            # Scaling
            if scaling_strategy == "standard":
                scaler = StandardScaler()
                numerical_steps.append(('scaler', scaler))
                self._log_decision(
                    "scaling",
                    "StandardScaler applied to numerical features",

                    "info"
                )
            elif scaling_strategy == "robust":
                scaler = RobustScaler()
                numerical_steps.append(('scaler', scaler))
                self._log_decision(
                    "scaling",
                    "RobustScaler applied to numerical features",

                    "info"
                )
            elif scaling_strategy == "minmax":
                scaler = MinMaxScaler()
                numerical_steps.append(('scaler', scaler))
                self._log_decision(
                    "scaling",
                    "MinMaxScaler applied to numerical features",

                    "info"
                )
            # else: scaling_strategy is "none", no scaler added
            
            # If we have numerical features but no steps, add a passthrough
            # This ensures the features are included even without transformation
            if numerical_steps:
                numerical_pipeline = Pipeline(steps=numerical_steps)
                transformers.append(('num', numerical_pipeline, self.numerical_features))
            else:
                # No transformation needed, but we still need to include the features
                from sklearn.preprocessing import FunctionTransformer
                passthrough = FunctionTransformer()
                numerical_pipeline = Pipeline(steps=[('passthrough', passthrough)])
                transformers.append(('num', numerical_pipeline, self.numerical_features))
                self._log_decision(
                    "passthrough",
                    "Numerical features passed through without transformation",
                    "Tree-based models don't require scaling or imputation",
                    "info"
                )
        
        # Categorical pipeline
        if self.categorical_features:
            categorical_steps = []
            
            # Imputation - PER-COLUMN strategy
            cat_columns_to_drop = []
            cat_columns_with_indicators = {}
            
            for col in self.categorical_features:
                # Check if column has missing values
                missing_count = df[col].isnull().sum()
                if missing_count == 0:
                    continue  # Skip columns without missing values
                
                imp_config = self.choose_imputation_strategy(df, "categorical", col)
                
                # Track columns to drop (>80% missing)
                if imp_config['drop']:
                    cat_columns_to_drop.append(col)
                    continue
                
                # Track columns needing indicators
                if imp_config['add_indicator']:
                    cat_columns_with_indicators[col] = imp_config['strategy']
            
            # Remove columns with excessive missing values
            if cat_columns_to_drop:
                df = df.drop(columns=cat_columns_to_drop)
                self.categorical_features = [c for c in self.categorical_features if c not in cat_columns_to_drop]
                # Also add to dropped columns and high_missing_columns for tracking
                self.dropped_columns.extend(cat_columns_to_drop)
                self.high_missing_columns.extend(cat_columns_to_drop)
                self._log_decision(
                    "drop_high_missing",
                    f"Dropped {len(cat_columns_to_drop)} categorical columns",
                    f"Columns with >80% missing: {', '.join(cat_columns_to_drop)}",
                    "warning"
                )
            
            # If we still have categorical features after dropping
            if self.categorical_features:
                # Use most_frequent (mode) for categorical with optional indicator
                imputer = SimpleImputer(
                    strategy='most_frequent',
                    add_indicator=len(cat_columns_with_indicators) > 0
                )
                categorical_steps.append(('imputer', imputer))
                
                # Track imputation details for columns with missing values
                for col in self.categorical_features:
                    missing_count = df[col].isnull().sum()
                    if missing_count > 0:
                        self.imputation_details.append({
                            'column': col,
                            'strategy': 'Mode (most frequent)',
                            'missing_count': int(missing_count),
                            'type': 'categorical'
                        })
                
                indicator_note = f" + {len(cat_columns_with_indicators)} indicator features" if cat_columns_with_indicators else ""
                self._log_decision(
                    "imputation",
                    f"Categorical imputation: mode{indicator_note}",
                    f"Applies to {len([c for c in self.categorical_features if df[c].isnull().sum() > 0])} columns",
                    "info"
                )
            
            # Encoding - use same strategy for all (could be per-column in advanced version)
            sample_col = self.categorical_features[0] if self.categorical_features else None
            encoding_strategy = self.choose_encoding_strategy(sample_col)
            
            if encoding_strategy == "onehot":
                encoder = OneHotEncoder(
                    handle_unknown='ignore',
                    sparse_output=False,
                    dtype=np.float64
                )
                # Track encoding details for each column
                for col in self.categorical_features:
                    self.encoding_details.append({
                        'column': col,
                        'strategy': 'One-Hot Encoding',
                        'creates_new_columns': True
                    })
                self._log_decision(
                    "encoding",
                    "Using OneHotEncoder",

                    "info"
                )
            else:  # ordinal
                encoder = OrdinalEncoder(
                    handle_unknown='use_encoded_value',
                    unknown_value=-1,
                    dtype=np.float64
                )
                # Track encoding details for each column
                for col in self.categorical_features:
                    self.encoding_details.append({
                        'column': col,
                        'strategy': 'Ordinal Encoding',
                        'creates_new_columns': False
                    })
                self._log_decision(
                    "encoding",
                    "Using OrdinalEncoder",
                    f"Ordinal encoding for {len(self.categorical_features)} categorical columns",
                    "info"
                )
            
            categorical_steps.append(('encoder', encoder))
            categorical_pipeline = Pipeline(steps=categorical_steps)
            transformers.append(('cat', categorical_pipeline, self.categorical_features))
        
        # Text features (TF-IDF vectorization)
        if self.text_features:
            print(f"\n📝 Creating text pipeline for {len(self.text_features)} text columns: {self.text_features}")
            
            text_pipeline = Pipeline(steps=[
                ('vectorizer', TextVectorizer(
                    max_features=5000,
                    min_df=2,
                    max_df=0.95
                ))
            ])
            
            transformers.append(('text', text_pipeline, self.text_features))
            
            # Create detailed message for UI
            text_cols_str = ", ".join(self.text_features[:3])
            if len(self.text_features) > 3:
                text_cols_str += f" (+{len(self.text_features) - 3} more)"
            
            self._log_decision(
                "text_vectorization",
                f"Vectorizing {len(self.text_features)} text column(s) with TF-IDF",
                f"Columns: {text_cols_str} | 5000 features per column",
                "info"
            )
        
        # Datetime features (drop for now, could extract features in future)
        if self.datetime_features:
            self._log_decision(
                "datetime_handling",
                f"Dropping {len(self.datetime_features)} datetime columns",
                "Datetime feature extraction not yet implemented",
                "warning"
            )
        
        # Create column transformer with sparse output support
        if transformers:
            preprocessor = ColumnTransformer(
                transformers=transformers,
                sparse_threshold=0.3  # Use sparse matrix if >30% of features are sparse (e.g., text features)
            )
            self.pipeline = preprocessor
        else:
            # Provide helpful error message
            error_msg = "No features available for preprocessing after column removal. "

            error_msg += f"(ID: {len(self.id_columns)}, Constant: {len(self.constant_columns)}, "
            error_msg += f"High missing: {len(self.high_missing_columns)}). "
            error_msg += "Dataset may be too small or have too many problematic columns."
            raise ValueError(error_msg)
        
        return self.pipeline
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        target_column: Optional[str] = None,
        model_type: Optional[str] = None,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """
        Complete preprocessing workflow with train/test split
        
        Args:
            df: Input DataFrame
            target_column: Optional target column name
            model_type: Optional model type for optimization
            test_size: Test set size (0 for no split)
            random_state: Random seed
        
        Returns:
            Dictionary with preprocessed data and metadata
        """
        self._log_decision("preprocessing", "Starting adaptive preprocessing", "", "info")
        
        initial_shape = df.shape
        
        # Remove problematic columns FIRST (before creating pipeline)
        df_cleaned = self.detect_and_remove_problematic_columns(df, target_column)
        
        # Remove duplicate rows (must be done before splitting to maintain consistency)
        df_cleaned = self.detect_and_remove_duplicates(df_cleaned)
        print(f"\n{'='*80}")
        print(f"🔍 DEBUG: After detect_and_remove_duplicates")
        print(f"   DataFrame shape: {df_cleaned.shape}")
        print(f"   Columns: {df_cleaned.columns.tolist()}")
        print(f"{'='*80}\n")
        
        # Remove outlier rows if user requested (must be before splitting)
        df_cleaned = self.remove_outlier_rows(df_cleaned, target_column)
        print(f"\n{'='*80}")
        print(f"🔍 DEBUG: After remove_outlier_rows")
        print(f"   DataFrame shape: {df_cleaned.shape}")
        print(f"   Outlier rows removed: {self.outlier_rows_removed}")
        print(f"{'='*80}\n")
        
        # Detect and parse datetime columns BEFORE feature categorization
        df_cleaned = self.detect_and_parse_datetime(df_cleaned)
        print(f"\n{'='*80}")
        print(f"🔍 DEBUG: After detect_and_parse_datetime")
        print(f"   DataFrame shape: {df_cleaned.shape}")
        print(f"   Datetime features detected: {self.datetime_features}")
        print(f"   Datetime column dtypes: {df_cleaned[self.datetime_features].dtypes.to_dict() if self.datetime_features else 'None'}")
        print(f"{'='*80}\n")
        
        # Extract datetime features based on model family
        print(f"\n{'='*80}")
        print(f"🔍 DEBUG: About to call extract_datetime_features")
        print(f"   target_column parameter: '{target_column}'")
        print(f"   DataFrame shape: {df_cleaned.shape}")
        print(f"   Datetime features to process: {self.datetime_features}")
        print(f"{'='*80}\n")
        
        df_cleaned = self.extract_datetime_features(df_cleaned, target_column)
        print(f"\n{'='*80}")
        print(f"🔍 DEBUG: After extract_datetime_features")
        print(f"   DataFrame shape: {df_cleaned.shape}")
        print(f"   Columns: {df_cleaned.columns.tolist()}")
        print(f"   New features created: {self.preprocessing_metadata.get('datetime_features', {}).get('features_created', [])}")
        print(f"   Datetime columns dropped: {self.preprocessing_metadata.get('datetime_features', {}).get('columns_dropped', [])}")
        print(f"   Numerical features: {self.numerical_features}")
        print(f"   Categorical features: {self.categorical_features}")
        print(f"{'='*80}\n")
        
        # Handle multicollinearity (must be before pipeline creation)
        correlated_pairs = self.detect_high_correlation(df_cleaned, threshold=0.9, target_column=target_column)
        if correlated_pairs:
            print(f"\n{'='*80}")
            print(f"🔍 DEBUG: Detected {len(correlated_pairs)} highly correlated feature pairs")
            for pair in correlated_pairs[:5]:  # Show first 5
                print(f"   {pair['feature1']} <-> {pair['feature2']}: r={pair['correlation']:.3f}, relationship={pair['semantic_relationship']}")
            print(f"{'='*80}\n")
            
            df_cleaned = self.handle_multicollinearity(df_cleaned, correlated_pairs, target_column)
            
            print(f"\n{'='*80}")
            print(f"🔍 DEBUG: After handle_multicollinearity")

            print(f"   Columns ({len(df_cleaned.columns)}): {df_cleaned.columns.tolist()}")
            print(f"{'='*80}\n")
        
        # Create pipeline AFTER removing problematic columns (so it knows correct feature count)
        # NOTE: create_preprocessing_pipeline may identify additional columns to drop (high-missing categorical)
        self.create_preprocessing_pipeline(df_cleaned, target_column, model_type)
        print(f"\n{'='*80}")
        print(f"🔍 DEBUG: After create_preprocessing_pipeline")
        print(f"   DataFrame shape: {df_cleaned.shape}")
        print(f"   Columns ({len(df_cleaned.columns)}): {df_cleaned.columns.tolist()}")
        print(f"   Numerical features: {self.numerical_features}")
        print(f"   Categorical features: {self.categorical_features}")
        if self.pipeline and hasattr(self.pipeline, 'transformers'):
            print(f"   Pipeline transformers: {[(name, type(trans).__name__, cols) for name, trans, cols in self.pipeline.transformers]}")
        else:
            print(f"   Pipeline: {type(self.pipeline).__name__ if self.pipeline else 'None'}")
        print(f"{'='*80}\n")
        
        # Drop any additional columns identified during pipeline creation (e.g., high-missing categorical)
        additional_drops = [col for col in self.dropped_columns if col in df_cleaned.columns]
        
        if additional_drops:
            print(f"🧹 Dropping {len(additional_drops)} additional high-missing columns: {additional_drops}")
            df_cleaned = df_cleaned.drop(columns=additional_drops)
            print(f"   DataFrame shape after additional drops: {df_cleaned.shape}")
            print(f"   Columns ({len(df_cleaned.columns)}): {df_cleaned.columns.tolist()}")
        
        print(f"\n{'='*80}")
        print(f"🔍 DEBUG: Before separating features and target")
        print(f"   DataFrame shape: {df_cleaned.shape}")
        print(f"   Target column: {target_column}")
        print(f"{'='*80}\n")
        
        # Separate features and target
        if target_column and target_column in df_cleaned.columns:
            X = df_cleaned.drop(columns=[target_column])
            y = df_cleaned[target_column].values
            
            # Convert datetime target to numeric (Unix timestamp)
            if pd.api.types.is_datetime64_any_dtype(df_cleaned[target_column]):
                # Convert to Unix timestamp, handling NaT values
                y_datetime = pd.to_datetime(y)
                
                # Check if this is a time-only column (all dates are the same)
                if pd.notna(y_datetime).any():
                    # Create a Series to access .dt accessor
                    y_series = pd.Series(y_datetime)
                    unique_dates = y_series.dt.date.nunique()
                    if unique_dates == 1:
                        # Time-only: convert to seconds since midnight
                        y = (y_series.dt.hour * 3600 + 
                             y_series.dt.minute * 60 + 
                             y_series.dt.second).values.astype(float)
                        self._log_decision(
                            "target_encoding",
                            f"Converted time-only target '{target_column}' to seconds since midnight",
                            f"Range: {np.nanmin(y):.0f} to {np.nanmax(y):.0f} seconds (0 = midnight, 86399 = 23:59:59)",
                            "info"
                        )
                    else:
                        # Full datetime: convert to Unix timestamp
                        y = y_datetime.astype('int64') / 10**9  # Convert to Unix timestamp (seconds)
                        
                        # Check for NaT/NaN values
                        nan_count = np.isnan(y).sum()
                        if nan_count > 0:
                            self._log_decision(
                                "target_encoding",
                                f"Warning: {nan_count} NaT values in datetime target",
                                f"These will become NaN in the target variable",
                                "warning"
                            )
                        
                        self._log_decision(
                            "target_encoding",
                            f"Converted datetime target '{target_column}' to Unix timestamp",
                            f"Range: {np.nanmin(y):.0f} to {np.nanmax(y):.0f} seconds since epoch (excluding NaN)",
                            "info"
                        )
                else:
                    # All NaT values
                    y = y_datetime.astype('int64') / 10**9
            
            # CRITICAL: Remove rows with NaN target BEFORE encoding or class balancing
            # NaN in target cannot be used for supervised learning
            if pd.isna(y).any():
                nan_mask = pd.isna(y)
                nan_count = nan_mask.sum()
                
                # Track removal
                self.nan_target_rows_removed = int(nan_count)
                
                # Remove rows with NaN target
                X = X[~nan_mask].reset_index(drop=True)
                y = y[~nan_mask]
                
                self._log_decision(
                    "target_handling",
                    f"Removed {nan_count} rows with NaN/missing target values",
                    f"Remaining samples: {len(y)} (Cannot train on missing targets)",
                    "warning"
                )
                
                # Update initial shape to reflect this removal
                initial_shape = (len(y), initial_shape[1])
            
            # Encode target if classification
            if self.task_type == TaskType.CLASSIFICATION:
                self.label_encoder = LabelEncoder()
                y = self.label_encoder.fit_transform(y)
                self._log_decision(
                    "target_encoding",
                    f"Encoded target '{target_column}'",
                    f"Classes: {list(self.label_encoder.classes_)}",
                    "info"
                )
                
                # Detect class imbalance (only for classification)
                imbalance_info = self._detect_class_imbalance(y)
            else:
                # For regression or unsupervised, no class imbalance
                imbalance_info = None
        else:
            X = df_cleaned
            y = None
            imbalance_info = None
        
        # CRITICAL: Split BEFORE preprocessing to avoid data leakage
        # Train/test split if supervised
        if y is not None and test_size > 0:
            # Decide whether we can stratify: stratification requires at least 2 samples
            # in each class so that each split can contain at least one sample per class.
            stratify_param = None
            if self.task_type == TaskType.CLASSIFICATION:
                try:
                    unique, counts = np.unique(y, return_counts=True)
                    min_count = int(np.min(counts)) if len(counts) > 0 else 0
                except Exception:
                    # Fallback: if any issue computing counts, do not stratify
                    unique, counts = None, None
                    min_count = 0

                if min_count >= 2:
                    stratify_param = y
                else:
                    # Can't stratify when a class has fewer than 2 members
                    self._log_decision(
                        "data_split",
                        "Requested stratified split but the least populated class has fewer than 2 samples. Falling back to unstratified random split.",
                        f"Class counts: {dict(zip(unique.tolist() if unique is not None else [], counts.tolist() if counts is not None else []))}",
                        "warning"
                    )

            X_train_raw, X_test_raw, y_train, y_test = train_test_split(
                X, y,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify_param
            )
            
            self._log_decision(
                "data_split",
                f"Split data BEFORE preprocessing: {len(X_train_raw)} train, {len(X_test_raw)} test",
                f"Test size: {test_size*100:.0f}% (prevents data leakage)",
                "info"
            )
            
            # Store original split sizes (before SMOTE)
            original_train_size = len(X_train_raw)
            original_test_size = len(X_test_raw)
            
            print(f"\n{'='*80}")
            print(f"🔍 DEBUG: About to fit pipeline on TRAINING data")
            print(f"   X_train_raw shape: {X_train_raw.shape}")
            print(f"   X_train_raw columns ({X_train_raw.shape[1]}): {X_train_raw.columns.tolist()}")




            print(f"{'='*80}\n")
            
            # Fit pipeline on TRAINING data only
            X_train = self.pipeline.fit_transform(X_train_raw)
            
            # Transform TEST data using the fitted pipeline (no fitting on test!)
            X_test = self.pipeline.transform(X_test_raw)
            
            self._log_decision(
                "preprocessing_application",
                "Fitted preprocessing on TRAIN data only",
                "Applied same transformations to TEST data (no leakage)",
                "info"
            )
            
            # Handle class imbalance on TRAINING data only (after preprocessing)
            if imbalance_info and imbalance_info['is_imbalanced'] and self.task_type == TaskType.CLASSIFICATION:
                X_train, y_train = self._handle_class_imbalance(X_train, y_train, imbalance_info)
            
        else:
            # No split - fit and transform all data
            original_train_size = len(X)
            original_test_size = 0
            
            print(f"\n{'='*80}")
            print(f"🔍 DEBUG: About to fit pipeline on ALL data (unsupervised)")
            print(f"   X shape: {X.shape}")
            print(f"   X type: {type(X)}")
            print(f"   X columns ({len(X.columns)}): {X.columns.tolist()}")
            print(f"   Numerical features: {self.numerical_features}")
            print(f"   Categorical features: {self.categorical_features}")
            
            # Check if features exist in X
            missing_numerical = [f for f in self.numerical_features if f not in X.columns]
            missing_categorical = [f for f in self.categorical_features if f not in X.columns]
            print(f"   Missing numerical features: {missing_numerical}")
            print(f"   Missing categorical features: {missing_categorical}")
            
            # DEBUG: Check if outlier capper columns exist
            if hasattr(self, 'outlier_columns'):
                print(f"   Outlier columns to cap: {self.outlier_columns}")
                outlier_missing = [f for f in self.outlier_columns if f not in X.columns]
                print(f"   Missing outlier columns: {outlier_missing}")
            
            # DEBUG: Verify data types
            print(f"   X dtypes:")
            for col in X.columns[:5]:  # Show first 5
                print(f"      {col}: {X[col].dtype}, sample: {X[col].iloc[0] if len(X) > 0 else 'N/A'}")
            
            print(f"   Pipeline structure: {self.pipeline}")
            print(f"{'='*80}\n")
            
            X_train = self.pipeline.fit_transform(X)
            X_test, y_train, y_test = None, y, None
            
            # Handle class imbalance on all data (if no split)
            if imbalance_info and imbalance_info['is_imbalanced'] and self.task_type == TaskType.CLASSIFICATION:
                X_train, y_train = self._handle_class_imbalance(X_train, y_train, imbalance_info)
        
        # Ensure float64
        if X_train.dtype != np.float64:
            X_train = X_train.astype(np.float64)
        if X_test is not None and X_test.dtype != np.float64:
            X_test = X_test.astype(np.float64)
        
        # Get feature names
        self.feature_names = self._extract_feature_names()
        
        # Calculate quality metrics (use combined shape for reporting)
        if X_test is not None:
            combined_shape = (X_train.shape[0] + X_test.shape[0], X_train.shape[1])
        else:
            combined_shape = X_train.shape
        
        quality_metrics = self._calculate_quality_metrics(
            initial_shape, combined_shape, df
        )
        
        # Prepare results
        total_columns_removed = (
            len(self.id_columns) + 
            len(self.constant_columns) + 
            len(self.low_variance_columns) + 
            len(self.high_missing_columns) +
            len(getattr(self, 'high_cardinality_columns', []))
        )
        
        results = {
            "X_train": X_train,
            "X_test": X_test,
            "y_train": y_train,
            "y_test": y_test,
            "feature_names": self.feature_names,
            "task_type": self.task_type.value if self.task_type else None,
            "model_family": self.model_family.value,
            "preprocessing_metadata": {
                "initial_shape": (int(initial_shape[0]), int(initial_shape[1])),
                "final_shape": (int(combined_shape[0]), int(combined_shape[1])),
                "numerical_features": self.numerical_features,
                "categorical_features": self.categorical_features,
                "text_features": self.text_features,
                "total_columns_removed": total_columns_removed,
                "datetime_features_details": self.preprocessing_metadata.get("datetime_features", {
                    "original_columns": [],
                    "features_created": [],
                    "columns_dropped": []
                }),
                "id_columns_removed": self.id_columns,
                "constant_columns_removed": self.constant_columns + self.low_variance_columns,  # Combined list
                "constant_column_details": self.constant_column_details,  # Detailed info for each
                "high_missing_columns_removed": self.high_missing_columns,
                "high_cardinality_columns_removed": getattr(self, 'high_cardinality_columns', []),

                "target_classes": [str(c) for c in self.label_encoder.classes_] if self.label_encoder else None,
                "outliers_handled_count": int(self.outliers_handled_count),
                "outlier_columns": self.outlier_columns,
                "outlier_details": self.outlier_details,
                "outlier_rows_removed": int(self.outlier_rows_removed),
                "skewness_details": self.skewness_details,
                "skewed_features_transformed": self.skewed_features_transformed,
                "imputation_details": self.imputation_details,
                "encoding_details": self.encoding_details,
                "duplicates_removed_count": int(self.duplicates_removed_count),
                "duplicate_removal_details": self.duplicate_removal_details,
                "nan_target_rows_removed": int(self.nan_target_rows_removed),
                "class_imbalance_handled": self.class_imbalance_handled,
                "imbalance_strategy": self.imbalance_strategy,
                "original_class_distribution": self.original_class_distribution,
                "resampled_class_distribution": self.resampled_class_distribution,
                "original_train_size": int(original_train_size),
                "original_test_size": int(original_test_size),
                "multicollinearity_handled": self.preprocessing_metadata.get("multicollinearity_handled", {
                    "pairs_detected": 0,
                    "features_dropped": [],
                    "features_combined": [],
                    "model_family": self.model_family.value
                }),
            },
            "quality_metrics": quality_metrics,
            "decision_log": self.decision_log,
        }
        
        self._log_decision(
            "preprocessing_complete",
            f"Preprocessing complete: {initial_shape} → {combined_shape}",
            f"Quality score: {quality_metrics['quality_score']:.1f}/100",
            "info"
        )
        
        return results
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform new data using fitted pipeline
        
        Args:
            df: Input DataFrame
        
        Returns:
            Transformed array (may be sparse if text features present)
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not fitted. Call fit_transform first.")
        
        # Create copy to avoid modifying original
        df_cleaned = df.copy()
        
        # Remove same columns as during fitting
        columns_to_drop = set()
        
        # Add all columns that were dropped during training
        columns_to_drop.update(self.id_columns)
        columns_to_drop.update(self.constant_columns)
        columns_to_drop.update(self.low_variance_columns)
        columns_to_drop.update(self.high_missing_columns)
        columns_to_drop.update(self.datetime_features)  # Datetime features are currently dropped
        
        # Remove target column if present
        if self.target_column and self.target_column in df_cleaned.columns:
            columns_to_drop.add(self.target_column)
        
        # Drop all identified columns
        existing_cols_to_drop = [col for col in columns_to_drop if col in df_cleaned.columns]
        if existing_cols_to_drop:
            df_cleaned = df_cleaned.drop(columns=existing_cols_to_drop)
        
        return self.pipeline.transform(df_cleaned)
    
    def _extract_feature_names(self) -> List[str]:
        """Extract feature names after transformation"""
        feature_names = []
        
        print(f"\n{'='*80}")
        print(f"🔍 DEBUG: _extract_feature_names")
        print(f"   Pipeline type: {type(self.pipeline).__name__}")
        
        # Check if pipeline has been fitted
        if not hasattr(self.pipeline, 'transformers_'):
            print(f"   ⚠️  WARNING: Pipeline not fitted yet, using original transformers")
            transformers_list = self.pipeline.transformers
        else:
            print(f"   Number of transformers: {len(self.pipeline.transformers_)}")
            transformers_list = self.pipeline.transformers_
        
        for name, transformer, columns in transformers_list:
            print(f"\n   Transformer: {name}")
            print(f"   Columns passed to transformer ({len(columns)}): {columns}")
            
            if name == 'num':
                # Check if numerical imputer adds indicator columns
                imputer = transformer.named_steps.get('imputer')
                actual_columns = list(columns)
                
                if imputer is not None and hasattr(imputer, 'indicator_') and imputer.indicator_ is not None:
                    print(f"   Numerical imputer adds indicators: True")
                    print(f"   Indicator features ({len(imputer.indicator_.features_)}): {imputer.indicator_.features_}")
                    # Add indicator column names for numerical features
                    for idx in imputer.indicator_.features_:
                        indicator_col_name = f"missingindicator_{columns[idx]}"
                        actual_columns.append(indicator_col_name)
                    print(f"   Actual columns after imputation ({len(actual_columns)}): {actual_columns}")
                else:
                    print(f"   Numerical imputer adds indicators: False")
                
                feature_names.extend(actual_columns)
                print(f"   ✓ Added {len(actual_columns)} numerical features")
            elif name == 'cat':
                encoder = transformer.named_steps['encoder']
                imputer = transformer.named_steps.get('imputer')
                
                # Get actual columns after imputation (including indicator columns)
                actual_columns = list(columns)
                if imputer is not None and hasattr(imputer, 'indicator_') and imputer.indicator_ is not None:
                    print(f"   Imputer adds indicators: True")
                    print(f"   Indicator features ({len(imputer.indicator_.features_)}): {imputer.indicator_.features_}")
                    # Add indicator column names
                    for idx in imputer.indicator_.features_:
                        indicator_col_name = f"missingindicator_{columns[idx]}"
                        actual_columns.append(indicator_col_name)
                    print(f"   Actual columns after imputation ({len(actual_columns)}): {actual_columns}")
                else:
                    print(f"   Imputer adds indicators: False")
                
                if isinstance(encoder, OneHotEncoder):
                    # OneHotEncoder creates multiple columns per feature
                    cat_features = encoder.get_feature_names_out(actual_columns)
                    feature_names.extend(cat_features.tolist())
                    print(f"   ✓ Added {len(cat_features)} one-hot encoded features")
                else:
                    # OrdinalEncoder keeps column structure but may have indicator columns
                    feature_names.extend(actual_columns)
                    print(f"   ✓ Added {len(actual_columns)} categorical features (ordinal encoded)")
            elif name == 'text':
                # Text vectorizer creates TF-IDF features
                vectorizer = transformer.named_steps['vectorizer']
                if hasattr(vectorizer, 'feature_names_'):
                    text_features = vectorizer.feature_names_
                    feature_names.extend(text_features)
                    print(f"   ✓ Added {len(text_features)} text features (TF-IDF)")
                else:
                    print(f"   ⚠️ Text vectorizer not fitted yet")
        
        print(f"\n   Total feature names: {len(feature_names)}")
        print(f"{'='*80}\n")
        
        return feature_names
    
    def _calculate_quality_metrics(
        self,
        initial_shape: Tuple[int, int],
        final_shape: Tuple[int, int],
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Calculate preprocessing quality metrics"""
        initial_rows, initial_cols = initial_shape
        final_rows, final_cols = final_shape
        
        # Missing data that was actually imputed (from imputation_details)
        total_cells = initial_rows * initial_cols
        missing_cells_imputed = sum(detail.get('missing_count', 0) for detail in self.imputation_details)
        
        print(f"\n{'='*80}")
        print(f"🔍 DEBUG: Quality Metrics Calculation")
        print(f"   Imputation details count: {len(self.imputation_details)}")
        for detail in self.imputation_details:
            print(f"   - {detail['column']}: {detail['missing_count']} ({detail['type']})")
        print(f"   Total missing cells imputed: {missing_cells_imputed}")
        print(f"{'='*80}\n")
        
        quality_score = 100.0
        quality_score -= (missing_cells_imputed / total_cells) * 30  # Penalty for missing data
        quality_score = max(0, min(100, quality_score))
        
        # Calculate total columns removed
        columns_removed = (
            len(self.id_columns) + 
            len(self.constant_columns) + 
            len(self.low_variance_columns) + 
            len(self.high_missing_columns) +
            len(getattr(self, 'high_cardinality_columns', []))
        )
        
        return {
            "quality_score": float(quality_score),
            "rows_removed": int(initial_rows - final_rows),
            "columns_removed": int(columns_removed),
            "features_generated": int(final_cols),
            "missing_data_handled": int(missing_cells_imputed),
            "data_retention_ratio": float(final_rows / initial_rows) if initial_rows > 0 else 0.0,
        }
    
    def save_pipeline(self, filepath: str):
        """Save preprocessing pipeline and metadata"""
        if self.pipeline is None:
            raise ValueError("No pipeline to save")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        pipeline_data = {
            'pipeline': self.pipeline,
            'label_encoder': self.label_encoder,
            'feature_names': self.feature_names,
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features,
            'datetime_features': self.datetime_features,
            'dropped_columns': self.dropped_columns,
            'task_type': self.task_type.value if self.task_type else None,
            'model_family': self.model_family.value,
            'target_column': self.target_column,
            'decision_log': self.decision_log,
        }
        
        joblib.dump(pipeline_data, filepath)
        self._log_decision("save", f"Pipeline saved to {filepath}", "", "info")
    
    def load_pipeline(self, filepath: str):
        """Load preprocessing pipeline and metadata"""
        pipeline_data = joblib.load(filepath)
        
        self.pipeline = pipeline_data['pipeline']
        self.label_encoder = pipeline_data['label_encoder']
        self.feature_names = pipeline_data['feature_names']
        self.numerical_features = pipeline_data.get('numerical_features', [])
        self.categorical_features = pipeline_data.get('categorical_features', [])
        self.datetime_features = pipeline_data.get('datetime_features', [])
        self.dropped_columns = pipeline_data.get('dropped_columns', [])
        self.target_column = pipeline_data['target_column']
        self.decision_log = pipeline_data.get('decision_log', [])
        
        # Restore enums
        task_type_str = pipeline_data.get('task_type')
        self.task_type = TaskType(task_type_str) if task_type_str else None
        
        model_family_str = pipeline_data.get('model_family')
        self.model_family = ModelFamily(model_family_str) if model_family_str else ModelFamily.UNKNOWN
        
        self._log_decision("load", f"Pipeline loaded from {filepath}", "", "info")
    
    def _detect_class_imbalance(self, y: np.ndarray) -> Dict[str, Any]:
        """
        Detect class imbalance in target variable
        
        Args:
            y: Encoded target values
            
        Returns:
            Dictionary with imbalance information
        """
        unique, counts = np.unique(y, return_counts=True)
        class_distribution = dict(zip(unique, counts))
        total_samples = len(y)
        
        # Calculate percentages
        class_percentages = {cls: (count / total_samples) * 100 
                           for cls, count in class_distribution.items()}
        
        # Find minority class
        minority_class = min(class_distribution, key=class_distribution.get)
        minority_percentage = class_percentages[minority_class]
        majority_class = max(class_distribution, key=class_distribution.get)
        majority_percentage = class_percentages[majority_class]
        
        # Imbalance ratio
        imbalance_ratio = class_distribution[majority_class] / class_distribution[minority_class]
        
        # Detect imbalance (threshold: minority class < 40% or imbalance ratio > 2.5)
        is_imbalanced = minority_percentage < 40.0 or imbalance_ratio > 2.5
        
        # Store original distribution
        self.original_class_distribution = {
            str(self.label_encoder.classes_[cls]): int(count)
            for cls, count in class_distribution.items()
        }
        
        if is_imbalanced:
            # Log warning
            class_names = list(self.label_encoder.classes_)
            distribution_str = ", ".join([
                f"{class_names[cls]}: {class_percentages[cls]:.1f}%"
                for cls in sorted(unique)
            ])
            
            self._log_decision(
                "class_imbalance",
                f"⚠️ Class imbalance detected (ratio: {imbalance_ratio:.1f}:1)",
                f"Distribution: {distribution_str}. "
                f"Minority class '{class_names[minority_class]}' is {minority_percentage:.1f}% of data. "
                f"Will apply resampling to balance classes.",
                "warning"
            )
        else:
            # Log that classes are balanced
            self._log_decision(
                "class_balance",
                f"✓ Classes are reasonably balanced (ratio: {imbalance_ratio:.1f}:1)",
                f"Minority class is {minority_percentage:.1f}% of data",
                "info"
            )
        
        return {
            "is_imbalanced": is_imbalanced,
            "imbalance_ratio": float(imbalance_ratio),
            "minority_class": int(minority_class),
            "minority_percentage": float(minority_percentage),
            "class_distribution": class_distribution,
            "total_samples": total_samples
        }
    
    def _handle_class_imbalance(
        self,
        X: np.ndarray,
        y: np.ndarray,
        imbalance_info: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Handle class imbalance using appropriate resampling technique
        
        Args:
            X: Feature matrix
            y: Target vector
            imbalance_info: Information about class imbalance
            
        Returns:
            Resampled X and y
        """
        if not imbalance_info['is_imbalanced']:
            return X, y
        
        n_samples = imbalance_info['total_samples']
        imbalance_ratio = imbalance_info['imbalance_ratio']
        minority_count = min(imbalance_info['class_distribution'].values())
        
        # Choose strategy based on dataset size and imbalance severity
        try:
            # For severe imbalance (>5:1) with enough data: SMOTE
            if imbalance_ratio > 5.0 and minority_count >= 6:
                # SMOTE requires at least 6 samples (k_neighbors=5 by default)
                k_neighbors = min(5, minority_count - 1)
                sampler = SMOTE(random_state=42, k_neighbors=k_neighbors)
                self.imbalance_strategy = f"SMOTE (k_neighbors={k_neighbors})"
                self._log_decision(
                    "class_imbalance_handling",
                    f"Applying SMOTE oversampling",
                    f"Severe imbalance ({imbalance_ratio:.1f}:1) - generating synthetic minority samples",
                    "info"
                )
            
            # For moderate imbalance (2.5-5:1): SMOTETomek (combination)
            elif imbalance_ratio > 2.5 and minority_count >= 6:
                k_neighbors = min(5, minority_count - 1)
                sampler = SMOTETomek(random_state=42, smote=SMOTE(k_neighbors=k_neighbors))
                self.imbalance_strategy = f"SMOTE-Tomek (k_neighbors={k_neighbors})"
                self._log_decision(
                    "class_imbalance_handling",
                    f"Applying SMOTE-Tomek hybrid sampling",
                    f"Moderate imbalance ({imbalance_ratio:.1f}:1) - oversampling minority + cleaning majority",
                    "info"
                )
            
            # For small datasets or low minority samples: Random oversampling
            elif minority_count < 6:
                sampler = RandomOverSampler(random_state=42)
                self.imbalance_strategy = "Random Oversampling"
                self._log_decision(
                    "class_imbalance_handling",
                    f"Applying random oversampling",
                    f"Limited minority samples ({minority_count}) - duplicating minority class samples",
                    "info"
                )
            
            # Fallback: Random oversampling
            else:
                sampler = RandomOverSampler(random_state=42)
                self.imbalance_strategy = "Random Oversampling"
                self._log_decision(
                    "class_imbalance_handling",
                    f"Applying random oversampling",
                    f"Imbalance ratio {imbalance_ratio:.1f}:1 - duplicating minority class samples",
                    "info"
                )
            
            # Apply resampling
            X_resampled, y_resampled = sampler.fit_resample(X, y)
            
            # Track resampled distribution
            unique, counts = np.unique(y_resampled, return_counts=True)
            self.resampled_class_distribution = {
                str(self.label_encoder.classes_[cls]): int(count)
                for cls, count in zip(unique, counts)
            }
            
            self.class_imbalance_handled = True
            
            # Log results
            original_total = len(y)
            resampled_total = len(y_resampled)
            new_ratio = max(counts) / min(counts)
            
            self._log_decision(
                "class_imbalance_result",
                f"✓ Class imbalance handled: {original_total} → {resampled_total} samples",
                f"New class ratio: {new_ratio:.1f}:1. Strategy: {self.imbalance_strategy}",
                "info"
            )
            
            return X_resampled, y_resampled
            
        except Exception as e:
            # If resampling fails, return original data and log warning
            self._log_decision(
                "class_imbalance_error",
                f"⚠️ Failed to apply resampling: {str(e)}",
                "Continuing with original imbalanced data. Consider manual handling.",
                "warning"
            )
            return X, y
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """Get comprehensive preprocessing summary"""
        total_removed = (
            len(self.id_columns) + 
            len(self.constant_columns) + 
            len(self.low_variance_columns) + 
            len(self.high_missing_columns) +
            len(getattr(self, 'high_cardinality_columns', []))
        )
        
        return {
            "task_type": self.task_type.value if self.task_type else None,
            "model_family": self.model_family.value,
            "target_column": self.target_column,
            "feature_counts": {
                "numerical": len(self.numerical_features),
                "categorical": len(self.categorical_features),
                "text": len(self.text_features),
                "datetime": len(self.datetime_features),
                "total_final": len(self.feature_names),
            },
            "text_features": {
                "columns": self.text_features,
                "vectorization": "TF-IDF" if self.text_features else None,
                "max_features": 5000 if self.text_features else None,
                "details": [
                    f"Text column detected: {col} - Applying TF-IDF vectorization with 5000 features"
                    for col in self.text_features
                ] if self.text_features else []
            },
            "total_columns_removed": total_removed,
            "columns_removed": {
                "id_columns": self.id_columns,
                "constant_columns": self.constant_columns,
                "constant_column_details": self.constant_column_details,
                "high_missing_columns": self.high_missing_columns,
                "low_variance_columns": self.low_variance_columns,
                "high_cardinality_columns": getattr(self, 'high_cardinality_columns', []),
            },
            "multicollinearity": self.preprocessing_metadata.get("multicollinearity_handled", {
                "pairs_detected": 0,
                "features_dropped": [],
                "features_combined": [],
                "model_family": self.model_family.value
            }),
            "decision_log": self.decision_log,
            "decisions_count": len(self.decision_log),
        }
