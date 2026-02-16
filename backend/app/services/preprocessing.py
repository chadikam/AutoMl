"""
Automated preprocessing service for datasets
Handles missing values, feature encoding, scaling, and pipeline creation
Enhanced with advanced detection and quality assessment features
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, OneHotEncoder, OrdinalEncoder, LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple, List, Dict, Any, Optional
import joblib
import os
from datetime import datetime
from app.config import settings


class PreprocessingService:
    """
    Service for automated dataset preprocessing
    """
    
    def __init__(self):
        self.preprocessing_pipeline = None
        self.feature_names = []
        self.numerical_features = []
        self.categorical_features = []
        self.datetime_features = []
        self.constant_columns = []
        self.high_missing_columns = []
        self.id_columns = []  # New: Track ID columns
        self.duplicate_count = 0
        self.outlier_info = {}
        self.preprocessing_log = []
        self.quality_score = 0.0
    
    def detect_and_remove_id_columns(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Detect and remove ID columns before preprocessing.
        ID columns typically: are named 'id' or end with '_id', have unique values for each row,
        or are sequential integers starting from 1.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Tuple of (cleaned DataFrame, list of dropped ID columns)
        """
        id_columns = []
        
        for col in df.columns:
            is_id = False
            col_lower = col.lower()
            
            # Check 1: Column name is exactly 'id' or ends with '_id' (not just contains 'id')
            if col_lower == 'id' or col_lower.endswith('_id') or col_lower.startswith('id_'):
                is_id = True
                self._log_preprocessing_step(f"Detected ID column by name: {col}")
            
            # Check 2: All values are unique (every row has unique value)
            elif df[col].nunique() == len(df) and len(df) > 0:
                is_id = True
                self._log_preprocessing_step(f"Detected ID column by uniqueness: {col}")
            
            # Check 3: Sequential integers starting from 1
            elif pd.api.types.is_numeric_dtype(df[col]):
                col_values = df[col].dropna()
                if len(col_values) > 0:
                    # Check if values are integers
                    if all(col_values == col_values.astype(int)):
                        # Check if sequential starting from 1
                        sorted_vals = sorted(col_values.astype(int))
                        expected_sequence = list(range(1, len(sorted_vals) + 1))
                        if sorted_vals == expected_sequence:
                            is_id = True
                            self._log_preprocessing_step(f"Detected ID column by sequence: {col}")
            
            if is_id:
                id_columns.append(col)
        
        # Remove ID columns
        df_cleaned = df.drop(columns=id_columns) if id_columns else df.copy()
        self.id_columns = id_columns
        
        if id_columns:
            self._log_preprocessing_step(f"Removed {len(id_columns)} ID columns: {', '.join(id_columns)}")
        
        return df_cleaned, id_columns
    
    def validate_transformation(self, X_transformed: np.ndarray, original_df: pd.DataFrame, 
                               feature_names: List[str]) -> Dict[str, Any]:
        """
        Validate that transformation was successful.
        
        Args:
            X_transformed: Transformed feature array
            original_df: Original DataFrame before transformation
            feature_names: List of feature names after transformation
        
        Returns:
            Dictionary with validation results
        
        Raises:
            ValueError: If validation fails
        """
        validation_results = {
            'passed': True,
            'checks': [],
            'warnings': [],
            'errors': []
        }
        
        # Check 1: All values are numeric (float)
        if not np.issubdtype(X_transformed.dtype, np.floating):
            validation_results['errors'].append(
                f"Data type is {X_transformed.dtype}, expected float64. "
                "OneHotEncoder may be producing boolean arrays."
            )
            validation_results['passed'] = False
        else:
            validation_results['checks'].append("✓ All values are numeric (float64)")
        
        # Check 2: No NaN values
        nan_count = np.isnan(X_transformed).sum()
        if nan_count > 0:
            validation_results['errors'].append(
                f"Found {nan_count} NaN values after transformation. "
                "Imputation may have failed."
            )
            validation_results['passed'] = False
        else:
            validation_results['checks'].append("✓ No NaN values present")
        
        # Check 3: No infinite values
        inf_count = np.isinf(X_transformed).sum()
        if inf_count > 0:
            validation_results['errors'].append(
                f"Found {inf_count} infinite values after transformation. "
                "Scaling may have produced invalid values."
            )
            validation_results['passed'] = False
        else:
            validation_results['checks'].append("✓ No infinite values present")
        
        # Check 4: Shape is correct
        if X_transformed.shape[1] != len(feature_names):
            validation_results['errors'].append(
                f"Feature count mismatch: transformed data has {X_transformed.shape[1]} columns "
                f"but {len(feature_names)} feature names provided"
            )
            validation_results['passed'] = False
        else:
            validation_results['checks'].append(
                f"✓ Feature count matches: {len(feature_names)} features"
            )
        
        # Check 5: No ID columns in feature names
        id_cols_remaining = [name for name in feature_names if 'id' in name.lower()]
        if id_cols_remaining:
            validation_results['warnings'].append(
                f"Potential ID columns still present: {', '.join(id_cols_remaining)}"
            )
        else:
            validation_results['checks'].append("✓ No ID columns in features")
        
        # Check 6: Verify scaling was applied (numerical features should have mean ≈ 0, std ≈ 1)
        if X_transformed.shape[1] > 0:
            means = np.mean(X_transformed, axis=0)
            stds = np.std(X_transformed, axis=0)
            
            # For StandardScaler, check if numerical features are scaled
            # (allowing some tolerance for one-hot encoded features)
            scaled_features = np.sum((np.abs(means) < 0.5) & (np.abs(stds - 1.0) < 0.5))
            if scaled_features > 0:
                validation_results['checks'].append(
                    f"✓ {scaled_features} features appear properly scaled"
                )
        
        # Check 7: All values in each row are valid numbers
        sample_row = X_transformed[0] if len(X_transformed) > 0 else []
        if len(sample_row) > 0:
            all_numeric = all(isinstance(val, (int, float, np.integer, np.floating)) 
                            for val in sample_row)
            if not all_numeric:
                validation_results['errors'].append(
                    "First row contains non-numeric values"
                )
                validation_results['passed'] = False
            else:
                validation_results['checks'].append("✓ All values are numeric types")
        
        # Log results
        self._log_preprocessing_step("=== Transformation Validation ===")
        for check in validation_results['checks']:
            self._log_preprocessing_step(check)
        for warning in validation_results['warnings']:
            self._log_preprocessing_step(f"⚠ {warning}")
        for error in validation_results['errors']:
            self._log_preprocessing_step(f"✗ {error}")
        
        # Raise error if validation failed
        if not validation_results['passed']:
            error_msg = "Transformation validation failed:\n" + "\n".join(validation_results['errors'])
            raise ValueError(error_msg)
        
        return validation_results
    
    def analyze_dataset(self, df: pd.DataFrame, threshold: float = 0.5) -> Dict[str, Any]:
        """
        Comprehensive dataset analysis with quality assessment
        
        Args:
            df: Input DataFrame
            threshold: Threshold for dropping high-missing columns (default: 0.5)
        
        Returns:
            Dictionary with comprehensive analysis results
        """
        self._log_preprocessing_step("Starting dataset analysis")
        
        analysis = {
            "rows": len(df),
            "columns": len(df.columns),
            "column_names": df.columns.tolist(),
            "column_types": {},
            "missing_values": {},
            "summary_statistics": {},
            "datetime_columns": 0,
            "numerical_columns": 0,
            "categorical_columns": 0,
            "duplicates": {},
            "outliers": {},
            "data_quality": {},
            "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024 / 1024)
        }
        
        # Detect column types (including datetime detection)
        numerical_cols = []
        categorical_cols = []
        datetime_cols = []
        
        for col in df.columns:
            # Check if datetime
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                analysis["column_types"][col] = "datetime"
                datetime_cols.append(col)
            elif pd.api.types.is_numeric_dtype(df[col]):
                analysis["column_types"][col] = "numerical"
                numerical_cols.append(col)
            else:
                analysis["column_types"][col] = "categorical"
                categorical_cols.append(col)
            
            # Count missing values for ALL columns
            missing_count = df[col].isna().sum()
            missing_pct = (missing_count / len(df) * 100) if len(df) > 0 else 0
            analysis["missing_values"][col] = {
                "count": int(missing_count),
                "percentage": float(missing_pct)
            }
        
        analysis["datetime_columns"] = len(datetime_cols)
        analysis["numerical_columns"] = len(numerical_cols)
        analysis["categorical_columns"] = len(categorical_cols)
        
        # Debug logging
        print(f"\n📊 Dataset Analysis Results:")
        print(f"   Total columns: {len(df.columns)}")
        print(f"   DateTime columns ({len(datetime_cols)}): {datetime_cols}")
        print(f"   Numerical columns ({len(numerical_cols)}): {numerical_cols[:5]}{'...' if len(numerical_cols) > 5 else ''}")
        print(f"   Categorical columns ({len(categorical_cols)}): {categorical_cols[:5]}{'...' if len(categorical_cols) > 5 else ''}")
        
        # Analyze duplicates
        duplicate_count = df.duplicated().sum()
        analysis["duplicates"] = {
            "total_duplicates": int(duplicate_count),
            "duplicate_percentage": float((duplicate_count / len(df) * 100) if len(df) > 0 else 0),
            "has_duplicates": bool(duplicate_count > 0)
        }
        self.duplicate_count = duplicate_count
        
        # Generate summary statistics for numerical columns
        if numerical_cols:
            stats = df[numerical_cols].describe().to_dict()
            stats = self._sanitize_stats(stats)
            analysis["summary_statistics"] = stats
        
        # Detect outliers for numerical columns (excluding ID columns)
        outliers_info = self._detect_outliers_detailed(df, numerical_cols)
        analysis["outliers"] = outliers_info
        self.outlier_info = outliers_info
        
        # Assess data quality
        quality_assessment = self._assess_data_quality(df)
        analysis["data_quality"] = quality_assessment
        self.quality_score = quality_assessment["quality_score"]
        
        self._log_preprocessing_step(f"Analysis complete - Quality Score: {self.quality_score:.1f}/100")
        
        return analysis
    
    def _sanitize_stats(self, stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Replace NaN and inf values with None for JSON serialization
        
        Args:
            stats: Dictionary with statistics
        
        Returns:
            Sanitized dictionary
        """
        sanitized = {}
        for col, col_stats in stats.items():
            sanitized[col] = {}
            for stat_name, value in col_stats.items():
                if pd.isna(value) or np.isinf(value):
                    sanitized[col][stat_name] = None
                else:
                    sanitized[col][stat_name] = float(value)
        return sanitized
    
    def _log_preprocessing_step(self, message: str):
        """Log a preprocessing step with timestamp"""
        self.preprocessing_log.append({
            'message': message,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    def _detect_outliers_detailed(self, df: pd.DataFrame, numerical_cols: List[str]) -> Dict[str, Any]:
        """
        Detect outliers using IQR method with detailed statistics
        
        Args:
            df: Input DataFrame
            numerical_cols: List of numerical column names
        
        Returns:
            Dictionary with outlier information
        """
        outliers = {}
        total_outliers = 0
        
        for col in numerical_cols:
            # Skip columns with 'id' in the name
            if 'id' in col.lower():
                continue
            
            col_data = df[col].dropna()
            if len(col_data) > 0:
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
                outlier_count = outlier_mask.sum()
                
                if outlier_count > 0:
                    total_outliers += outlier_count
                    outliers[col] = {
                        'count': int(outlier_count),
                        'percentage': float((outlier_count / len(col_data) * 100) if len(col_data) > 0 else 0),
                        'lower_bound': float(lower_bound) if not pd.isna(lower_bound) else None,
                        'upper_bound': float(upper_bound) if not pd.isna(upper_bound) else None,
                        'method': 'IQR'
                    }
        
        return {
            'outliers_by_column': outliers,
            'total_outliers': int(total_outliers),
            'columns_with_outliers': len(outliers)
        }
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Assess overall data quality with scoring
        
        Args:
            df: Input DataFrame
        
        Returns:
            Dictionary with quality metrics
        """
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        duplicate_rows = df.duplicated().sum()
        
        # Calculate quality score (0-100)
        missing_penalty = (missing_cells / total_cells) * 50 if total_cells > 0 else 0
        duplicate_penalty = (duplicate_rows / len(df)) * 30 if len(df) > 0 else 0
        
        quality_score = max(0, 100 - missing_penalty - duplicate_penalty)
        
        return {
            'quality_score': float(quality_score),
            'completeness': float((1 - missing_cells / total_cells) * 100) if total_cells > 0 else 100,
            'uniqueness': float((1 - duplicate_rows / len(df)) * 100) if len(df) > 0 else 100,
            'assessment': 'excellent' if quality_score >= 90 else 'good' if quality_score >= 70 else 'fair' if quality_score >= 50 else 'poor',
            'total_cells': int(total_cells),
            'missing_cells': int(missing_cells),
            'duplicate_rows': int(duplicate_rows)
        }
    
    def remove_duplicates(self, df: pd.DataFrame, keep: str = 'first') -> Tuple[pd.DataFrame, int]:
        """
        Remove duplicate rows from dataset
        
        Args:
            df: Input DataFrame
            keep: Which duplicates to keep ('first', 'last', False for removing all)
        
        Returns:
            Tuple of (cleaned DataFrame, number of duplicates removed)
        """
        initial_count = len(df)
        df_cleaned = df.drop_duplicates(keep=keep)
        duplicates_removed = initial_count - len(df_cleaned)
        
        if duplicates_removed > 0:
            self._log_preprocessing_step(f"Removed {duplicates_removed} duplicate rows")
        
        return df_cleaned, duplicates_removed
    
    def handle_outliers(
        self, 
        df: pd.DataFrame, 
        method: str = 'clip', 
        columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Handle outliers in numerical columns
        
        Args:
            df: Input DataFrame
            method: Method to handle outliers ('clip', 'remove', 'none')
            columns: Specific columns to process (None = all numerical)
        
        Returns:
            DataFrame with outliers handled
        """
        if method == 'none':
            return df
        
        df_copy = df.copy()
        numerical_cols = columns if columns else df.select_dtypes(include=[np.number]).columns.tolist()
        outliers_handled = 0
        
        for col in numerical_cols:
            # Skip ID columns
            if 'id' in col.lower():
                continue
            
            col_data = df_copy[col].dropna()
            if len(col_data) > 0:
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                if method == 'clip':
                    # Clip outliers to bounds
                    df_copy[col] = df_copy[col].clip(lower=lower_bound, upper=upper_bound)
                    outliers_handled += 1
                elif method == 'remove':
                    # Remove rows with outliers (careful with this!)
                    mask = (df_copy[col] >= lower_bound) & (df_copy[col] <= upper_bound)
                    df_copy = df_copy[mask]
        
        if outliers_handled > 0:
            self._log_preprocessing_step(f"Handled outliers in {outliers_handled} columns using '{method}' method")
        
        return df_copy
    
    def detect_columns_to_drop(self, df: pd.DataFrame, missing_threshold: float = 0.5) -> List[str]:
        """
        Detect columns that should be dropped (constant or high missing rate)
        
        Args:
            df: Input DataFrame
            missing_threshold: Threshold for dropping columns with too many missing values
        
        Returns:
            List of column names to drop
        """
        drop_columns = []
        
        for col in df.columns:
            # Check for constant columns (only one unique value)
            if df[col].nunique() <= 1:
                drop_columns.append(col)
                self.constant_columns.append(col)
                continue
            
            # Check for high missing rate
            missing_rate = df[col].isna().sum() / len(df)
            if missing_rate > missing_threshold:
                drop_columns.append(col)
                self.high_missing_columns.append(col)
        
        return drop_columns
    
    def separate_features(self, df: pd.DataFrame, target_column: str, drop_columns: List[str] = None):
        """
        Separate features into numerical, categorical, and datetime
        
        Args:
            df: Input DataFrame
            target_column: Name of the target column
            drop_columns: List of columns to drop
        """
        if drop_columns is None:
            drop_columns = []
        
        # Get feature columns (exclude target, drop columns, and ID columns)
        feature_columns = [
            col for col in df.columns 
            if col != target_column 
            and col not in drop_columns
            and col not in self.id_columns  # Exclude ID columns
        ]
        
        # Separate numerical, categorical, and datetime features
        self.numerical_features = []
        self.categorical_features = []
        self.datetime_features = []
        
        for col in feature_columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                self.datetime_features.append(col)
            elif pd.api.types.is_numeric_dtype(df[col]):
                # Check if it's a binary column (only 0 and 1)
                unique_vals = df[col].dropna().unique()
                if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                    # Treat binary numerical columns as categorical
                    self.categorical_features.append(col)
                    self._log_preprocessing_step(f"Treating binary column '{col}' as categorical")
                else:
                    self.numerical_features.append(col)
            else:
                # Check if it's a datetime string
                is_datetime = False
                non_null_values = df[col].dropna()
                if len(non_null_values) > 0:
                    sample = non_null_values.head(100)
                    try:
                        pd.to_datetime(sample, errors='raise')
                        is_datetime = True
                    except:
                        pass
                
                if is_datetime:
                    self.datetime_features.append(col)
                else:
                    self.categorical_features.append(col)
        
        # Log the categorization
        self._log_preprocessing_step(
            f"Feature categorization: {len(self.numerical_features)} numerical, "
            f"{len(self.categorical_features)} categorical, "
            f"{len(self.datetime_features)} datetime"
        )
        if self.numerical_features:
            self._log_preprocessing_step(f"  Numerical: {', '.join(self.numerical_features)}")
        if self.categorical_features:
            self._log_preprocessing_step(f"  Categorical: {', '.join(self.categorical_features)}")
        if self.datetime_features:
            self._log_preprocessing_step(f"  Datetime: {', '.join(self.datetime_features)}")
    
    def create_preprocessing_pipeline(
        self,
        numerical_strategy: str = "mean",
        categorical_strategy: str = "most_frequent",
        scaling_method: str = "standard",
        encoding_method: str = "onehot",
        handle_outliers: bool = False
    ) -> Pipeline:
        """
        Create sklearn preprocessing pipeline with advanced options
        
        Args:
            numerical_strategy: Strategy for numerical missing values (mean, median, knn)
            categorical_strategy: Strategy for categorical missing values (most_frequent, constant)
            scaling_method: Scaling method (standard, minmax, robust)
            encoding_method: Encoding method (onehot, ordinal, label)
            handle_outliers: Whether to use RobustScaler for outlier resistance
        
        Returns:
            Sklearn Pipeline object
        """
        transformers = []
        
        # Numerical features pipeline
        if self.numerical_features:
            # Choose imputer based on strategy
            if numerical_strategy == "knn":
                imputer = KNNImputer(n_neighbors=5)
            else:
                imputer = SimpleImputer(strategy=numerical_strategy)
            
            # Choose scaler based on method and outlier handling
            if handle_outliers or scaling_method == "robust":
                scaler = RobustScaler()
            elif scaling_method == "minmax":
                scaler = MinMaxScaler()
            else:
                scaler = StandardScaler()
            
            numerical_transformer = Pipeline(steps=[
                ('imputer', imputer),
                ('scaler', scaler)
            ])
            transformers.append(('num', numerical_transformer, self.numerical_features))
            
            self._log_preprocessing_step(f"Numerical pipeline: {numerical_strategy} imputation + {scaling_method} scaling")
        
        # Categorical features pipeline
        if self.categorical_features:
            # Choose encoder based on method
            if encoding_method == "ordinal":
                encoder = OrdinalEncoder(
                    handle_unknown='use_encoded_value', 
                    unknown_value=-1,
                    dtype=np.float64  # Ensure float output
                )
            else:  # onehot is default
                encoder = OneHotEncoder(
                    handle_unknown='ignore', 
                    sparse_output=False,
                    dtype=np.float64  # Force numeric output instead of boolean
                )
            
            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy=categorical_strategy)),
                ('encoder', encoder)
            ])
            transformers.append(('cat', categorical_transformer, self.categorical_features))
            
            self._log_preprocessing_step(f"Categorical pipeline: {categorical_strategy} imputation + {encoding_method} encoding")
        
        # Note: Datetime features are dropped by default in current implementation
        # Future enhancement: extract datetime features (year, month, day, etc.)
        if self.datetime_features:
            self._log_preprocessing_step(f"Warning: {len(self.datetime_features)} datetime columns will be dropped (not yet supported)")
        
        # Create column transformer
        preprocessor = ColumnTransformer(transformers=transformers)
        
        self.preprocessing_pipeline = preprocessor
        return preprocessor
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        target_column: str,
        numerical_strategy: str = "mean",
        categorical_strategy: str = "most_frequent",
        scaling_method: str = "standard",
        encoding_method: str = "onehot",
        missing_threshold: float = 0.5,
        remove_duplicates: bool = True,
        outlier_method: str = "none",
        handle_outliers_scaling: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """
        Complete preprocessing workflow: analyze, clean, transform
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            numerical_strategy: Strategy for numerical imputation (mean, median, knn)
            categorical_strategy: Strategy for categorical imputation
            scaling_method: Scaling method (standard, minmax, robust)
            encoding_method: Encoding method (onehot, ordinal)
            missing_threshold: Threshold for dropping columns
            remove_duplicates: Whether to remove duplicate rows
            outlier_method: How to handle outliers (clip, remove, none)
            handle_outliers_scaling: Use RobustScaler for outlier resistance
        
        Returns:
            Tuple of (X_transformed, y, preprocessing_info)
        """
        self.preprocessing_log = []
        self._log_preprocessing_step("=== Starting Preprocessing Workflow ===")
        
        initial_shape = df.shape
        preprocessing_info = {
            'initial_shape': initial_shape,
            'steps': [],
            'dropped_columns': [],
            'id_columns_removed': [],
            'feature_types': {},
            'final_shape': None,
            'quality_metrics': {},
            'validation': {}
        }
        
        # ==================== STEP 1: Remove ID columns (HIGHEST PRIORITY) ====================
        self._log_preprocessing_step("Step 1: Detecting and removing ID columns")
        df, id_cols = self.detect_and_remove_id_columns(df)
        if id_cols:
            preprocessing_info['id_columns_removed'] = id_cols
            preprocessing_info['steps'].append(f"Removed {len(id_cols)} ID columns: {', '.join(id_cols)}")
        else:
            preprocessing_info['steps'].append("No ID columns detected")
        
        # ==================== STEP 2: Remove duplicates ====================
        if remove_duplicates and df.duplicated().sum() > 0:
            self._log_preprocessing_step("Step 2: Removing duplicate rows")
            df, dup_count = self.remove_duplicates(df)
            preprocessing_info['steps'].append(f"Removed {dup_count} duplicate rows")
        else:
            self._log_preprocessing_step("Step 2: No duplicates to remove")
        
        # ==================== STEP 3: Detect and drop constant/high-missing columns ====================
        self._log_preprocessing_step("Step 3: Detecting columns to drop")
        drop_columns = self.detect_columns_to_drop(df, missing_threshold)
        if drop_columns:
            preprocessing_info['dropped_columns'] = drop_columns
            preprocessing_info['steps'].append(f"Dropped {len(drop_columns)} columns")
            
            if self.constant_columns:
                preprocessing_info['steps'].append(f"  - Constant columns: {', '.join(self.constant_columns)}")
            if self.high_missing_columns:
                preprocessing_info['steps'].append(f"  - High missing columns (>{missing_threshold*100}%): {', '.join(self.high_missing_columns)}")
            
            df = df.drop(columns=drop_columns)
        else:
            preprocessing_info['steps'].append("No columns need to be dropped")
        
        # ==================== STEP 4: Handle outliers (before scaling) ====================
        if outlier_method != "none":
            self._log_preprocessing_step(f"Step 4: Handling outliers using '{outlier_method}' method")
            df = self.handle_outliers(df, method=outlier_method)
            preprocessing_info['steps'].append(f"Handled outliers using '{outlier_method}' method")
        else:
            self._log_preprocessing_step("Step 4: Skipping outlier handling")
        
        # ==================== STEP 5: Separate features (excluding ID columns) ====================
        self._log_preprocessing_step("Step 5: Categorizing features")
        self.separate_features(df, target_column, drop_columns)
        preprocessing_info['feature_types'] = {
            'numerical': self.numerical_features,
            'categorical': self.categorical_features,
            'datetime': self.datetime_features
        }
        preprocessing_info['steps'].append(
            f"Categorized features: {len(self.numerical_features)} numerical, "
            f"{len(self.categorical_features)} categorical, "
            f"{len(self.datetime_features)} datetime"
        )
        
        # ==================== STEP 6: Extract X and y ====================
        self._log_preprocessing_step("Step 6: Extracting features and target")
        X = df.drop(columns=[target_column])
        y = df[target_column].values
        
        # ==================== STEP 7: Create preprocessing pipeline ====================
        self._log_preprocessing_step("Step 7: Creating preprocessing pipeline")
        self.create_preprocessing_pipeline(
            numerical_strategy=numerical_strategy,
            categorical_strategy=categorical_strategy,
            scaling_method=scaling_method,
            encoding_method=encoding_method,
            handle_outliers=handle_outliers_scaling
        )
        preprocessing_info['steps'].append(
            f"Created pipeline: {numerical_strategy} imputation, "
            f"{scaling_method} scaling, {encoding_method} encoding"
        )
        
        # ==================== STEP 8: Fit and transform features ====================
        self._log_preprocessing_step("Step 8: Fitting and transforming features")
        X_transformed = self.preprocessing_pipeline.fit_transform(X)
        
        # Ensure output is float64
        if X_transformed.dtype != np.float64:
            self._log_preprocessing_step(f"Converting output from {X_transformed.dtype} to float64")
            X_transformed = X_transformed.astype(np.float64)
        
        preprocessing_info['steps'].append(f"Transformed features - Shape: {X_transformed.shape}")
        
        # ==================== STEP 9: Get feature names ====================
        self._log_preprocessing_step("Step 9: Extracting feature names")
        self.feature_names = self._get_feature_names()
        preprocessing_info['feature_names'] = self.feature_names
        preprocessing_info['steps'].append(f"Generated {len(self.feature_names)} feature names")
        
        # ==================== STEP 10: Validate transformation (CRITICAL) ====================
        self._log_preprocessing_step("Step 10: Validating transformation")
        try:
            validation_results = self.validate_transformation(X_transformed, df, self.feature_names)
            preprocessing_info['validation'] = validation_results
            preprocessing_info['steps'].append("✓ Transformation validation passed")
        except ValueError as e:
            self._log_preprocessing_step(f"✗ Validation failed: {str(e)}")
            preprocessing_info['validation'] = {'passed': False, 'error': str(e)}
            raise
        
        # ==================== STEP 11: Final quality metrics ====================
        preprocessing_info['final_shape'] = X_transformed.shape
        preprocessing_info['quality_metrics'] = {
            'rows_before': initial_shape[0],
            'rows_after': X_transformed.shape[0],
            'columns_before': initial_shape[1],
            'columns_after': X_transformed.shape[1],
            'features_after_encoding': len(self.feature_names),
            'rows_removed': initial_shape[0] - X_transformed.shape[0],
            'columns_removed': len(drop_columns),
            'id_columns_removed': len(id_cols),
            'data_type': str(X_transformed.dtype),
            'has_nan': bool(np.isnan(X_transformed).any()),
            'has_inf': bool(np.isinf(X_transformed).any())
        }
        
        # Add preprocessing log
        preprocessing_info['log'] = self.preprocessing_log
        
        self._log_preprocessing_step(
            f"=== Preprocessing Complete - Final shape: {X_transformed.shape}, "
            f"dtype: {X_transformed.dtype} ==="
        )
        
        return X_transformed, y, preprocessing_info
    
    def transform(self, df: pd.DataFrame, target_column: str = None) -> np.ndarray:
        """
        Transform new data using fitted pipeline
        
        Args:
            df: Input DataFrame
            target_column: Optional target column to exclude
        
        Returns:
            Transformed feature array
        """
        if self.preprocessing_pipeline is None:
            raise ValueError("Pipeline not fitted. Call fit_transform first.")
        
        if target_column and target_column in df.columns:
            X = df.drop(columns=[target_column])
        else:
            X = df
        
        return self.preprocessing_pipeline.transform(X)
    
    def _get_feature_names(self) -> List[str]:
        """
        Extract feature names after transformation with proper ordering
        
        Returns:
            List of feature names matching transformed array columns
        """
        feature_names = []
        
        # Iterate through transformers in order
        for name, transformer, columns in self.preprocessing_pipeline.transformers_:
            if name == 'num':
                # Numerical features keep their original names
                feature_names.extend(columns)
                self._log_preprocessing_step(f"Added {len(columns)} numerical feature names")
                
            elif name == 'cat':
                # For categorical features, get encoded names
                encoder = transformer.named_steps['encoder']
                
                if isinstance(encoder, OneHotEncoder):
                    # OneHotEncoder creates multiple columns per original feature
                    cat_features = encoder.get_feature_names_out(columns)
                    feature_names.extend(cat_features.tolist())
                    self._log_preprocessing_step(
                        f"Added {len(cat_features)} one-hot encoded feature names "
                        f"from {len(columns)} categorical features"
                    )
                else:
                    # OrdinalEncoder keeps original column names
                    feature_names.extend(columns)
                    self._log_preprocessing_step(f"Added {len(columns)} ordinal encoded feature names")
        
        # Validation: Ensure we have the right number of features
        self._log_preprocessing_step(f"Total feature names generated: {len(feature_names)}")
        
        return feature_names
    
    def save_pipeline(self, filepath: str):
        """
        Save preprocessing pipeline to disk
        
        Args:
            filepath: Path to save the pipeline
        """
        if self.preprocessing_pipeline is None:
            raise ValueError("No pipeline to save. Fit the pipeline first.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        pipeline_data = {
            'pipeline': self.preprocessing_pipeline,
            'feature_names': self.feature_names,
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features
        }
        
        joblib.dump(pipeline_data, filepath)
    
    def load_pipeline(self, filepath: str):
        """
        Load preprocessing pipeline from disk
        
        Args:
            filepath: Path to the saved pipeline
        """
        pipeline_data = joblib.load(filepath)
        self.preprocessing_pipeline = pipeline_data['pipeline']
        self.feature_names = pipeline_data['feature_names']
        self.numerical_features = pipeline_data['numerical_features']
        self.categorical_features = pipeline_data['categorical_features']
    
    def generate_preprocessing_recommendations(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Generate preprocessing recommendations based on dataset analysis
        
        Args:
            df: Input DataFrame
        
        Returns:
            List of recommendation dictionaries
        """
        recommendations = []
        
        # Helper function to get clean column name
        def get_column_display_name(col):
            if col.startswith('Unnamed:'):
                try:
                    col_idx = int(col.split(':')[1].strip())
                    return f"Column {col_idx} (unnamed)"
                except:
                    return col
            return col
        
        # Check missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            high_missing_cols = [col for col in df.columns if missing[col] / len(df) > 0.5]
            if high_missing_cols:
                display_names = [get_column_display_name(col) for col in high_missing_cols]
                recommendations.append({
                    'type': 'warning',
                    'category': 'missing_values',
                    'message': f"Consider removing columns with >50% missing values: {', '.join(display_names)}",
                    'action': 'drop_columns',
                    'affected_columns': high_missing_cols
                })
            
            moderate_missing_cols = [col for col in df.columns if 0 < missing[col] / len(df) <= 0.5]
            if moderate_missing_cols:
                recommendations.append({
                    'type': 'info',
                    'category': 'missing_values',
                    'message': f"Apply imputation for {len(moderate_missing_cols)} columns with missing data",
                    'action': 'impute',
                    'affected_columns': moderate_missing_cols
                })
        
        # Check duplicates
        duplicate_count = df.duplicated().sum()
        if duplicate_count > 0:
            recommendations.append({
                'type': 'warning',
                'category': 'duplicates',
                'message': f"Remove {duplicate_count} duplicate rows to improve data quality",
                'action': 'remove_duplicates',
                'count': int(duplicate_count)
            })
        
        # Check constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_cols:
            display_names = [get_column_display_name(col) for col in constant_cols]
            recommendations.append({
                'type': 'warning',
                'category': 'constant_features',
                'message': f"Remove constant columns (no variance): {', '.join(display_names)}",
                'action': 'drop_columns',
                'affected_columns': constant_cols
            })
        
        # Check high cardinality categorical
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        high_cardinality = [col for col in categorical_cols if df[col].nunique() > 50]
        if high_cardinality:
            display_names = [get_column_display_name(col) for col in high_cardinality]
            recommendations.append({
                'type': 'info',
                'category': 'encoding',
                'message': f"Consider encoding or binning high cardinality features: {', '.join(display_names)}",
                'action': 'encode_or_bin',
                'affected_columns': high_cardinality
            })
        
        # Check outliers - skip ID columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        outlier_cols = []
        for col in numerical_cols:
            if 'id' in col.lower():
                continue
            
            col_data = df[col].dropna()
            if len(col_data) > 0:
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                outliers = ((col_data < (Q1 - 1.5 * IQR)) | (col_data > (Q3 + 1.5 * IQR))).sum()
                
                if outliers > len(df) * 0.05:  # More than 5% outliers
                    outlier_cols.append((col, outliers))
        
        if outlier_cols:
            total_outliers = sum(count for _, count in outlier_cols)
            recommendations.append({
                'type': 'info',
                'category': 'outliers',
                'message': f"Consider handling outliers in {len(outlier_cols)} columns ({total_outliers} total outliers detected)",
                'action': 'handle_outliers',
                'affected_columns': [col for col, _ in outlier_cols],
                'methods': ['clip', 'remove', 'robust_scaling']
            })
        
        # Check for highly correlated features
        if len(numerical_cols) >= 2:
            corr_matrix = df[numerical_cols].corr()
            high_corr_pairs = []
            for i in range(len(numerical_cols)):
                for j in range(i + 1, len(numerical_cols)):
                    if abs(corr_matrix.iloc[i, j]) > 0.9:
                        high_corr_pairs.append((numerical_cols[i], numerical_cols[j]))
            
            if high_corr_pairs:
                recommendations.append({
                    'type': 'info',
                    'category': 'correlation',
                    'message': f"Consider removing highly correlated features to reduce multicollinearity ({len(high_corr_pairs)} pairs found)",
                    'action': 'remove_correlated',
                    'pairs': high_corr_pairs[:5]  # Show only first 5 pairs
                })
        
        # Add success message if no issues
        if not recommendations:
            recommendations.append({
                'type': 'success',
                'category': 'quality',
                'message': "Dataset is in good shape! Ready for preprocessing and model training",
                'action': 'none'
            })
        
        return recommendations
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the preprocessing configuration and results
        
        Returns:
            Dictionary with preprocessing summary
        """
        return {
            'feature_types': {
                'numerical': len(self.numerical_features),
                'categorical': len(self.categorical_features),
                'datetime': len(self.datetime_features)
            },
            'feature_names': {
                'numerical': self.numerical_features,
                'categorical': self.categorical_features,
                'datetime': self.datetime_features
            },
            'dropped_columns': {
                'constant': self.constant_columns,
                'high_missing': self.high_missing_columns
            },
            'final_features': self.feature_names,
            'total_features': len(self.feature_names),
            'quality_score': self.quality_score,
            'preprocessing_log': self.preprocessing_log
        }

