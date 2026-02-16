"""
Exploratory Data Analysis (EDA) Service
Performs comprehensive automated analysis of datasets
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List
from datetime import datetime
import json


class EDAService:
    """
    Service for automated exploratory data analysis
    """
    
    def __init__(self):
        self.analysis_results = {}
        self.steps_log = []
    
    def _infer_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert object columns that contain only numeric values to numeric type.
        
        This prevents numerical columns (like house prices) that were read as strings
        from being treated as categorical columns.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with corrected numeric types
        """
        df_copy = df.copy()
        
        for col in df_copy.columns:
            # Only process object/string columns
            if df_copy[col].dtype == 'object':
                try:
                    # Try to convert to numeric
                    # This will succeed if all non-null values are numeric
                    numeric_col = pd.to_numeric(df_copy[col], errors='coerce')
                    
                    # Check if conversion was successful for most values
                    # If > 80% of non-null values converted successfully, treat as numeric
                    non_null_count = df_copy[col].notna().sum()
                    if non_null_count > 0:
                        successful_conversions = numeric_col.notna().sum()
                        success_rate = successful_conversions / non_null_count
                        
                        if success_rate > 0.8:
                            df_copy[col] = numeric_col
                            print(f"   ✓ Converted '{col}' from object to numeric ({success_rate:.1%} success rate)")
                except:
                    # If conversion fails, keep as object
                    pass
        
        return df_copy
    
    def perform_eda(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform complete exploratory data analysis
        
        Args:
            df: Input DataFrame
        
        Returns:
            Dictionary with comprehensive EDA results
        """
        self.analysis_results = {}
        self.steps_log = []
        
        # Step 0: Infer numeric columns (convert object columns with numbers to numeric type)
        print("🔍 Inferring numeric columns from object types...")
        df = self._infer_numeric_columns(df)
        
        # Step 1: Load dataset
        self._log_step("Loading dataset", "completed")
        
        # Step 2: Basic information
        self._log_step("Analyzing dataset structure", "completed")
        self.analysis_results['basic_info'] = self._analyze_basic_info(df)
        
        # Step 3: Check missing values
        self._log_step("Checking missing values", "completed")
        self.analysis_results['missing_values'] = self._analyze_missing_values(df)
        
        # Step 4: Check duplicates
        self._log_step("Detecting duplicate rows", "completed")
        self.analysis_results['duplicates'] = self._analyze_duplicates(df)
        
        # Step 5: Analyze distributions
        self._log_step("Analyzing distributions", "completed")
        self.analysis_results['distributions'] = self._analyze_distributions(df)
        
        # Step 6: Detect outliers
        self._log_step("Detecting outliers", "completed")
        self.analysis_results['outliers'] = self._detect_outliers(df)
        
        # Step 7: Correlation analysis
        self._log_step("Computing correlations", "completed")
        self.analysis_results['correlations'] = self._analyze_correlations(df)
        
        # Step 8: Categorical analysis
        self._log_step("Analyzing categorical features", "completed")
        self.analysis_results['categorical_analysis'] = self._analyze_categorical(df)
        
        # Step 9: Data quality assessment
        self._log_step("Assessing data quality", "completed")
        self.analysis_results['data_quality'] = self._assess_data_quality(df)
        
        # Step 10: Generate recommendations
        self._log_step("Generating recommendations", "completed")
        self.analysis_results['recommendations'] = self._generate_recommendations(df)
        
        # Add steps log to results
        self.analysis_results['steps_log'] = self.steps_log
        self.analysis_results['completed_at'] = datetime.utcnow().isoformat()
        
        return self.analysis_results
    
    def _log_step(self, step_name: str, status: str):
        """Log an EDA step"""
        self.steps_log.append({
            'step': step_name,
            'status': status,
            'timestamp': datetime.utcnow().isoformat()
        })
    
    def _analyze_basic_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze basic dataset information"""
        # Identify numerical, categorical, and datetime columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        
        # Try to detect datetime in object columns
        for col in categorical_cols[:]:  # Use slice to iterate over copy
            if len(df[col].dropna()) > 0:
                sample = df[col].dropna().head(100)
                try:
                    pd.to_datetime(sample, errors='raise')
                    datetime_cols.append(col)
                    categorical_cols.remove(col)
                except:
                    pass
        
        return {
            'shape': {
                'rows': int(df.shape[0]),
                'columns': int(df.shape[1])
            },
            'column_types': {
                'numerical_count': len(numerical_cols),
                'categorical_count': len(categorical_cols),
                'datetime_count': len(datetime_cols),
                'numerical_columns': numerical_cols,
                'categorical_columns': categorical_cols,
                'datetime_columns': datetime_cols
            },
            'memory_usage_mb': float(df.memory_usage(deep=True).sum() / 1024 / 1024),
            'column_names': df.columns.tolist(),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()}
        }
    
    def _analyze_missing_values(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing values in dataset - for all columns"""
        if df.empty:
            return {
                'total_missing': 0,
                'missing_percentage': 0.0,
                'columns_with_missing': {},
                'columns_missing_count': 0
            }
        
        # Get missing values for ALL columns (not just numerical)
        missing_counts = df.isnull().sum()
        missing_percentages = (missing_counts / len(df) * 100).round(2)
        
        total_cells = len(df) * len(df.columns)
        
        return {
            'total_missing': int(missing_counts.sum()),
            'missing_percentage': float((missing_counts.sum() / total_cells * 100) if total_cells > 0 else 0),
            'columns_with_missing': {
                col: {
                    'count': int(count),
                    'percentage': float(missing_percentages[col])
                }
                for col, count in missing_counts.items() if count > 0
            },
            'columns_missing_count': int((missing_counts > 0).sum())
        }
    
    def _analyze_duplicates(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze duplicate rows with detailed information"""
        duplicate_mask = df.duplicated(keep=False)  # Mark all duplicates
        duplicate_count = df.duplicated().sum()  # Count excluding first occurrence
        
        duplicate_details = []
        
        if duplicate_count > 0:
            # Group duplicate rows and get their indices
            df_with_index = df.copy()
            df_with_index['_original_index'] = range(len(df))
            
            # Find groups of duplicates
            duplicate_groups = df_with_index[duplicate_mask].groupby(
                list(df.columns), dropna=False
            )['_original_index'].apply(list).reset_index(drop=True)
            
            # Limit to top 20 duplicate groups for performance
            for i, (_, group) in enumerate(duplicate_groups.items()):
                if i >= 20:  # Limit to 20 groups
                    break
                
                indices = group
                if len(indices) > 1:  # Only if actually duplicated
                    # Get the first occurrence as sample
                    first_idx = indices[0]
                    sample_row = df.iloc[first_idx].to_dict()
                    
                    # Convert values to JSON-serializable types
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
                        'row_indices': [int(idx) for idx in indices[:10]],  # Limit to first 10 indices
                        'sample_data': sample_row_clean,
                        'total_indices': len(indices)
                    })
        
        return {
            'total_duplicates': int(duplicate_count),
            'duplicate_percentage': float((duplicate_count / len(df) * 100) if len(df) > 0 else 0),
            'has_duplicates': bool(duplicate_count > 0),
            'duplicate_groups_count': len(duplicate_details),
            'duplicate_details': duplicate_details
        }
    
    def _analyze_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze numerical distributions"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        distributions = {}
        
        for col in numerical_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                distributions[col] = {
                    'mean': float(col_data.mean()) if not pd.isna(col_data.mean()) else None,
                    'median': float(col_data.median()) if not pd.isna(col_data.median()) else None,
                    'std': float(col_data.std()) if not pd.isna(col_data.std()) else None,
                    'min': float(col_data.min()) if not pd.isna(col_data.min()) else None,
                    'max': float(col_data.max()) if not pd.isna(col_data.max()) else None,
                    'q25': float(col_data.quantile(0.25)) if not pd.isna(col_data.quantile(0.25)) else None,
                    'q50': float(col_data.quantile(0.50)) if not pd.isna(col_data.quantile(0.50)) else None,
                    'q75': float(col_data.quantile(0.75)) if not pd.isna(col_data.quantile(0.75)) else None,
                    'skewness': float(col_data.skew()) if not pd.isna(col_data.skew()) else None,
                    'kurtosis': float(col_data.kurtosis()) if not pd.isna(col_data.kurtosis()) else None
                }
        
        return {
            'numerical_columns_count': len(numerical_cols),
            'distributions': distributions
        }
    
    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers using IQR method - excludes ID columns and includes distribution data"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        outliers = {}
        
        for col in numerical_cols:
            # Skip columns with 'id' in the name (case-insensitive)
            if 'id' in col.lower():
                continue
                
            col_data = df[col].dropna()
            if len(col_data) > 0:
                Q1 = col_data.quantile(0.25)
                Q3 = col_data.quantile(0.75)
                IQR = Q3 - Q1
                
                # Fallback to standard deviation method if IQR is too small
                # This happens when the middle 50% of data has very low variance
                if IQR < 1e-10:  # IQR essentially zero
                    mean = col_data.mean()
                    std = col_data.std()
                    # Use mean ± 3 standard deviations (99.7% rule for normal distribution)
                    lower_bound = mean - 3 * std
                    upper_bound = mean + 3 * std
                else:
                    lower_bound = Q1 - 2.5 * IQR
                    upper_bound = Q3 + 2.5 * IQR
                
                outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
                outlier_count = outlier_mask.sum()
                
                # Calculate distribution statistics for bell curve visualization
                mean = float(col_data.mean())
                std = float(col_data.std())
                
                # Get outlier values for visualization
                outlier_values = col_data[outlier_mask].tolist() if outlier_count > 0 else []
                inlier_values = col_data[~outlier_mask].tolist()
                
                outliers[col] = {
                    'count': int(outlier_count),
                    'percentage': float((outlier_count / len(col_data) * 100) if len(col_data) > 0 else 0),
                    'lower_bound': float(lower_bound) if not pd.isna(lower_bound) else None,
                    'upper_bound': float(upper_bound) if not pd.isna(upper_bound) else None,
                    'mean': mean,
                    'std': std,
                    'min': float(col_data.min()),
                    'max': float(col_data.max()),
                    'outlier_values': outlier_values[:100],  # Limit to 100 outliers for performance
                    'inlier_sample': inlier_values[:200] if len(inlier_values) > 200 else inlier_values  # Sample for histogram
                }
        
        return {
            'outliers_by_column': outliers,
            'total_columns_with_outliers': sum(1 for v in outliers.values() if v['count'] > 0)
        }
    
    def _analyze_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between numerical features"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numerical_cols) < 2:
            return {
                'correlation_matrix': {},
                'high_correlations': []
            }
        
        # Compute correlation matrix
        corr_matrix = df[numerical_cols].corr()
        
        # Find high correlations (> 0.7 or < -0.7)
        high_corr = []
        for i in range(len(numerical_cols)):
            for j in range(i + 1, len(numerical_cols)):
                corr_value = corr_matrix.iloc[i, j]
                if not pd.isna(corr_value) and abs(corr_value) > 0.7:
                    high_corr.append({
                        'feature1': numerical_cols[i],
                        'feature2': numerical_cols[j],
                        'correlation': float(corr_value)
                    })
        
        # Convert correlation matrix to dict
        corr_dict = {}
        for col in numerical_cols:
            corr_dict[col] = {
                other_col: float(val) if not pd.isna(val) else None
                for other_col, val in corr_matrix[col].items()
            }
        
        return {
            'correlation_matrix': corr_dict,
            'high_correlations': high_corr,
            'high_correlation_count': len(high_corr)
        }
    
    def _analyze_categorical(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze categorical features"""
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        categorical_analysis = {}
        
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            
            # Detect rare values (potential data quality issues)
            rare_values = self._detect_rare_categories(df[col])
            
            categorical_analysis[col] = {
                'unique_count': int(df[col].nunique()),
                'most_frequent': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                'top_5_values': {
                    str(k): int(v) for k, v in value_counts.head(5).items()
                },
                'cardinality': 'high' if df[col].nunique() > 50 else 'medium' if df[col].nunique() > 10 else 'low',
                'rare_values': rare_values  # Ratio-based rare value detection
            }
        
        return {
            'categorical_columns_count': len(categorical_cols),
            'categorical_features': categorical_analysis
        }
    
    def _detect_rare_categories(self, series: pd.Series, rare_threshold: float = 0.02) -> List[Dict[str, Any]]:
        """
        Detect rare categorical values using ratio-based approach
        
        Strategy: If a column has dominant categories (high %), rare categories (<2%) 
        are likely typos or data quality issues that should be removed.
        
        Example:
            - Category A: 490 rows (49%) ✓ Dominant
            - Category B: 490 rows (49%) ✓ Dominant  
            - 10 other categories: 20 rows total (2%) ⚠️ Likely typos/errors
        
        Args:
            series: Categorical column
            rare_threshold: Maximum percentage to consider "rare" (default 2% = 0.02)
        
        Returns:
            List of rare categories that may be data quality issues
        """
        # Get value counts (excluding NaN)
        value_counts = series.value_counts()
        total_count = len(series.dropna())
        
        if total_count == 0 or len(value_counts) == 0:
            return []
        
        # Calculate percentages
        value_percentages = (value_counts / total_count * 100).to_dict()
        
        # Find dominant categories (those with >10% occurrence)
        dominant_categories = [cat for cat, pct in value_percentages.items() if pct > 10]
        
        # Only flag rare values if there ARE dominant categories
        # This prevents flagging in evenly distributed columns
        if len(dominant_categories) < 2:
            return []
        
        # Find rare categories (< threshold %)
        rare_categories = []
        for category, count in value_counts.items():
            percentage = value_percentages[category]
            
            # Flag if rare AND there are dominant alternatives
            if percentage < (rare_threshold * 100):  # Convert to percentage
                rare_categories.append({
                    'rare_value': str(category),
                    'count': int(count),
                    'percentage': float(round(percentage, 2)),
                    'total_rows': total_count,
                    'dominant_categories': [str(cat) for cat in dominant_categories[:3]]  # Show top 3
                })
        
        # Sort by count (descending) - most impactful first
        rare_categories.sort(key=lambda x: x['count'], reverse=True)
        
        return rare_categories
    
    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Assess overall data quality with comprehensive metrics.
        
        Quality factors considered:
        - Completeness (missing values)
        - Uniqueness (duplicate rows)
        - Outliers (extreme values in numerical columns)
        - Skewness (distribution normality)
        - Rare values (potential typos/errors in categorical)
        - Variance (constant/low variance columns)
        - Multicollinearity (highly correlated features)
        """
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = df.isnull().sum().sum()
        duplicate_rows = df.duplicated().sum()
        
        # Start with perfect score
        quality_score = 100.0
        
        # 1. COMPLETENESS PENALTY (up to -25 points)
        # Missing data severely impacts ML models
        missing_ratio = missing_cells / total_cells if total_cells > 0 else 0
        completeness_penalty = missing_ratio * 25
        quality_score -= completeness_penalty
        
        # 2. UNIQUENESS PENALTY (up to -15 points)
        # Duplicate rows waste computation and can bias models
        duplicate_ratio = duplicate_rows / len(df) if len(df) > 0 else 0
        uniqueness_penalty = duplicate_ratio * 15
        quality_score -= uniqueness_penalty
        
        # 3. OUTLIER PENALTY (up to -20 points)
        # Outliers can severely skew model training
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        total_outliers = 0
        outlier_columns = 0
        
        for col in numerical_cols:
            if 'id' in col.lower():  # Skip ID columns
                continue
            
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR < 1e-10:
                mean = df[col].mean()
                std = df[col].std()
                outliers = ((df[col] < (mean - 3 * std)) | (df[col] > (mean + 3 * std))).sum()
            else:
                outliers = ((df[col] < (Q1 - 2.5 * IQR)) | (df[col] > (Q3 + 2.5 * IQR))).sum()
            
            if outliers > 0:
                total_outliers += outliers
                outlier_columns += 1
        
        if len(numerical_cols) > 0:
            outlier_ratio = total_outliers / (len(df) * len(numerical_cols))
            outlier_penalty = min(20, outlier_ratio * 100)  # Cap at 20 points
            quality_score -= outlier_penalty
        
        # 4. SKEWNESS PENALTY (up to -10 points)
        # Highly skewed data often needs transformation
        high_skew_count = 0
        for col in numerical_cols:
            if 'id' in col.lower():
                continue
            try:
                skewness = abs(df[col].skew())
                if skewness > 2:  # Highly skewed
                    high_skew_count += 1
            except:
                pass
        
        if len(numerical_cols) > 0:
            skew_ratio = high_skew_count / len(numerical_cols)
            skewness_penalty = skew_ratio * 10
            quality_score -= skewness_penalty
        
        # 5. RARE VALUES PENALTY (up to -15 points)
        # Rare categorical values may be typos or data entry errors
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        rare_value_issues = 0
        
        for col in categorical_cols:
            rare_values = self._detect_rare_categories(df[col])
            if rare_values:
                rare_value_issues += len(rare_values)
        
        if len(categorical_cols) > 0:
            rare_ratio = rare_value_issues / (len(categorical_cols) * 5)  # Normalize by expected rare values
            rare_penalty = min(15, rare_ratio * 15)
            quality_score -= rare_penalty
        
        # 6. LOW VARIANCE PENALTY (up to -10 points)
        # Constant or near-constant columns provide no information
        low_variance_cols = 0
        
        for col in numerical_cols:
            if df[col].nunique() <= 1:
                low_variance_cols += 1
            elif df[col].std() < 1e-6:  # Near-constant
                low_variance_cols += 1
        
        for col in categorical_cols:
            if df[col].nunique() <= 1:
                low_variance_cols += 1
        
        if len(df.columns) > 0:
            variance_ratio = low_variance_cols / len(df.columns)
            variance_penalty = variance_ratio * 10
            quality_score -= variance_penalty
        
        # 7. MULTICOLLINEARITY PENALTY (up to -10 points)
        # Highly correlated features are redundant
        if len(numerical_cols) > 1:
            try:
                corr_matrix = df[numerical_cols].corr().abs()
                # Get upper triangle (excluding diagonal)
                upper_triangle = corr_matrix.where(
                    np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
                )
                # Count pairs with correlation > 0.9
                high_corr_pairs = (upper_triangle > 0.9).sum().sum()
                
                max_possible_pairs = len(numerical_cols) * (len(numerical_cols) - 1) / 2
                if max_possible_pairs > 0:
                    corr_ratio = high_corr_pairs / max_possible_pairs
                    multicollinearity_penalty = corr_ratio * 10
                    quality_score -= multicollinearity_penalty
            except:
                pass
        
        # Ensure score is between 0 and 100
        quality_score = max(0, min(100, quality_score))
        
        # Calculate component scores
        completeness_score = (1 - missing_ratio) * 100
        uniqueness_score = (1 - duplicate_ratio) * 100
        
        # Determine assessment based on stricter thresholds
        if quality_score >= 90:
            assessment = 'excellent'
        elif quality_score >= 75:
            assessment = 'good'
        elif quality_score >= 60:
            assessment = 'fair'
        elif quality_score >= 40:
            assessment = 'poor'
        else:
            assessment = 'very poor'
        
        return {
            'quality_score': float(quality_score),
            'completeness': float(completeness_score),
            'uniqueness': float(uniqueness_score),
            'assessment': assessment
        }
    
    def _generate_recommendations(self, df: pd.DataFrame) -> List[str]:
        """Generate data preprocessing recommendations"""
        recommendations = []
        
        # Helper function to get clean column name
        def get_column_display_name(col):
            """Get a user-friendly column name"""
            if col.startswith('Unnamed:'):
                # Extract the column index from "Unnamed: X"
                try:
                    col_idx = int(col.split(':')[1].strip())
                    return f"Column {col_idx} (unnamed column)"
                except:
                    return col
            return col
        
        # Check missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            high_missing_cols = [col for col in df.columns if missing[col] / len(df) > 0.5]
            if high_missing_cols:
                display_names = [get_column_display_name(col) for col in high_missing_cols]
                recommendations.append(f"Consider removing columns with >50% missing values: {', '.join(display_names)}")
            else:
                recommendations.append("Apply missing value imputation for columns with missing data")
        
        # Check duplicates
        if df.duplicated().sum() > 0:
            recommendations.append(f"Remove {df.duplicated().sum()} duplicate rows to improve data quality")
        
        # Check constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_cols:
            display_names = [get_column_display_name(col) for col in constant_cols]
            recommendations.append(f"Remove constant columns (no variance): {', '.join(display_names)}")
        
        # Check high cardinality categorical
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        high_cardinality = [col for col in categorical_cols if df[col].nunique() > 50]
        if high_cardinality:
            display_names = [get_column_display_name(col) for col in high_cardinality]
            recommendations.append(f"Consider encoding or binning high cardinality features: {', '.join(display_names)}")
        
        # Check for rare categorical values (ratio-based detection)
        for col in categorical_cols:
            rare_values = self._detect_rare_categories(df[col])
            if rare_values:
                # Calculate total rare value count
                total_rare_count = sum(r['count'] for r in rare_values)
                total_rare_pct = sum(r['percentage'] for r in rare_values)
                
                # Flag ANY rare values (even just 1-2 typos)
                # This catches small typos like 2 rows in a 30k dataset
                if rare_values:  # Simply if there are any rare values
                    rare_examples = ', '.join([f"'{r['rare_value']}' ({r['count']} rows)" for r in rare_values[:3]])
                    if len(rare_values) > 3:
                        rare_examples += f" and {len(rare_values) - 3} more"
                    
                    recommendations.append(
                        f"⚠️ Data Quality: '{get_column_display_name(col)}' has {len(rare_values)} rare values "
                        f"({total_rare_count} rows, {total_rare_pct:.1f}% total) with <2% occurrence each: {rare_examples}. "
                        f"These may be typos or data entry errors. Consider removing these rows."
                    )
        
        # Check outliers - skip ID columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            # Skip columns with 'id' in the name (case-insensitive)
            if 'id' in col.lower():
                continue
                
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Fallback to standard deviation if IQR is too small
            if IQR < 1e-10:
                mean = df[col].mean()
                std = df[col].std()
                outliers = ((df[col] < (mean - 3 * std)) | (df[col] > (mean + 3 * std))).sum()
            else:
                outliers = ((df[col] < (Q1 - 2.5 * IQR)) | (df[col] > (Q3 + 2.5 * IQR))).sum()
            
            if outliers > len(df) * 0.05:  # More than 5% outliers
                recommendations.append(f"Consider handling outliers in '{get_column_display_name(col)}' ({outliers} detected)")
                break
        
        # Check correlations
        if len(numerical_cols) >= 2:
            corr_matrix = df[numerical_cols].corr()
            high_corr = []
            for i in range(len(numerical_cols)):
                for j in range(i + 1, len(numerical_cols)):
                    if abs(corr_matrix.iloc[i, j]) > 0.9:
                        high_corr.append((numerical_cols[i], numerical_cols[j]))
            if high_corr:
                recommendations.append(f"Consider removing highly correlated features to reduce multicollinearity")
        
        if not recommendations:
            recommendations.append("Dataset is in good shape! Consider proceeding with model training")
        
        return recommendations
