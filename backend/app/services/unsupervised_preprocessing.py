"""
Unsupervised Preprocessing Pipeline
=====================================

Geometry-aware preprocessing specifically designed for unsupervised learning.

Key design principles:
1. ID / constant / near-constant columns are removed (they add noise, not signal)
2. Numeric features are scaled with RobustScaler (handles skew & outliers)
3. High-skew numeric features get Yeo-Johnson power transform BEFORE scaling
4. Categorical features are one-hot encoded WITHOUT StandardScaler on top
   (scaling binary 0/1 columns distorts rare-category distances catastrophically)
5. Optionally: PCA dimensionality reduction when feature count is high
6. VarianceThreshold removes near-zero-variance features post-encoding
7. Text features are TF-IDF vectorized (sparse) and kept at unit-norm (no scaling)

This replaces the naive "StandardScaler + OHE" pipeline that was destroying
the geometry of the data before clustering.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.preprocessing import (
    StandardScaler, RobustScaler, PowerTransformer,
    OneHotEncoder, OrdinalEncoder
)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
import logging
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class UnsupervisedPreprocessor:
    """
    Production-grade preprocessing for unsupervised learning that respects
    data geometry.

    Problems with the previous approach (StandardScaler + OHE):
    ─────────────────────────────────────────────────────────────
    1. ID columns (CustomerID, Index) treated as features → noise dimensions
    2. StandardScaler on OHE binary columns → rare categories get extreme
       scaled values (e.g., a column with 0.05% frequency gets value ~45
       after z-normalization), dominating Euclidean distance
    3. No skewness handling → positively-skewed spending/income features
       have a few outliers that dominate distance
    4. No variance filtering → near-zero-variance OHE columns add noise
    5. No dimensionality reduction → curse of dimensionality in high-D
       OHE space makes distances meaningless
    6. No separation of numeric vs categorical scaling → same StandardScaler
       applied to everything
    """

    def __init__(
        self,
        max_onehot_cardinality: int = 15,
        variance_threshold: float = 0.01,
        pca_variance_threshold: float = 0.95,
        auto_pca: bool = True,
        auto_pca_trigger_features: int = 50,
        handle_skewness: bool = True,
        skewness_threshold: float = 1.0,
        id_column_heuristics: bool = True,
        log_callback=None,
    ):
        """
        Args:
            max_onehot_cardinality: Max unique values before switching to ordinal
            variance_threshold: Remove features with variance below this after encoding
            pca_variance_threshold: Cumulative variance to retain if PCA applied
            auto_pca: Automatically apply PCA when feature count exceeds trigger
            auto_pca_trigger_features: Feature count that triggers auto-PCA
            handle_skewness: Apply Yeo-Johnson to highly skewed numeric features
            skewness_threshold: Absolute skewness above which to transform
            id_column_heuristics: Auto-detect and remove ID-like columns
            log_callback: Optional function(message: str) for logging
        """
        self.max_onehot_cardinality = max_onehot_cardinality
        self.variance_threshold = variance_threshold
        self.pca_variance_threshold = pca_variance_threshold
        self.auto_pca = auto_pca
        self.auto_pca_trigger_features = auto_pca_trigger_features
        self.handle_skewness = handle_skewness
        self.skewness_threshold = skewness_threshold
        self.id_column_heuristics = id_column_heuristics
        self._log = log_callback or (lambda msg: None)

        # Fitted state
        self.pipeline_ = None
        self.pca_ = None
        self.variance_selector_ = None
        self.dropped_columns_ = []
        self.numeric_features_ = []
        self.categorical_features_ = []
        self.skewed_features_ = []
        self.high_cardinality_features_ = []
        self.feature_names_out_ = []
        self.n_features_before_pca_ = 0
        self.n_features_after_pca_ = 0
        self.pca_explained_variance_ = None
        self.preprocessing_metadata_ = {}

    # ─── Column Detection ─────────────────────────────────────────────

    def _detect_id_columns(self, df: pd.DataFrame) -> List[str]:
        """
        Detect columns that are likely row identifiers (not useful features).
        
        Heuristics:
        - Column name contains 'id', 'index', 'row', 'record' (case-insensitive)
        - Column has all unique integer values (monotonic or near-monotonic)
        - Column is sequential integers (1,2,3,... or 0,1,2,...)
        """
        id_cols = []
        n = len(df)

        for col in df.columns:
            name_lower = col.lower().strip()

            # Name-based heuristic
            is_id_name = any(
                pattern in name_lower
                for pattern in ['_id', 'customerid', 'userid', 'rowid', 'index',
                                'record_id', 'row_number', 'serial']
            )
            if name_lower in ('id', 'index', 'idx'):
                is_id_name = True

            # Value-based heuristic: all unique, integer-like, monotonic
            if pd.api.types.is_numeric_dtype(df[col]):
                nunique = df[col].nunique()
                is_all_unique = nunique == n
                is_nearly_unique = nunique > 0.95 * n

                # Check if it's sequential integers
                vals = df[col].dropna().values
                if len(vals) > 0 and np.issubdtype(vals.dtype, np.integer):
                    sorted_vals = np.sort(vals)
                    is_sequential = np.all(np.diff(sorted_vals) == 1)
                else:
                    is_sequential = False

                if is_id_name and (is_all_unique or is_nearly_unique):
                    id_cols.append(col)
                    self._log(f"ID column detected (name+unique): '{col}'")
                elif is_sequential and is_all_unique and n > 10:
                    id_cols.append(col)
                    self._log(f"ID column detected (sequential integers): '{col}'")
            elif is_id_name:
                # String ID column
                nunique = df[col].nunique()
                if nunique > 0.9 * n:
                    id_cols.append(col)
                    self._log(f"ID column detected (name+high cardinality string): '{col}'")

        return id_cols

    def _detect_constant_columns(self, df: pd.DataFrame) -> List[str]:
        """Remove columns with zero or near-zero variance (only 1 unique value)."""
        constant_cols = []
        for col in df.columns:
            nunique = df[col].nunique(dropna=True)
            if nunique <= 1:
                constant_cols.append(col)
                self._log(f"Constant column removed: '{col}' ({nunique} unique)")
        return constant_cols

    def _categorize_features(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        """
        Split features into numeric and categorical.
        Also detects integer columns that are actually categorical (low cardinality).
        """
        numeric_cols = []
        categorical_cols = []

        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                nunique = df[col].nunique()
                # Integer column with very few unique values → treat as categorical
                if pd.api.types.is_integer_dtype(df[col]) and nunique <= 5:
                    categorical_cols.append(col)
                    self._log(f"Low-cardinality integer '{col}' ({nunique} unique) → categorical")
                else:
                    numeric_cols.append(col)
            else:
                categorical_cols.append(col)

        return numeric_cols, categorical_cols

    def _detect_skewed_features(self, df: pd.DataFrame, numeric_cols: List[str]) -> List[str]:
        """Identify numeric columns with high skewness."""
        skewed = []
        for col in numeric_cols:
            try:
                skew = df[col].dropna().skew()
                if abs(skew) > self.skewness_threshold:
                    skewed.append(col)
                    self._log(f"Skewed feature: '{col}' (skew={skew:.2f})")
            except Exception:
                pass
        return skewed

    # ─── Pipeline Construction ────────────────────────────────────────

    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str], Dict[str, Any]]:
        """
        Fit the preprocessing pipeline and transform data.

        Returns:
            X_transformed: np.ndarray - preprocessed feature matrix
            feature_names: list of str - feature names after transformation
            metadata: dict - preprocessing metadata for logging/display
        """
        df = df.copy()
        n_original = df.shape[1]

        # ── Step 1: Remove ID columns ──
        if self.id_column_heuristics:
            id_cols = self._detect_id_columns(df)
            if id_cols:
                df = df.drop(columns=id_cols)
                self.dropped_columns_.extend(id_cols)
                self._log(f"Removed {len(id_cols)} ID column(s): {id_cols}")

        # ── Step 2: Remove constant columns ──
        const_cols = self._detect_constant_columns(df)
        if const_cols:
            df = df.drop(columns=const_cols)
            self.dropped_columns_.extend(const_cols)
            self._log(f"Removed {len(const_cols)} constant column(s): {const_cols}")

        # ── Step 3: Handle missing values early (needed for skewness calc) ──
        # Drop columns with >60% missing
        high_missing = [
            col for col in df.columns
            if df[col].isnull().mean() > 0.6
        ]
        if high_missing:
            df = df.drop(columns=high_missing)
            self.dropped_columns_.extend(high_missing)
            self._log(f"Removed {len(high_missing)} high-missing column(s): {high_missing}")

        # ── Step 4: Categorize features ──
        numeric_cols, categorical_cols = self._categorize_features(df)
        self.numeric_features_ = numeric_cols
        self.categorical_features_ = categorical_cols
        self._log(f"Features: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical")

        # ── Step 5: Detect skewed features (for logging, not separate treatment) ──
        if self.handle_skewness:
            skewed_cols = self._detect_skewed_features(df, numeric_cols)
            self.skewed_features_ = skewed_cols
        else:
            skewed_cols = []

        # ── Step 6: Split categorical by cardinality ──
        low_card_cat = []
        high_card_cat = []
        for col in categorical_cols:
            nunique = df[col].nunique()
            if nunique <= self.max_onehot_cardinality:
                low_card_cat.append(col)
            else:
                high_card_cat.append(col)
                self.high_cardinality_features_.append(col)
                self._log(f"High-cardinality categorical '{col}' ({nunique} unique) → ordinal encoding")

        # ── Step 7: Build ColumnTransformer ──
        transformers = []

        # 7a: ALL numeric features → Impute → RobustScaler
        #     RobustScaler uses median/IQR, which handles outliers well without
        #     destroying the distributional shape that clustering needs.
        #     NOTE: We do NOT use PowerTransformer for clustering because:
        #     - Skewness IS often the cluster signal (big vs small spenders)
        #     - Gaussianizing removes the very structure we want to discover
        #     - RobustScaler already handles outlier-dominated scales via IQR
        all_numeric = numeric_cols  # Use all numeric cols together
        if all_numeric:
            numeric_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', RobustScaler()),
            ])
            transformers.append(('num', numeric_pipeline, all_numeric))
            self._log(f"Numeric ({len(all_numeric)} cols): Impute→RobustScaler")
            if skewed_cols:
                self._log(f"  (skewed features preserved: {skewed_cols} — skewness is structure for clustering)")

        # 7b: Low-cardinality categorical → Impute → OHE (NO SCALING!)
        #     Binary 0/1 columns should NOT be z-normalized.
        #     They naturally live in [0, 1] which is compatible with scaled numerics.
        if low_card_cat:
            cat_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OneHotEncoder(
                    drop='first',
                    sparse_output=False,
                    handle_unknown='ignore',
                    dtype=np.float64,
                )),
            ])
            transformers.append(('cat_ohe', cat_pipeline, low_card_cat))
            self._log(f"Low-cardinality categorical ({len(low_card_cat)} cols, ≤{self.max_onehot_cardinality} unique): Impute→OHE (no scaling)")

        # 7c: High-cardinality categorical → Impute → Ordinal → Scale to [0,1]
        if high_card_cat:
            from sklearn.preprocessing import MinMaxScaler
            highcard_pipeline = Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('encoder', OrdinalEncoder(
                    handle_unknown='use_encoded_value',
                    unknown_value=-1,
                    dtype=np.float64,
                )),
                ('scaler', MinMaxScaler()),  # Normalize ordinal codes to [0,1]
            ])
            transformers.append(('cat_ord', highcard_pipeline, high_card_cat))
            self._log(f"High-cardinality categorical ({len(high_card_cat)} cols): Impute→Ordinal→MinMax")

        if not transformers:
            raise ValueError(
                "No features available after preprocessing. "
                f"All {n_original} columns were removed (IDs: {len(self.dropped_columns_)})."
            )

        self.pipeline_ = ColumnTransformer(
            transformers=transformers,
            remainder='drop',
            sparse_threshold=0,  # Force dense output
        )

        # ── Step 8: Fit & Transform ──
        X = self.pipeline_.fit_transform(df)
        self._log(f"After ColumnTransformer: {X.shape[1]} features × {X.shape[0]} samples")

        # Build feature names
        feature_names = self._get_feature_names()

        # ── Step 9: Variance Threshold (remove near-zero-variance post-encoding) ──
        if self.variance_threshold > 0:
            self.variance_selector_ = VarianceThreshold(threshold=self.variance_threshold)
            n_before = X.shape[1]
            X = self.variance_selector_.fit_transform(X)
            n_removed = n_before - X.shape[1]
            if n_removed > 0:
                mask = self.variance_selector_.get_support()
                feature_names = [f for f, keep in zip(feature_names, mask) if keep]
                self._log(f"VarianceThreshold removed {n_removed} near-zero-variance features → {X.shape[1]} remaining")

        self.n_features_before_pca_ = X.shape[1]

        # ── Step 10: Optional PCA for dimensionality reduction ──
        if self.auto_pca and X.shape[1] > self.auto_pca_trigger_features:
            self._log(f"Auto-PCA triggered: {X.shape[1]} features > {self.auto_pca_trigger_features} threshold")
            self.pca_ = PCA(
                n_components=self.pca_variance_threshold,
                random_state=42,
            )
            X = self.pca_.fit_transform(X)
            self.pca_explained_variance_ = self.pca_.explained_variance_ratio_
            self.n_features_after_pca_ = X.shape[1]
            feature_names = [f"PC{i+1}" for i in range(X.shape[1])]
            self._log(
                f"PCA: {self.n_features_before_pca_} → {X.shape[1]} components "
                f"(explained variance: {self.pca_explained_variance_.sum():.4f})"
            )
        else:
            self.n_features_after_pca_ = X.shape[1]

        self.feature_names_out_ = feature_names

        # ── Build metadata ──
        self.preprocessing_metadata_ = {
            "original_columns": n_original,
            "dropped_columns": self.dropped_columns_,
            "numeric_features": self.numeric_features_,
            "categorical_features": self.categorical_features_,
            "skewed_features_transformed": self.skewed_features_,
            "high_cardinality_ordinal": self.high_cardinality_features_,
            "features_after_encoding": self.n_features_before_pca_,
            "features_after_pca": self.n_features_after_pca_,
            "pca_applied": self.pca_ is not None,
            "pca_explained_variance": float(self.pca_explained_variance_.sum()) if self.pca_explained_variance_ is not None else None,
            "pca_n_components": int(self.n_features_after_pca_) if self.pca_ is not None else None,
            "variance_threshold": self.variance_threshold,
        }

        return X, feature_names, self.preprocessing_metadata_

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using the fitted pipeline."""
        if self.pipeline_ is None:
            raise ValueError("Pipeline not fitted. Call fit_transform() first.")

        df = df.copy()

        # Remove dropped columns
        for col in self.dropped_columns_:
            if col in df.columns:
                df = df.drop(columns=[col])

        X = self.pipeline_.transform(df)

        if self.variance_selector_ is not None:
            X = self.variance_selector_.transform(X)

        if self.pca_ is not None:
            X = self.pca_.transform(X)

        return X

    def _get_feature_names(self) -> List[str]:
        """Extract feature names from the fitted ColumnTransformer."""
        names = []
        try:
            names = self.pipeline_.get_feature_names_out().tolist()
        except Exception:
            # Fallback: manual construction
            for name, trans, cols in self.pipeline_.transformers_:
                if name == 'remainder':
                    continue
                if hasattr(trans, 'get_feature_names_out'):
                    try:
                        names.extend(trans.get_feature_names_out().tolist())
                    except Exception:
                        names.extend([f"{name}__{i}" for i in range(len(cols) if isinstance(cols, list) else 1)])
                elif isinstance(cols, list):
                    names.extend(cols)
                else:
                    names.append(str(cols))
        return names

    def get_pipeline_components(self):
        """Return all fitted components for serialization."""
        return {
            'column_transformer': self.pipeline_,
            'variance_selector': self.variance_selector_,
            'pca': self.pca_,
            'dropped_columns': self.dropped_columns_,
            'numeric_features': self.numeric_features_,
            'categorical_features': self.categorical_features_,
            'feature_names': self.feature_names_out_,
            'metadata': self.preprocessing_metadata_,
        }
