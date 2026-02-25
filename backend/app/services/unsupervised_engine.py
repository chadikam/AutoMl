"""
Unsupervised Learning Engine with Optuna Hyperparameter Optimization
=====================================================================

Implements AutoML for unsupervised learning tasks:
1. Clustering (KMeans, MiniBatchKMeans, DBSCAN, Agglomerative, HDBSCAN)
2. Dimensionality Reduction (PCA, TruncatedSVD)
3. Anomaly Detection (IsolationForest, LocalOutlierFactor)

Evaluation uses internal metrics (no target labels required):
- Silhouette Score, Davies-Bouldin, Calinski-Harabasz for clustering
- Explained Variance Ratio for dimensionality reduction
- Anomaly score consistency for anomaly detection

Integrates with TrainingManager for status tracking, cancellation, and skip.
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
import logging
import gc
import os
import joblib
from datetime import datetime

warnings.filterwarnings('ignore')

# Scikit-learn clustering
from sklearn.cluster import (
    KMeans, MiniBatchKMeans, DBSCAN, AgglomerativeClustering
)

# Scikit-learn dimensionality reduction
from sklearn.decomposition import PCA, TruncatedSVD

# Scikit-learn anomaly detection
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Metrics
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score
)
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

# Optional: HDBSCAN
try:
    from hdbscan import HDBSCAN
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False

# Optional: UMAP
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False


class UnsupervisedTaskType(str, Enum):
    """Unsupervised sub-task types"""
    CLUSTERING = "clustering"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
    ANOMALY_DETECTION = "anomaly_detection"


@dataclass
class UnsupervisedModelResult:
    """Results for a single unsupervised model"""
    model_name: str
    model_type: str
    task_subtype: UnsupervisedTaskType

    # Trained model
    best_model: Any
    best_params: Dict[str, Any]

    # Primary score (unified, higher = better)
    primary_score: float

    # Detailed metrics (task-specific)
    detailed_metrics: Dict[str, Any] = field(default_factory=dict)

    # Optuna
    n_trials: int = 0
    best_trial_number: int = 0
    optimization_time: float = 0.0

    # Stability
    stability_status: str = "stable"
    stability_reason: Optional[str] = None

    # Rejection
    rejected: bool = False
    rejection_reason: Optional[str] = None

    # Cluster labels / predictions
    labels: Optional[np.ndarray] = None

    # For dimensionality reduction
    transformed_data: Optional[np.ndarray] = None
    explained_variance_ratio: Optional[np.ndarray] = None

    # For anomaly detection
    anomaly_scores: Optional[np.ndarray] = None
    anomaly_labels: Optional[np.ndarray] = None


@dataclass
class UnsupervisedResult:
    """Final unsupervised AutoML results"""
    task_subtype: UnsupervisedTaskType
    best_model: UnsupervisedModelResult
    all_models: List[UnsupervisedModelResult]
    total_models_evaluated: int
    models_rejected: int
    plot_paths: Dict[str, str] = field(default_factory=dict)
    training_timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    feature_names: List[str] = field(default_factory=list)
    n_samples: int = 0
    n_features: int = 0


class UnsupervisedEngine:
    """
    AutoML engine for unsupervised learning tasks.

    Supports clustering, dimensionality reduction, and anomaly detection
    with Optuna hyperparameter optimization and internal evaluation metrics.
    """

    # Model registries
    CLUSTERING_MODELS = {
        'kmeans': KMeans,
        'mini_batch_kmeans': MiniBatchKMeans,
        'dbscan': DBSCAN,
        'agglomerative': AgglomerativeClustering,
    }

    DIMENSIONALITY_REDUCTION_MODELS = {
        'pca': PCA,
        'truncated_svd': TruncatedSVD,
    }

    ANOMALY_DETECTION_MODELS = {
        'isolation_forest': IsolationForest,
        'local_outlier_factor': LocalOutlierFactor,
    }

    def __init__(
        self,
        task_subtype: UnsupervisedTaskType = UnsupervisedTaskType.CLUSTERING,
        n_trials: int = 50,
        random_state: int = 42,
        max_cpu_cores: int = 4,
        verbose: bool = True,
    ):
        self.task_subtype = task_subtype
        self.n_trials = n_trials
        self.random_state = random_state
        self.max_cpu_cores = max_cpu_cores
        self.verbose = verbose

        self._setup_cpu_limits()

        # Add optional models
        if HAS_HDBSCAN:
            self.CLUSTERING_MODELS = {**self.CLUSTERING_MODELS, 'hdbscan': HDBSCAN}
        if HAS_UMAP:
            self.DIMENSIONALITY_REDUCTION_MODELS = {
                **self.DIMENSIONALITY_REDUCTION_MODELS,
                'umap': UMAP,
            }

        # Select model registry
        if task_subtype == UnsupervisedTaskType.CLUSTERING:
            self.models = dict(self.CLUSTERING_MODELS)
        elif task_subtype == UnsupervisedTaskType.DIMENSIONALITY_REDUCTION:
            self.models = dict(self.DIMENSIONALITY_REDUCTION_MODELS)
        elif task_subtype == UnsupervisedTaskType.ANOMALY_DETECTION:
            self.models = dict(self.ANOMALY_DETECTION_MODELS)
        else:
            self.models = dict(self.CLUSTERING_MODELS)

        self.results: List[UnsupervisedModelResult] = []
        self.best_result: Optional[UnsupervisedModelResult] = None

        # Data-adaptive stats (computed in fit())
        self._nn_distances: Optional[np.ndarray] = None  # k-NN distances for DBSCAN eps calibration

    def _compute_data_stats(self, X: np.ndarray):
        """Compute data-adaptive statistics for smart hyperparameter ranges."""
        from sklearn.neighbors import NearestNeighbors
        k = min(5, X.shape[0] - 1)
        nn = NearestNeighbors(n_neighbors=k)
        nn.fit(X)
        distances, _ = nn.kneighbors(X)
        # k-th nearest neighbor distance distribution (used for DBSCAN eps)
        self._nn_distances = np.sort(distances[:, -1])

    def _setup_cpu_limits(self):
        """Configure CPU core limits"""
        os.environ['OMP_NUM_THREADS'] = str(self.max_cpu_cores)
        os.environ['MKL_NUM_THREADS'] = str(self.max_cpu_cores)
        os.environ['OPENBLAS_NUM_THREADS'] = str(self.max_cpu_cores)
        os.environ['NUMEXPR_NUM_THREADS'] = str(self.max_cpu_cores)

        try:
            import psutil
            process = psutil.Process()
            total_cores = psutil.cpu_count(logical=True)
            cores_to_use = list(range(min(self.max_cpu_cores, total_cores)))
            process.cpu_affinity(cores_to_use)
            if self.verbose:
                print(f"✓ CPU cores limited to {len(cores_to_use)} of {total_cores}")
        except Exception:
            pass

    # ─── Search Spaces ────────────────────────────────────────────────

    def _define_search_space(
        self,
        trial: optuna.Trial,
        model_name: str,
        n_samples: int,
        n_features: int,
    ) -> Dict[str, Any]:
        """Define Optuna search space per model."""

        max_clusters = min(20, max(2, n_samples // 10))

        # ─── Clustering ───
        if model_name == 'kmeans':
            return {
                'n_clusters': trial.suggest_int('n_clusters', 2, max_clusters),
                'init': trial.suggest_categorical('init', ['k-means++', 'random']),
                'n_init': trial.suggest_int('n_init', 5, 20),
                'max_iter': trial.suggest_int('max_iter', 100, 500),
                'random_state': self.random_state,
            }

        elif model_name == 'mini_batch_kmeans':
            return {
                'n_clusters': trial.suggest_int('n_clusters', 2, max_clusters),
                'batch_size': trial.suggest_int('batch_size', 100, min(1024, n_samples)),
                'n_init': trial.suggest_int('n_init', 3, 10),
                'max_iter': trial.suggest_int('max_iter', 100, 500),
                'random_state': self.random_state,
            }

        elif model_name == 'dbscan':
            # Data-adaptive eps range using k-NN distance distribution
            if self._nn_distances is not None and len(self._nn_distances) > 10:
                eps_low = float(np.percentile(self._nn_distances, 5))
                eps_high = float(np.percentile(self._nn_distances, 95))
                eps_low = max(eps_low, 0.01)
                eps_high = max(eps_high, eps_low * 2)
            else:
                eps_low, eps_high = 0.1, 5.0
            return {
                'eps': trial.suggest_float('eps', eps_low, eps_high, log=True),
                'min_samples': trial.suggest_int('min_samples', 2, min(50, max(2, n_samples // 20))),
                'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan']),
            }

        elif model_name == 'agglomerative':
            linkage = trial.suggest_categorical('linkage', ['ward', 'complete', 'average', 'single'])
            params = {
                'n_clusters': trial.suggest_int('n_clusters', 2, max_clusters),
                'linkage': linkage,
            }
            if linkage != 'ward':
                params['metric'] = trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'cosine'])
            return params

        elif model_name == 'hdbscan':
            return {
                'min_cluster_size': trial.suggest_int('min_cluster_size', 5, min(100, max(5, n_samples // 10))),
                'min_samples': trial.suggest_int('min_samples', 1, min(30, max(1, n_samples // 20))),
                'cluster_selection_method': trial.suggest_categorical(
                    'cluster_selection_method', ['eom', 'leaf']
                ),
            }

        # ─── Dimensionality Reduction ───
        elif model_name == 'pca':
            return {
                'n_components': trial.suggest_int('n_components', 2, min(n_features, max(2, n_features - 1))),
                'random_state': self.random_state,
            }

        elif model_name == 'truncated_svd':
            return {
                'n_components': trial.suggest_int('n_components', 2, min(n_features - 1, max(2, n_features - 1))),
                'n_iter': trial.suggest_int('n_iter', 5, 20),
                'random_state': self.random_state,
            }

        elif model_name == 'umap':
            return {
                'n_components': trial.suggest_int('n_components', 2, min(50, n_features)),
                'n_neighbors': trial.suggest_int('n_neighbors', 5, min(50, n_samples - 1)),
                'min_dist': trial.suggest_float('min_dist', 0.0, 0.99),
                'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'cosine']),
                'random_state': self.random_state,
            }

        # ─── Anomaly Detection ───
        elif model_name == 'isolation_forest':
            return {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'contamination': trial.suggest_float('contamination', 0.01, 0.3),
                'max_features': trial.suggest_float('max_features', 0.5, 1.0),
                'max_samples': trial.suggest_categorical('max_samples', ['auto', 0.5, 0.8, 1.0]),
                'random_state': self.random_state,
            }

        elif model_name == 'local_outlier_factor':
            return {
                'n_neighbors': trial.suggest_int('n_neighbors', 5, min(50, n_samples - 1)),
                'contamination': trial.suggest_float('contamination', 0.01, 0.3),
                'metric': trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'minkowski']),
                'novelty': True,  # Required for predict()
            }

        return {}

    # ─── Objective Functions ──────────────────────────────────────────

    def _create_clustering_objective(
        self,
        model_name: str,
        model_class,
        X: np.ndarray,
        timeout_flag: list,
        cancellation_callback=None,
        skip_callback=None,
        trial_progress_callback=None,
    ):
        """
        Optuna objective for clustering: composite score.

        Uses silhouette as the primary metric but applies penalties for:
        - Too much noise (DBSCAN)
        - Degenerate cluster sizes (all points in one cluster)
        - Trivial 2-cluster solutions that just split data in half
        """
        n_samples, n_features = X.shape

        def objective(trial: optuna.Trial) -> float:
            if skip_callback and skip_callback():
                timeout_flag[0] = True
                return -9999
            if cancellation_callback and cancellation_callback():
                timeout_flag[0] = True
                return -9999

            if trial_progress_callback:
                try:
                    best_val = trial.study.best_value
                except ValueError:
                    best_val = None
                trial_progress_callback(trial.number + 1, self.n_trials, best_val)

            params = self._define_search_space(trial, model_name, n_samples, n_features)

            try:
                model = model_class(**params)
                model.fit(X)

                labels = model.labels_ if hasattr(model, 'labels_') else model.predict(X)
                n_clusters_found = len(set(labels) - {-1})

                if n_clusters_found < 2:
                    return -1.0  # Need at least 2 clusters

                # Use non-noise points for scoring
                mask = labels != -1
                n_valid = mask.sum()
                if n_valid < 2:
                    return -1.0

                sil = silhouette_score(X[mask], labels[mask])

                # Penalty: too much noise (>50% of points are noise)
                noise_ratio = 1.0 - (n_valid / n_samples)
                if noise_ratio > 0.5:
                    sil *= (1.0 - noise_ratio)  # Heavy penalty
                elif noise_ratio > 0.2:
                    sil *= (1.0 - noise_ratio * 0.3)  # Mild penalty

                # Penalty: highly imbalanced clusters
                unique_labels = set(labels[mask])
                if len(unique_labels) >= 2:
                    cluster_sizes = [np.sum(labels[mask] == l) for l in unique_labels]
                    largest = max(cluster_sizes)
                    if largest / n_valid > 0.9:
                        # One cluster has >90% of points → near-trivial
                        sil *= 0.5

                return sil

            except Exception as e:
                logger.warning(f"Clustering trial failed for {model_name}: {e}")
                return -9999

        return objective

    def _create_dimred_objective(
        self,
        model_name: str,
        model_class,
        X: np.ndarray,
        timeout_flag: list,
        cancellation_callback=None,
        skip_callback=None,
        trial_progress_callback=None,
    ):
        """Optuna objective for dimensionality reduction: maximize explained variance."""
        n_samples, n_features = X.shape

        def objective(trial: optuna.Trial) -> float:
            if skip_callback and skip_callback():
                timeout_flag[0] = True
                return -9999
            if cancellation_callback and cancellation_callback():
                timeout_flag[0] = True
                return -9999

            if trial_progress_callback:
                try:
                    best_val = trial.study.best_value
                except ValueError:
                    best_val = None
                trial_progress_callback(trial.number + 1, self.n_trials, best_val)

            params = self._define_search_space(trial, model_name, n_samples, n_features)

            try:
                model = model_class(**params)
                model.fit(X)

                if hasattr(model, 'explained_variance_ratio_'):
                    total_explained = np.sum(model.explained_variance_ratio_)
                    n_components = params.get('n_components', n_features)
                    # Reward: high explained variance with fewer components
                    # Penalty for too many components (prefer parsimony)
                    parsimony_bonus = 1.0 - (n_components / n_features) * 0.1
                    score = total_explained * parsimony_bonus
                else:
                    # For UMAP and others without explained_variance: use reconstruction quality
                    X_transformed = model.transform(X)
                    # Score: inverse of mean pairwise distance distortion (higher = better)
                    # Approximate by checking local neighborhood preservation
                    from sklearn.neighbors import NearestNeighbors
                    k = min(10, n_samples - 1)
                    nn_orig = NearestNeighbors(n_neighbors=k).fit(X)
                    nn_trans = NearestNeighbors(n_neighbors=k).fit(X_transformed)
                    _, idx_orig = nn_orig.kneighbors(X)
                    _, idx_trans = nn_trans.kneighbors(X_transformed)

                    # Trustworthiness: fraction of neighbors preserved
                    preserved = 0
                    for i in range(n_samples):
                        preserved += len(set(idx_orig[i]) & set(idx_trans[i]))
                    score = preserved / (n_samples * k)

                return score

            except Exception as e:
                logger.warning(f"DimRed trial failed for {model_name}: {e}")
                return -9999

        return objective

    def _create_anomaly_objective(
        self,
        model_name: str,
        model_class,
        X: np.ndarray,
        timeout_flag: list,
        cancellation_callback=None,
        skip_callback=None,
        trial_progress_callback=None,
    ):
        """
        Optuna objective for anomaly detection.
        Uses internal consistency: silhouette score of inlier/outlier separation.
        """
        n_samples, n_features = X.shape

        def objective(trial: optuna.Trial) -> float:
            if skip_callback and skip_callback():
                timeout_flag[0] = True
                return -9999
            if cancellation_callback and cancellation_callback():
                timeout_flag[0] = True
                return -9999

            if trial_progress_callback:
                try:
                    best_val = trial.study.best_value
                except ValueError:
                    best_val = None
                trial_progress_callback(trial.number + 1, self.n_trials, best_val)

            params = self._define_search_space(trial, model_name, n_samples, n_features)

            try:
                model = model_class(**params)

                if model_name == 'local_outlier_factor':
                    model.fit(X)
                    labels = model.predict(X)
                else:
                    labels = model.fit_predict(X)

                # Labels: 1 = inlier, -1 = outlier
                n_outliers = np.sum(labels == -1)
                n_inliers = np.sum(labels == 1)

                # Need both classes
                if n_outliers == 0 or n_inliers == 0:
                    return -1.0

                # Too few or too many outliers is suspicious
                outlier_ratio = n_outliers / n_samples
                if outlier_ratio > 0.5 or outlier_ratio < 0.001:
                    return -0.5

                # Internal consistency: silhouette of inlier vs outlier groups
                binary_labels = (labels == 1).astype(int)
                if len(np.unique(binary_labels)) > 1:
                    score = silhouette_score(X, binary_labels)
                else:
                    score = -1.0

                return score

            except Exception as e:
                logger.warning(f"Anomaly trial failed for {model_name}: {e}")
                return -9999

        return objective

    # ─── Training ─────────────────────────────────────────────────────

    def _train_single_model(
        self,
        model_name: str,
        X: np.ndarray,
        feature_names: List[str],
        cancellation_callback=None,
        skip_callback=None,
        trial_progress_callback=None,
    ) -> UnsupervisedModelResult:
        """Train a single unsupervised model with Optuna optimization."""

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"Training: {model_name.upper()} ({self.task_subtype.value})")
            print(f"{'='*60}")

        model_class = self.models[model_name]
        start_time = datetime.now()
        n_samples, n_features = X.shape

        # Create Optuna study (maximize)
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=self.random_state),
            pruner=MedianPruner(),
        )

        timeout_flag = [False]
        stopped_early = [False]
        consecutive_failures = [0]

        def check_early_stop(study: optuna.Study, trial: optuna.Trial):
            if cancellation_callback and cancellation_callback():
                stopped_early[0] = True
                study.stop()
                return
            if skip_callback and skip_callback():
                stopped_early[0] = True
                study.stop()
                return
            if timeout_flag[0]:
                stopped_early[0] = True
                study.stop()
                return
            if trial.value is not None and trial.value <= -9999:
                consecutive_failures[0] += 1
                if consecutive_failures[0] >= 5:
                    stopped_early[0] = True
                    study.stop()
            else:
                consecutive_failures[0] = 0

        # Select objective
        if self.task_subtype == UnsupervisedTaskType.CLUSTERING:
            objective = self._create_clustering_objective(
                model_name, model_class, X, timeout_flag,
                cancellation_callback, skip_callback, trial_progress_callback,
            )
        elif self.task_subtype == UnsupervisedTaskType.DIMENSIONALITY_REDUCTION:
            objective = self._create_dimred_objective(
                model_name, model_class, X, timeout_flag,
                cancellation_callback, skip_callback, trial_progress_callback,
            )
        elif self.task_subtype == UnsupervisedTaskType.ANOMALY_DETECTION:
            objective = self._create_anomaly_objective(
                model_name, model_class, X, timeout_flag,
                cancellation_callback, skip_callback, trial_progress_callback,
            )
        else:
            raise ValueError(f"Unknown task subtype: {self.task_subtype}")

        # Check pre-cancellation
        if cancellation_callback and cancellation_callback():
            return self._make_cancelled_result(model_name, model_class)

        try:
            study.optimize(
                objective,
                n_trials=self.n_trials,
                show_progress_bar=False,
                n_jobs=1,
                callbacks=[check_early_stop],
            )
        except optuna.exceptions.OptunaError:
            pass

        optimization_time = (datetime.now() - start_time).total_seconds()

        # Handle skip / cancel
        if skip_callback and skip_callback():
            return self._make_skipped_result(model_name, model_class, optimization_time)
        if cancellation_callback and cancellation_callback():
            return self._make_cancelled_result(model_name, model_class, optimization_time)

        # Check for valid trials
        valid_trials = [t for t in study.trials if t.value is not None and t.value > -9999]
        if not valid_trials:
            if self.verbose:
                print(f"❌ No valid trials for {model_name}")
            return UnsupervisedModelResult(
                model_name=model_name,
                model_type=model_class.__name__,
                task_subtype=self.task_subtype,
                best_model=None,
                best_params={},
                primary_score=-1.0,
                n_trials=len(study.trials),
                optimization_time=optimization_time,
                rejected=True,
                rejection_reason="No valid trials found",
                stability_status="unstable",
                stability_reason="All trials failed",
            )

        # Retrain with best params
        best_params = study.best_params
        best_trial = study.best_trial

        if self.verbose:
            print(f"Best trial: {best_trial.number}, score: {best_trial.value:.4f}")
            print(f"Best params: {best_params}")

        # Rebuild full params (handle search space specifics)
        full_params = self._define_search_space(
            best_trial, model_name, n_samples, n_features
        )
        # Override with the actual best params from Optuna
        for k, v in best_params.items():
            full_params[k] = v

        best_model = model_class(**full_params)

        # Fit and evaluate
        result = self._evaluate_model(
            model_name, model_class, best_model, full_params, X, feature_names,
            best_trial, optimization_time, len(study.trials)
        )

        return result

    def _evaluate_model(
        self,
        model_name: str,
        model_class,
        model,
        params: Dict[str, Any],
        X: np.ndarray,
        feature_names: List[str],
        best_trial: optuna.trial.FrozenTrial,
        optimization_time: float,
        total_trials: int,
    ) -> UnsupervisedModelResult:
        """Fit final model and compute all metrics."""

        n_samples, n_features = X.shape

        if self.task_subtype == UnsupervisedTaskType.CLUSTERING:
            model.fit(X)
            labels = model.labels_ if hasattr(model, 'labels_') else model.predict(X)
            mask = labels != -1
            n_clusters = len(set(labels) - {-1})

            if n_clusters >= 2 and mask.sum() >= 2:
                sil = silhouette_score(X[mask], labels[mask])
                ch = calinski_harabasz_score(X[mask], labels[mask])
                db = davies_bouldin_score(X[mask], labels[mask])
                primary_score = sil
            else:
                sil, ch, db = -1.0, 0.0, float('inf')
                primary_score = -1.0

            # Cluster size distribution
            unique, counts = np.unique(labels[labels != -1], return_counts=True)
            cluster_sizes = {f"cluster_{int(u)}": int(c) for u, c in zip(unique, counts)}
            noise_count = int(np.sum(labels == -1))

            detailed_metrics = {
                'silhouette_score': float(sil),
                'calinski_harabasz_score': float(ch),
                'davies_bouldin_score': float(db),
                'n_clusters': n_clusters,
                'cluster_sizes': cluster_sizes,
                'noise_points': noise_count,
                'noise_ratio': noise_count / n_samples if n_samples > 0 else 0,
            }

            return UnsupervisedModelResult(
                model_name=model_name,
                model_type=model_class.__name__,
                task_subtype=self.task_subtype,
                best_model=model,
                best_params=params,
                primary_score=primary_score,
                detailed_metrics=detailed_metrics,
                labels=labels,
                n_trials=total_trials,
                best_trial_number=best_trial.number,
                optimization_time=optimization_time,
            )

        elif self.task_subtype == UnsupervisedTaskType.DIMENSIONALITY_REDUCTION:
            model.fit(X)
            X_transformed = model.transform(X)

            if hasattr(model, 'explained_variance_ratio_'):
                evr = model.explained_variance_ratio_
                total_explained = float(np.sum(evr))
                primary_score = total_explained
                detailed_metrics = {
                    'explained_variance_ratio': evr.tolist(),
                    'total_explained_variance': total_explained,
                    'n_components': int(model.n_components) if hasattr(model, 'n_components') else len(evr),
                    'cumulative_explained_variance': np.cumsum(evr).tolist(),
                }
            else:
                # Trustworthiness for UMAP
                from sklearn.neighbors import NearestNeighbors
                k = min(10, n_samples - 1)
                nn_orig = NearestNeighbors(n_neighbors=k).fit(X)
                nn_trans = NearestNeighbors(n_neighbors=k).fit(X_transformed)
                _, idx_orig = nn_orig.kneighbors(X)
                _, idx_trans = nn_trans.kneighbors(X_transformed)
                preserved = sum(
                    len(set(idx_orig[i]) & set(idx_trans[i]))
                    for i in range(n_samples)
                )
                trustworthiness = preserved / (n_samples * k)
                primary_score = trustworthiness
                detailed_metrics = {
                    'trustworthiness': trustworthiness,
                    'n_components': X_transformed.shape[1],
                }

            return UnsupervisedModelResult(
                model_name=model_name,
                model_type=model_class.__name__,
                task_subtype=self.task_subtype,
                best_model=model,
                best_params=params,
                primary_score=primary_score,
                detailed_metrics=detailed_metrics,
                transformed_data=X_transformed,
                explained_variance_ratio=evr if hasattr(model, 'explained_variance_ratio_') else None,
                n_trials=total_trials,
                best_trial_number=best_trial.number,
                optimization_time=optimization_time,
            )

        elif self.task_subtype == UnsupervisedTaskType.ANOMALY_DETECTION:
            if model_name == 'local_outlier_factor':
                model.fit(X)
                labels = model.predict(X)
                scores = -model.decision_function(X)  # Higher = more anomalous
            else:
                labels = model.fit_predict(X)
                scores = -model.decision_function(X) if hasattr(model, 'decision_function') else np.zeros(n_samples)

            n_outliers = int(np.sum(labels == -1))
            n_inliers = int(np.sum(labels == 1))
            outlier_ratio = n_outliers / n_samples if n_samples > 0 else 0

            # Primary score: silhouette of inlier/outlier separation
            binary_labels = (labels == 1).astype(int)
            if len(np.unique(binary_labels)) > 1 and n_outliers > 0 and n_inliers > 0:
                primary_score = silhouette_score(X, binary_labels)
            else:
                primary_score = -1.0

            detailed_metrics = {
                'n_outliers': n_outliers,
                'n_inliers': n_inliers,
                'outlier_ratio': outlier_ratio,
                'anomaly_score_mean': float(np.mean(scores)),
                'anomaly_score_std': float(np.std(scores)),
                'silhouette_score': float(primary_score),
            }

            return UnsupervisedModelResult(
                model_name=model_name,
                model_type=model_class.__name__,
                task_subtype=self.task_subtype,
                best_model=model,
                best_params=params,
                primary_score=primary_score,
                detailed_metrics=detailed_metrics,
                anomaly_labels=labels,
                anomaly_scores=scores,
                n_trials=total_trials,
                best_trial_number=best_trial.number,
                optimization_time=optimization_time,
            )

        # Fallback
        return UnsupervisedModelResult(
            model_name=model_name,
            model_type=model_class.__name__,
            task_subtype=self.task_subtype,
            best_model=model,
            best_params=params,
            primary_score=-1.0,
            rejected=True,
            rejection_reason="Unknown task subtype",
        )

    # ─── Helpers ──────────────────────────────────────────────────────

    def _make_cancelled_result(self, model_name, model_class, opt_time=0.0):
        return UnsupervisedModelResult(
            model_name=model_name,
            model_type=model_class.__name__,
            task_subtype=self.task_subtype,
            best_model=None,
            best_params={},
            primary_score=-9999,
            stability_status="cancelled",
            stability_reason="Cancelled by user",
            rejected=True,
            rejection_reason="Training cancelled",
            optimization_time=opt_time,
        )

    def _make_skipped_result(self, model_name, model_class, opt_time=0.0):
        return UnsupervisedModelResult(
            model_name=model_name,
            model_type=model_class.__name__,
            task_subtype=self.task_subtype,
            best_model=None,
            best_params={},
            primary_score=-9999,
            stability_status="skipped",
            stability_reason="Skipped by user",
            rejected=True,
            rejection_reason="Model skipped",
            optimization_time=opt_time,
        )

    # ─── Main Fit ─────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        model_subset: Optional[List[str]] = None,
        cancellation_callback=None,
        force_cancel_flag=None,
        skip_callback=None,
        on_model_start=None,
        on_model_complete=None,
        trial_progress_callback=None,
    ) -> UnsupervisedResult:
        """
        Train all unsupervised models and select the best one.

        Args:
            X: Feature matrix (all data, no train/test split for unsupervised)
            feature_names: Feature names
            model_subset: Specific models to train
            cancellation_callback: Returns True if cancelled
            force_cancel_flag: Mutable list [bool] for force cancel
            skip_callback: Returns True if skip requested
            on_model_start: Callback(model_name) when model training starts
            on_model_complete: Callback(model_name, score, status) when done
            trial_progress_callback: Callback(trial_num, total, best_score)

        Returns:
            UnsupervisedResult
        """
        if force_cancel_flag is None:
            force_cancel_flag = [False]
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]

        if self.verbose:
            print(f"\n🚀 Unsupervised Training Started")
            print(f"Task: {self.task_subtype.value}")
            print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}")
            print(f"Trials per model: {self.n_trials}")

        # Compute data-adaptive stats for DBSCAN eps calibration etc.
        if self.task_subtype == UnsupervisedTaskType.CLUSTERING:
            try:
                self._compute_data_stats(X)
                if self.verbose:
                    print(f"   k-NN eps range: [{self._nn_distances[int(len(self._nn_distances)*0.05)]:.4f}, {self._nn_distances[int(len(self._nn_distances)*0.95)]:.4f}]")
            except Exception as e:
                logger.warning(f"Could not compute data stats: {e}")

        models_to_train = model_subset or list(self.models.keys())

        for model_name in models_to_train:
            if model_name not in self.models:
                if self.verbose:
                    print(f"⚠️ Unknown model: {model_name}, skipping")
                continue

            # Check cancellation
            if cancellation_callback and cancellation_callback():
                result = self._make_cancelled_result(model_name, self.models[model_name])
                self.results.append(result)
                if on_model_complete:
                    on_model_complete(model_name, None, "cancelled")
                continue

            # Notify start
            if on_model_start:
                on_model_start(model_name)

            try:
                result = self._train_single_model(
                    model_name, X, feature_names,
                    cancellation_callback, skip_callback, trial_progress_callback,
                )
                self.results.append(result)

                if on_model_complete:
                    score = result.primary_score if result.primary_score != -9999 else None
                    result_status = "skipped" if result.stability_status == "skipped" else (
                        "cancelled" if result.stability_status == "cancelled" else "completed"
                    )
                    on_model_complete(model_name, score, result_status)

            except Exception as e:
                logger.error(f"Error training {model_name}: {e}")
                if on_model_complete:
                    on_model_complete(model_name, None, "failed")

            gc.collect()

        # Select best model
        valid = [r for r in self.results if not r.rejected and r.best_model is not None]
        if not valid:
            non_none = [r for r in self.results if r.best_model is not None]
            if non_none:
                self.best_result = max(non_none, key=lambda r: r.primary_score)
                self.best_result.rejected = False
            elif self.results:
                self.best_result = self.results[0]
            else:
                raise ValueError("No models could be trained. Check data quality.")
        else:
            self.best_result = max(valid, key=lambda r: r.primary_score)

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"🏆 BEST MODEL: {self.best_result.model_name}")
            print(f"   Score: {self.best_result.primary_score:.4f}")
            print(f"{'='*60}")

        return UnsupervisedResult(
            task_subtype=self.task_subtype,
            best_model=self.best_result,
            all_models=self.results,
            total_models_evaluated=len(self.results),
            models_rejected=len([r for r in self.results if r.rejected]),
            feature_names=feature_names,
            n_samples=X.shape[0],
            n_features=X.shape[1],
        )

    # ─── Predict ──────────────────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster labels or anomaly labels for new data."""
        if self.best_result is None or self.best_result.best_model is None:
            raise ValueError("No model trained. Call fit() first.")

        model = self.best_result.best_model
        if hasattr(model, 'predict'):
            return model.predict(X)
        elif hasattr(model, 'labels_'):
            # Models like AgglomerativeClustering don't have predict
            # Re-fit on combined data is not ideal; use nearest-centroid fallback
            raise ValueError(
                f"{self.best_result.model_name} does not support predict() on new data. "
                "Use a model with predict support (e.g., KMeans, IsolationForest)."
            )
        else:
            raise ValueError(f"Model {self.best_result.model_name} has no predict method.")

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data (for dimensionality reduction models)."""
        if self.best_result is None or self.best_result.best_model is None:
            raise ValueError("No model trained. Call fit() first.")
        if hasattr(self.best_result.best_model, 'transform'):
            return self.best_result.best_model.transform(X)
        raise ValueError(f"Model {self.best_result.model_name} does not support transform().")

    # ─── Save / Load ──────────────────────────────────────────────────

    def save(self, filepath: str, **kwargs):
        """Save model and results to disk."""
        if self.best_result is None:
            raise ValueError("No model trained.")

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Prepare labels/scores for serialization
        best_result_copy = UnsupervisedModelResult(
            model_name=self.best_result.model_name,
            model_type=self.best_result.model_type,
            task_subtype=self.best_result.task_subtype,
            best_model=self.best_result.best_model,
            best_params=self.best_result.best_params,
            primary_score=self.best_result.primary_score,
            detailed_metrics=self.best_result.detailed_metrics,
            n_trials=self.best_result.n_trials,
            best_trial_number=self.best_result.best_trial_number,
            optimization_time=self.best_result.optimization_time,
            stability_status=self.best_result.stability_status,
            stability_reason=self.best_result.stability_reason,
            rejected=self.best_result.rejected,
            rejection_reason=self.best_result.rejection_reason,
            labels=self.best_result.labels,
            anomaly_labels=self.best_result.anomaly_labels,
            anomaly_scores=self.best_result.anomaly_scores,
            explained_variance_ratio=self.best_result.explained_variance_ratio,
            # Don't save full transformed_data to keep file size down
            transformed_data=None,
        )

        save_data = {
            'best_model': self.best_result.best_model,
            'best_result': best_result_copy,
            'all_results': [
                UnsupervisedModelResult(
                    model_name=r.model_name,
                    model_type=r.model_type,
                    task_subtype=r.task_subtype,
                    best_model=r.best_model,
                    best_params=r.best_params,
                    primary_score=r.primary_score,
                    detailed_metrics=r.detailed_metrics,
                    n_trials=r.n_trials,
                    best_trial_number=r.best_trial_number,
                    optimization_time=r.optimization_time,
                    stability_status=r.stability_status,
                    stability_reason=r.stability_reason,
                    rejected=r.rejected,
                    rejection_reason=r.rejection_reason,
                    labels=r.labels,
                    anomaly_labels=r.anomaly_labels,
                    anomaly_scores=r.anomaly_scores,
                    explained_variance_ratio=r.explained_variance_ratio,
                    transformed_data=None,
                )
                for r in self.results
            ],
            'task_subtype': self.task_subtype,
            'preprocessing_pipeline': kwargs.get('preprocessing_pipeline'),
            'config': {
                'n_trials': self.n_trials,
                'random_state': self.random_state,
                'max_cpu_cores': self.max_cpu_cores,
            },
        }

        joblib.dump(save_data, filepath)

    @staticmethod
    def load(filepath: str) -> 'UnsupervisedEngine':
        """Load a saved unsupervised engine."""
        data = joblib.load(filepath)

        engine = UnsupervisedEngine(
            task_subtype=data['task_subtype'],
            n_trials=data['config']['n_trials'],
            random_state=data['config']['random_state'],
            max_cpu_cores=data['config']['max_cpu_cores'],
        )

        engine.best_result = data['best_result']
        engine.results = data['all_results']

        return engine
