"""
AutoML Evaluation Plots Generator
==================================

Generates comprehensive visualization plots for model evaluation:
- Classification: Confusion matrix, feature importance, ROC curves
- Regression: Predicted vs Actual, residuals, distribution plots
- Clustering: PCA 2D clusters, cluster size charts
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import os
from datetime import datetime

from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, roc_auc_score
)
from sklearn.decomposition import PCA
from sklearn.preprocessing import label_binarize

from .automl_engine import ModelResult, AutoMLResult, TaskType


class AutoMLPlotter:
    """
    Generates and saves evaluation plots for AutoML results.
    """
    
    def __init__(self, output_dir: str = "plots", dpi: int = 100):
        """
        Initialize plotter.
        
        Args:
            output_dir: Directory to save plots
            dpi: Resolution for saved plots
        """
        self.output_dir = output_dir
        self.dpi = dpi
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.facecolor'] = 'white'
    
    def _get_plot_path(self, filename: str) -> str:
        """Generate unique plot path with timestamp"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(filename)
        unique_filename = f"{base}_{timestamp}{ext}"
        return os.path.join(self.output_dir, unique_filename)
    
    # ==================== CLASSIFICATION PLOTS ====================
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str,
        classes: Optional[List[str]] = None
    ) -> str:
        """
        Plot confusion matrix for classification.
        
        Returns:
            Path to saved plot
        """
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=classes or np.unique(y_true),
            yticklabels=classes or np.unique(y_true),
            ax=ax, cbar_kws={'label': 'Count'}
        )
        
        ax.set_xlabel('Predicted Label', fontsize=12)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        # Save
        path = self._get_plot_path('confusion_matrix.png')
        plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return path
    
    def plot_feature_importance(
        self,
        feature_importance: Dict[str, float],
        model_name: str,
        top_n: int = 20
    ) -> str:
        """
        Plot feature importance bar chart.
        
        Returns:
            Path to saved plot
        """
        # Get top N features
        sorted_features = dict(sorted(
            feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_n])
        
        features = list(sorted_features.keys())
        importances = list(sorted_features.values())
        
        fig, ax = plt.subplots(figsize=(12, max(6, len(features) * 0.3)))
        
        # Horizontal bar chart
        y_pos = np.arange(len(features))
        ax.barh(y_pos, importances, color='steelblue', alpha=0.8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(f'Feature Importance - {model_name} (Top {top_n})', 
                     fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        path = self._get_plot_path('feature_importance.png')
        plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return path
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        model_name: str,
        classes: Optional[List[str]] = None
    ) -> str:
        """
        Plot ROC curve for binary or multiclass classification.
        
        Returns:
            Path to saved plot
        """
        n_classes = y_proba.shape[1] if len(y_proba.shape) > 1 else 2
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        if n_classes == 2:
            # Binary classification
            fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1] if len(y_proba.shape) > 1 else y_proba)
            roc_auc = auc(fpr, tpr)
            
            ax.plot(fpr, tpr, color='darkorange', lw=2,
                   label=f'ROC curve (AUC = {roc_auc:.3f})')
        else:
            # Multiclass - one curve per class
            y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
            
            for i in range(n_classes):
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
                roc_auc = auc(fpr, tpr)
                
                class_label = classes[i] if classes else f'Class {i}'
                ax.plot(fpr, tpr, lw=2,
                       label=f'{class_label} (AUC = {roc_auc:.3f})')
        
        # Diagonal reference line
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5,
               label='Random Classifier')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12)
        ax.set_ylabel('True Positive Rate', fontsize=12)
        ax.set_title(f'ROC Curve - {model_name}', fontsize=14, fontweight='bold')
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        path = self._get_plot_path('roc_curve.png')
        plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return path
    
    # ==================== REGRESSION PLOTS ====================
    
    def plot_predicted_vs_actual(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str
    ) -> str:
        """
        Plot predicted vs actual values for regression.
        
        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.5, s=50, color='steelblue', edgecolors='black', linewidth=0.5)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 
               'r--', lw=2, label='Perfect Prediction', alpha=0.7)
        
        # Calculate R²
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true, y_pred)
        
        ax.set_xlabel('Actual Values', fontsize=12)
        ax.set_ylabel('Predicted Values', fontsize=12)
        ax.set_title(f'Predicted vs Actual - {model_name}\nR² = {r2:.4f}', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        path = self._get_plot_path('predicted_vs_actual.png')
        plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return path
    
    def plot_residuals(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str
    ) -> str:
        """
        Plot residuals for regression.
        
        Returns:
            Path to saved plot
        """
        residuals = y_true - y_pred
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Scatter plot
        ax.scatter(y_pred, residuals, alpha=0.5, s=50, 
                  color='steelblue', edgecolors='black', linewidth=0.5)
        
        # Zero line
        ax.axhline(y=0, color='r', linestyle='--', lw=2, alpha=0.7)
        
        ax.set_xlabel('Predicted Values', fontsize=12)
        ax.set_ylabel('Residuals', fontsize=12)
        ax.set_title(f'Residual Plot - {model_name}', fontsize=14, fontweight='bold')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        path = self._get_plot_path('residuals.png')
        plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return path
    
    def plot_residual_distribution(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str
    ) -> str:
        """
        Plot distribution of residuals (normality check).
        
        Returns:
            Path to saved plot
        """
        residuals = y_true - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Histogram
        axes[0].hist(residuals, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0].axvline(x=0, color='r', linestyle='--', lw=2, alpha=0.7)
        axes[0].set_xlabel('Residuals', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Residual Distribution', fontsize=12, fontweight='bold')
        axes[0].grid(alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1])
        axes[1].set_title('Q-Q Plot', fontsize=12, fontweight='bold')
        axes[1].grid(alpha=0.3)
        
        fig.suptitle(f'Residual Analysis - {model_name}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save
        path = self._get_plot_path('residual_distribution.png')
        plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return path
    
    # ==================== CLUSTERING PLOTS ====================
    
    def plot_clusters_2d(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        model_name: str,
        feature_names: Optional[List[str]] = None
    ) -> str:
        """
        Plot clusters in 2D using PCA.
        
        Returns:
            Path to saved plot
        """
        # Apply PCA if more than 2 dimensions
        if X.shape[1] > 2:
            pca = PCA(n_components=2, random_state=42)
            X_2d = pca.fit_transform(X)
            variance = pca.explained_variance_ratio_
            xlabel = f'PC1 ({variance[0]:.2%} variance)'
            ylabel = f'PC2 ({variance[1]:.2%} variance)'
        else:
            X_2d = X
            xlabel = feature_names[0] if feature_names else 'Feature 1'
            ylabel = feature_names[1] if feature_names and len(feature_names) > 1 else 'Feature 2'
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Scatter plot with colors for each cluster
        unique_labels = np.unique(labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))
        
        for label, color in zip(unique_labels, colors):
            if label == -1:
                # Noise points (for DBSCAN)
                color = 'gray'
                marker = 'x'
                label_name = 'Noise'
            else:
                marker = 'o'
                label_name = f'Cluster {label}'
            
            mask = labels == label
            ax.scatter(
                X_2d[mask, 0], X_2d[mask, 1],
                c=[color], marker=marker, s=50,
                alpha=0.6, edgecolors='black', linewidth=0.5,
                label=label_name
            )
        
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'Cluster Visualization (PCA) - {model_name}', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=10, loc='best')
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        path = self._get_plot_path('clusters_2d.png')
        plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return path
    
    def plot_cluster_sizes(
        self,
        labels: np.ndarray,
        model_name: str
    ) -> str:
        """
        Plot bar chart of cluster sizes.
        
        Returns:
            Path to saved plot
        """
        unique_labels, counts = np.unique(labels, return_counts=True)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = ['gray' if label == -1 else 'steelblue' for label in unique_labels]
        bars = ax.bar(range(len(unique_labels)), counts, color=colors, alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}',
                   ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel('Cluster', fontsize=12)
        ax.set_ylabel('Number of Samples', fontsize=12)
        ax.set_title(f'Cluster Sizes - {model_name}', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(unique_labels)))
        ax.set_xticklabels(['Noise' if l == -1 else f'Cluster {l}' for l in unique_labels])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        path = self._get_plot_path('cluster_sizes.png')
        plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return path
    
    # ==================== COMPARISON PLOTS ====================
    
    def plot_model_comparison(
        self,
        results: List[ModelResult],
        task_type: TaskType
    ) -> str:
        """
        Plot comparison of all models based on generalization score.
        
        Returns:
            Path to saved plot
        """
        model_names = [r.model_name for r in results]
        gen_scores = [r.generalization_score for r in results]
        cv_scores = [r.cv_score for r in results]
        test_scores = [r.test_score for r in results]
        rejected = [r.rejected for r in results]
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(model_names))
        width = 0.25
        
        # Bars
        bars1 = ax.bar(x - width, cv_scores, width, label='CV Score', 
                      color='steelblue', alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x, test_scores, width, label='Test Score', 
                      color='orange', alpha=0.8, edgecolor='black')
        bars3 = ax.bar(x + width, gen_scores, width, label='Generalization Score', 
                      color='green', alpha=0.8, edgecolor='black')
        
        # Mark rejected models
        for i, is_rejected in enumerate(rejected):
            if is_rejected:
                ax.text(i, -0.1, '❌ REJECTED', ha='center', va='top', 
                       color='red', fontweight='bold', fontsize=9)
        
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title(f'Model Comparison - {task_type.value.title()}', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=0, color='black', linewidth=0.5)
        
        plt.tight_layout()
        
        # Save
        path = self._get_plot_path('model_comparison.png')
        plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return path
    
    def plot_overfitting_analysis(
        self,
        results: List[ModelResult]
    ) -> str:
        """
        Plot overfitting gap analysis for all models.
        
        Returns:
            Path to saved plot
        """
        model_names = [r.model_name for r in results]
        overfit_gaps = [r.overfit_gap for r in results]
        rejected = [r.rejected for r in results]
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        colors = ['red' if r else 'steelblue' for r in rejected]
        bars = ax.barh(range(len(model_names)), overfit_gaps, color=colors, 
                       alpha=0.7, edgecolor='black')
        
        # Threshold lines
        ax.axvline(x=0.05, color='green', linestyle='--', lw=2, 
                  label='Low Overfitting (0.05)', alpha=0.7)
        ax.axvline(x=0.10, color='orange', linestyle='--', lw=2, 
                  label='High Penalty (0.10)', alpha=0.7)
        ax.axvline(x=0.20, color='red', linestyle='--', lw=2, 
                  label='Rejection Threshold (0.20)', alpha=0.7)
        
        ax.set_yticks(range(len(model_names)))
        ax.set_yticklabels(model_names)
        ax.set_xlabel('Overfit Gap (|Train Score - CV Score|)', fontsize=12)
        ax.set_title('Overfitting Analysis', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        path = self._get_plot_path('overfitting_analysis.png')
        plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
        return path
    
    # ==================== MAIN PLOTTING FUNCTION ====================
    
    def generate_all_plots(
        self,
        automl_result: AutoMLResult,
        X: np.ndarray,
        y: np.ndarray,
        X_test: Optional[np.ndarray] = None,
        y_test: Optional[np.ndarray] = None
    ) -> Dict[str, str]:
        """
        Generate all relevant plots based on task type.
        
        Args:
            automl_result: AutoML results
            X: Feature matrix (training data)
            y: Target vector (training data)
            X_test: Optional test feature matrix
            y_test: Optional test target vector
        
        Returns:
            Dictionary mapping plot names to file paths
        """
        plot_paths = {}
        best = automl_result.best_model
        
        # Always generate comparison plots
        plot_paths['model_comparison'] = self.plot_model_comparison(
            automl_result.all_models, 
            automl_result.task_type
        )
        plot_paths['overfitting_analysis'] = self.plot_overfitting_analysis(
            automl_result.all_models
        )
        
        # Task-specific plots
        if automl_result.task_type == TaskType.CLASSIFICATION:
            if X_test is not None and y_test is not None:
                y_pred = best.best_model.predict(X_test)
                
                # Confusion matrix
                plot_paths['confusion_matrix'] = self.plot_confusion_matrix(
                    y_test, y_pred, best.model_name
                )
                
                # ROC curve (if model supports predict_proba)
                if hasattr(best.best_model, 'predict_proba'):
                    try:
                        y_proba = best.best_model.predict_proba(X_test)
                        plot_paths['roc_curve'] = self.plot_roc_curve(
                            y_test, y_proba, best.model_name
                        )
                    except:
                        pass
            
            # Feature importance
            if best.feature_importance:
                plot_paths['feature_importance'] = self.plot_feature_importance(
                    best.feature_importance, best.model_name
                )
        
        elif automl_result.task_type == TaskType.REGRESSION:
            if X_test is not None and y_test is not None:
                y_pred = best.best_model.predict(X_test)
                
                # Predicted vs Actual
                plot_paths['predicted_vs_actual'] = self.plot_predicted_vs_actual(
                    y_test, y_pred, best.model_name
                )
                
                # Residuals
                plot_paths['residuals'] = self.plot_residuals(
                    y_test, y_pred, best.model_name
                )
                
                # Residual distribution
                plot_paths['residual_distribution'] = self.plot_residual_distribution(
                    y_test, y_pred, best.model_name
                )
            
            # Feature importance
            if best.feature_importance:
                plot_paths['feature_importance'] = self.plot_feature_importance(
                    best.feature_importance, best.model_name
                )
        
        elif automl_result.task_type == TaskType.CLUSTERING:
            labels = best.detailed_metrics.get('labels')
            if labels is not None:
                labels = np.array(labels)
                
                # 2D cluster visualization
                plot_paths['clusters_2d'] = self.plot_clusters_2d(
                    X, labels, best.model_name, automl_result.feature_names
                )
                
                # Cluster sizes
                plot_paths['cluster_sizes'] = self.plot_cluster_sizes(
                    labels, best.model_name
                )
        
        return plot_paths

    # ==================== UNSUPERVISED PLOTS ====================

    def plot_unsupervised_model_comparison(
        self,
        results: list,
    ) -> str:
        """Plot comparison of unsupervised models by primary score."""
        model_names = [r.model_name for r in results if not r.rejected]
        scores = [r.primary_score for r in results if not r.rejected]

        if not model_names:
            model_names = [r.model_name for r in results]
            scores = [r.primary_score for r in results]

        fig, ax = plt.subplots(figsize=(12, 6))

        colors = ['steelblue' if s > 0 else '#cc3333' for s in scores]
        bars = ax.bar(range(len(model_names)), scores, color=colors, alpha=0.8, edgecolor='black')

        for bar, score in zip(bars, scores):
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2., height + 0.01,
                f'{score:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold',
            )

        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Unsupervised Model Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        ax.axhline(y=0, color='black', linewidth=0.5)

        plt.tight_layout()
        path = self._get_plot_path('unsupervised_model_comparison.png')
        plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        return path

    def plot_silhouette_analysis(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        model_name: str,
    ) -> str:
        """Plot per-cluster silhouette analysis."""
        from sklearn.metrics import silhouette_samples
        mask = labels != -1
        if mask.sum() < 2 or len(np.unique(labels[mask])) < 2:
            return ""

        sil_values = silhouette_samples(X[mask], labels[mask])
        sil_avg = np.mean(sil_values)
        unique_labels = np.unique(labels[mask])

        fig, ax = plt.subplots(figsize=(12, 8))

        y_lower = 0
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

        for i, (label, color) in enumerate(zip(unique_labels, colors)):
            cluster_sil = sil_values[labels[mask] == label]
            cluster_sil.sort()

            y_upper = y_lower + len(cluster_sil)
            ax.fill_betweenx(
                np.arange(y_lower, y_upper), 0, cluster_sil,
                facecolor=color, edgecolor=color, alpha=0.7,
            )
            ax.text(-0.05, y_lower + 0.5 * len(cluster_sil), str(int(label)),
                    fontsize=10, fontweight='bold')
            y_lower = y_upper + 5

        ax.axvline(x=sil_avg, color='red', linestyle='--', lw=2,
                   label=f'Average ({sil_avg:.3f})')
        ax.set_xlabel('Silhouette Coefficient', fontsize=12)
        ax.set_ylabel('Cluster', fontsize=12)
        ax.set_title(f'Silhouette Analysis - {model_name}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='x', alpha=0.3)

        plt.tight_layout()
        path = self._get_plot_path('silhouette_analysis.png')
        plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        return path

    def plot_explained_variance(
        self,
        explained_variance_ratio: np.ndarray,
        model_name: str,
    ) -> str:
        """Plot explained variance ratio for PCA/SVD."""
        cumulative = np.cumsum(explained_variance_ratio)

        fig, ax = plt.subplots(figsize=(12, 6))

        x = range(1, len(explained_variance_ratio) + 1)
        ax.bar(x, explained_variance_ratio, alpha=0.6, color='steelblue',
               edgecolor='black', label='Individual')
        ax.step(x, cumulative, where='mid', color='red', lw=2,
                label='Cumulative')
        ax.axhline(y=0.95, color='green', linestyle='--', lw=1.5, alpha=0.7,
                   label='95% threshold')

        ax.set_xlabel('Component', fontsize=12)
        ax.set_ylabel('Explained Variance Ratio', fontsize=12)
        ax.set_title(f'Explained Variance - {model_name}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        path = self._get_plot_path('explained_variance.png')
        plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        return path

    def plot_anomaly_scatter(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        scores: np.ndarray,
        model_name: str,
    ) -> str:
        """Plot anomaly detection results on 2D PCA projection."""
        if X.shape[1] > 2:
            pca = PCA(n_components=2, random_state=42)
            X_2d = pca.fit_transform(X)
            xlabel = f'PC1 ({pca.explained_variance_ratio_[0]:.2%})'
            ylabel = f'PC2 ({pca.explained_variance_ratio_[1]:.2%})'
        else:
            X_2d = X
            xlabel, ylabel = 'Feature 1', 'Feature 2'

        fig, ax = plt.subplots(figsize=(12, 8))

        inlier_mask = labels == 1
        outlier_mask = labels == -1

        ax.scatter(X_2d[inlier_mask, 0], X_2d[inlier_mask, 1],
                   c='steelblue', s=30, alpha=0.5, label=f'Inliers ({inlier_mask.sum()})')
        ax.scatter(X_2d[outlier_mask, 0], X_2d[outlier_mask, 1],
                   c='red', s=60, alpha=0.8, marker='x', linewidths=2,
                   label=f'Outliers ({outlier_mask.sum()})')

        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f'Anomaly Detection - {model_name}', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        path = self._get_plot_path('anomaly_scatter.png')
        plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        return path

    def plot_anomaly_score_distribution(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        model_name: str,
    ) -> str:
        """Plot anomaly score distribution."""
        fig, ax = plt.subplots(figsize=(12, 6))

        inlier_scores = scores[labels == 1]
        outlier_scores = scores[labels == -1]

        if len(inlier_scores) > 0:
            ax.hist(inlier_scores, bins=30, alpha=0.6, color='steelblue',
                    edgecolor='black', label='Inliers')
        if len(outlier_scores) > 0:
            ax.hist(outlier_scores, bins=30, alpha=0.6, color='red',
                    edgecolor='black', label='Outliers')

        ax.set_xlabel('Anomaly Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'Anomaly Score Distribution - {model_name}',
                     fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        path = self._get_plot_path('anomaly_score_distribution.png')
        plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        return path

    def generate_unsupervised_plots(
        self,
        unsupervised_result,
        X: np.ndarray,
        feature_names: list = None,
    ) -> Dict[str, str]:
        """
        Generate all plots for unsupervised learning results.

        Args:
            unsupervised_result: UnsupervisedResult from UnsupervisedEngine
            X: Feature matrix
            feature_names: Feature names

        Returns:
            Dict mapping plot names to file paths
        """
        plot_paths = {}
        best = unsupervised_result.best_model

        # Model comparison (always)
        try:
            plot_paths['model_comparison'] = self.plot_unsupervised_model_comparison(
                unsupervised_result.all_models,
            )
        except Exception as e:
            print(f"⚠️ Failed to generate model comparison plot: {e}")

        task_subtype = unsupervised_result.task_subtype
        from .unsupervised_engine import UnsupervisedTaskType

        if task_subtype == UnsupervisedTaskType.CLUSTERING:
            if best.labels is not None:
                labels = best.labels

                # 2D PCA cluster scatter
                try:
                    plot_paths['clusters_2d'] = self.plot_clusters_2d(
                        X, labels, best.model_name, feature_names,
                    )
                except Exception as e:
                    print(f"⚠️ Failed to generate cluster 2d plot: {e}")

                # Cluster sizes
                try:
                    plot_paths['cluster_sizes'] = self.plot_cluster_sizes(
                        labels, best.model_name,
                    )
                except Exception as e:
                    print(f"⚠️ Failed to generate cluster sizes plot: {e}")

                # Silhouette analysis
                try:
                    path = self.plot_silhouette_analysis(X, labels, best.model_name)
                    if path:
                        plot_paths['silhouette_analysis'] = path
                except Exception as e:
                    print(f"⚠️ Failed to generate silhouette plot: {e}")

        elif task_subtype == UnsupervisedTaskType.DIMENSIONALITY_REDUCTION:
            if best.explained_variance_ratio is not None:
                try:
                    plot_paths['explained_variance'] = self.plot_explained_variance(
                        best.explained_variance_ratio, best.model_name,
                    )
                except Exception as e:
                    print(f"⚠️ Failed to generate explained variance plot: {e}")

            # Also make a 2D scatter of transformed data if available
            if best.transformed_data is not None and best.transformed_data.shape[1] >= 2:
                try:
                    fig, ax = plt.subplots(figsize=(12, 8))
                    ax.scatter(
                        best.transformed_data[:, 0], best.transformed_data[:, 1],
                        alpha=0.5, s=30, c='steelblue', edgecolors='black', linewidth=0.3,
                    )
                    ax.set_xlabel('Component 1', fontsize=12)
                    ax.set_ylabel('Component 2', fontsize=12)
                    ax.set_title(f'2D Projection - {best.model_name}', fontsize=14, fontweight='bold')
                    ax.grid(alpha=0.3)
                    plt.tight_layout()
                    path = self._get_plot_path('dimred_2d_scatter.png')
                    plt.savefig(path, dpi=self.dpi, bbox_inches='tight')
                    plt.close()
                    plot_paths['dimred_2d_scatter'] = path
                except Exception as e:
                    print(f"⚠️ Failed to generate dimred scatter: {e}")

        elif task_subtype == UnsupervisedTaskType.ANOMALY_DETECTION:
            if best.anomaly_labels is not None:
                labels = best.anomaly_labels
                scores = best.anomaly_scores if best.anomaly_scores is not None else np.zeros(len(labels))

                # Anomaly scatter
                try:
                    plot_paths['anomaly_scatter'] = self.plot_anomaly_scatter(
                        X, labels, scores, best.model_name,
                    )
                except Exception as e:
                    print(f"⚠️ Failed to generate anomaly scatter: {e}")

                # Score distribution
                try:
                    plot_paths['anomaly_score_distribution'] = self.plot_anomaly_score_distribution(
                        scores, labels, best.model_name,
                    )
                except Exception as e:
                    print(f"⚠️ Failed to generate anomaly score dist: {e}")

        return plot_paths