"""
Quick Test Script for AutoML Engine
=====================================

This script demonstrates the AutoML engine with a sample classification task.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import sys
import os

# Add app to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'app'))

from services.automl_engine import AutoMLEngine, TaskType
from services.automl_plots import AutoMLPlotter


def test_classification():
    """Test AutoML with a classification dataset"""
    print("=" * 80)
    print("AUTOML ENGINE TEST - CLASSIFICATION")
    print("=" * 80)
    
    # Generate synthetic classification data
    print("\n1. Generating synthetic classification dataset...")
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=3,
        random_state=42
    )
    
    feature_names = [f"feature_{i}" for i in range(20)]
    print(f"   ✅ Generated {X.shape[0]} samples with {X.shape[1]} features")
    print(f"   ✅ Classes: {np.unique(y)}")
    
    # Initialize AutoML engine
    print("\n2. Initializing AutoML engine...")
    engine = AutoMLEngine(
        task_type=TaskType.CLASSIFICATION,
        n_trials=30,  # Reduced for quick testing
        cv_folds=3,   # Reduced for quick testing
        test_size=0.2,
        penalty_factor=2.0,
        overfit_threshold_reject=0.20,
        overfit_threshold_high=0.10,
        verbose=True
    )
    print("   ✅ Engine initialized")
    
    # Train models - test with just a few models
    print("\n3. Training AutoML models...")
    result = engine.fit(
        X=X,
        y=y,
        feature_names=feature_names,
        model_subset=['logistic_regression', 'random_forest', 'xgboost']
    )
    
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    
    print(f"\n🏆 Best Model: {result.best_model.model_name}")
    print(f"   Model Type: {result.best_model.model_type}")
    print(f"\n📊 Scores:")
    print(f"   Train Score:              {result.best_model.train_score:.4f}")
    print(f"   CV Score:                 {result.best_model.cv_score:.4f} (±{result.best_model.cv_std:.4f})")
    print(f"   Test Score:               {result.best_model.test_score:.4f}")
    print(f"\n🎯 Generalization Metrics:")
    print(f"   Overfit Gap:              {result.best_model.overfit_gap:.4f}")
    print(f"   Penalty:                  {result.best_model.penalty:.4f}")
    print(f"   Generalization Score:     {result.best_model.generalization_score:.4f}")
    print(f"   Overfitting Detected:     {result.best_model.overfitting}")
    print(f"   Model Rejected:           {result.best_model.rejected}")
    
    print(f"\n📈 All Models Evaluated:")
    print(f"   Total:                    {result.total_models_evaluated}")
    print(f"   Rejected:                 {result.models_rejected}")
    
    # Show comparison
    print("\n" + "-" * 80)
    print("MODEL COMPARISON TABLE")
    print("-" * 80)
    print(f"{'Model':<20} {'Train':<8} {'CV':<8} {'Test':<8} {'Gap':<8} {'Gen Score':<10} {'Status':<10}")
    print("-" * 80)
    
    for model in result.all_models:
        status = "REJECTED" if model.rejected else "✅ Valid"
        marker = "🏆" if model.model_name == result.best_model.model_name else "  "
        print(f"{marker} {model.model_name:<18} {model.train_score:.4f}   {model.cv_score:.4f}   "
              f"{model.test_score:.4f}   {model.overfit_gap:.4f}   {model.generalization_score:.4f}     {status}")
    
    print("-" * 80)
    
    # Generate plots
    print("\n4. Generating evaluation plots...")
    os.makedirs("test_plots", exist_ok=True)
    plotter = AutoMLPlotter(output_dir="test_plots")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    plot_paths = plotter.generate_all_plots(
        automl_result=result,
        X=X_train,
        y=y_train,
        X_test=X_test,
        y_test=y_test
    )
    
    print(f"   ✅ Generated {len(plot_paths)} plots:")
    for name, path in plot_paths.items():
        print(f"      - {name}: {path}")
    
    # Save model
    print("\n5. Saving model...")
    os.makedirs("test_models", exist_ok=True)
    model_path = "test_models/test_automl_model.joblib"
    engine.save(model_path)
    print(f"   ✅ Model saved to: {model_path}")
    
    # Test loading
    print("\n6. Testing model loading...")
    loaded_engine = AutoMLEngine.load(model_path)
    predictions = loaded_engine.predict(X_test[:5])
    print(f"   ✅ Model loaded successfully")
    print(f"   ✅ Sample predictions: {predictions}")
    
    print("\n" + "=" * 80)
    print("✅ TEST COMPLETED SUCCESSFULLY!")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("1. Model selection is based on GENERALIZATION SCORE, not raw test score")
    print("2. Models with high overfit gaps receive penalties")
    print("3. Models with gap > 0.20 are completely rejected")
    print("4. The best model balances performance with generalization")
    print("\nCheck the 'test_plots' directory for visualizations!")


def test_regression():
    """Test AutoML with a regression dataset"""
    print("\n\n" + "=" * 80)
    print("AUTOML ENGINE TEST - REGRESSION")
    print("=" * 80)
    
    from sklearn.datasets import make_regression
    
    print("\n1. Generating synthetic regression dataset...")
    X, y = make_regression(
        n_samples=1000,
        n_features=15,
        n_informative=10,
        random_state=42,
        noise=0.1
    )
    
    feature_names = [f"feature_{i}" for i in range(15)]
    print(f"   ✅ Generated {X.shape[0]} samples with {X.shape[1]} features")
    
    print("\n2. Training AutoML models for regression...")
    engine = AutoMLEngine(
        task_type=TaskType.REGRESSION,
        n_trials=30,
        cv_folds=3,
        test_size=0.2,
        verbose=True
    )
    
    result = engine.fit(
        X=X,
        y=y,
        feature_names=feature_names,
        model_subset=['ridge', 'random_forest', 'xgboost']
    )
    
    print("\n" + "=" * 80)
    print("REGRESSION RESULTS")
    print("=" * 80)
    print(f"\n🏆 Best Model: {result.best_model.model_name}")
    print(f"   R² (CV):                  {result.best_model.cv_score:.4f}")
    print(f"   R² (Test):                {result.best_model.test_score:.4f}")
    print(f"   Generalization Score:     {result.best_model.generalization_score:.4f}")
    
    # Generate plots
    print("\n3. Generating regression plots...")
    plotter = AutoMLPlotter(output_dir="test_plots")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    plot_paths = plotter.generate_all_plots(
        automl_result=result,
        X=X_train,
        y=y_train,
        X_test=X_test,
        y_test=y_test
    )
    
    print(f"   ✅ Generated {len(plot_paths)} plots")
    
    print("\n✅ REGRESSION TEST COMPLETED!")


if __name__ == "__main__":
    try:
        # Test classification
        test_classification()
        
        # Test regression
        test_regression()
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
