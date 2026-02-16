"""
Example: Adaptive Preprocessing Pipeline Demo
Demonstrates intelligent preprocessing with different datasets and scenarios
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from app.services.adaptive_preprocessing import AdaptivePreprocessor
from app.services.eda_service import EDAService
from typing import Dict, Any
import json


def print_section(title: str):
    """Print formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def print_decision_summary(results: Dict[str, Any]):
    """Print decision log summary with color coding"""
    print("\n📋 Decision Log Summary:")
    print("-" * 80)
    
    impact_colors = {
        'info': '✅',
        'warning': '⚠️',
        'critical': '🔴'
    }
    
    for decision in results['decision_log']:
        icon = impact_colors.get(decision['impact'], '•')
        print(f"{icon} [{decision['category'].upper()}]")
        print(f"   Decision: {decision['decision']}")
        print(f"   Reason: {decision['reason']}")
        print()


def print_quality_metrics(results: Dict[str, Any]):
    """Print quality metrics and preprocessing summary"""
    metrics = results['quality_metrics']
    metadata = results['preprocessing_metadata']
    
    print("\n📊 Preprocessing Summary:")
    print("-" * 80)
    print(f"Quality Score: {metrics['quality_score']:.1f}/100")
    print(f"Task Type: {results['task_type']}")
    print(f"Model Family: {results['model_family']}")
    print()
    print(f"Data Shape: {metadata['initial_shape']} → {metadata['final_shape']}")
    print(f"Rows Removed: {metrics['rows_removed']}")
    print(f"Columns Removed: {metrics['columns_removed']}")
    print(f"Features Generated: {metrics['features_generated']}")
    print(f"Data Retention: {metrics['data_retention_ratio']*100:.1f}%")
    print()
    print(f"Feature Types:")
    print(f"  - Numerical: {len(metadata['numerical_features'])}")
    print(f"  - Categorical: {len(metadata['categorical_features'])}")
    print(f"  - Datetime: {len(metadata['datetime_features'])}")
    print()
    
    if metadata.get('id_columns_removed'):
        print(f"ID Columns Removed: {metadata['id_columns_removed']}")
    if metadata.get('constant_columns_removed'):
        print(f"Constant Columns Removed: {metadata['constant_columns_removed']}")
    if metadata.get('high_missing_columns_removed'):
        print(f"High Missing Columns Removed: {metadata['high_missing_columns_removed']}")


def example_1_classification_with_eda():
    """
    Example 1: Classification with EDA insights
    Dataset: Iris (classic classification problem)
    """
    print_section("Example 1: Classification with Tree-Based Model")
    
    # Create synthetic Iris-like dataset
    print("📁 Loading dataset: Iris (classification)")
    np.random.seed(42)
    
    df = pd.DataFrame({
        'id': range(1, 151),  # ID column (should be removed)
        'sepal_length': np.random.normal(5.8, 0.8, 150),
        'sepal_width': np.random.normal(3.0, 0.4, 150),
        'petal_length': np.random.normal(3.8, 1.8, 150),
        'petal_width': np.random.normal(1.2, 0.8, 150),
        'species': np.random.choice(['setosa', 'versicolor', 'virginica'], 150)
    })
    
    # Add some missing values
    df.loc[10:15, 'sepal_width'] = np.nan
    
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print()
    
    # Step 1: Run EDA
    print("🔍 Running EDA analysis...")
    eda_service = EDAService()
    eda_results = eda_service.perform_eda(df)
    print(f"EDA complete - Quality score: {eda_results['data_quality']['quality_score']:.1f}/100")
    
    # Step 2: Adaptive preprocessing
    print("\n🔧 Running adaptive preprocessing...")
    preprocessor = AdaptivePreprocessor(eda_results=eda_results)
    
    results = preprocessor.fit_transform(
        df=df,
        target_column='species',
        model_type='random_forest',  # Tree-based model
        test_size=0.2
    )
    
    # Display results
    print_quality_metrics(results)
    print_decision_summary(results)
    
    # Show train/test split
    print("\n📦 Data Split:")
    print(f"Training set: {results['X_train'].shape}")
    print(f"Test set: {results['X_test'].shape}")
    print(f"Feature names: {results['feature_names']}")
    
    return preprocessor, results


def example_2_regression_with_outliers():
    """
    Example 2: Regression with outliers
    Dataset: House prices with outliers
    """
    print_section("Example 2: Regression with Outliers (Linear Model)")
    
    # Create synthetic house price dataset with outliers
    print("📁 Creating dataset: House Prices (regression)")
    np.random.seed(42)
    
    n_samples = 200
    df = pd.DataFrame({
        'square_feet': np.random.normal(2000, 500, n_samples),
        'bedrooms': np.random.randint(1, 6, n_samples),
        'bathrooms': np.random.randint(1, 4, n_samples),
        'age_years': np.random.randint(0, 50, n_samples),
        'location': np.random.choice(['urban', 'suburban', 'rural'], n_samples),
        'price': np.random.normal(300000, 100000, n_samples)
    })
    
    # Add outliers to square_feet and price
    outlier_indices = np.random.choice(n_samples, 20, replace=False)
    df.loc[outlier_indices, 'square_feet'] *= 3  # Extreme outliers
    df.loc[outlier_indices, 'price'] *= 2
    
    # Add missing values
    missing_indices = np.random.choice(n_samples, 15, replace=False)
    df.loc[missing_indices, 'bathrooms'] = np.nan
    
    print(f"Dataset shape: {df.shape}")
    print(f"Outliers added: {len(outlier_indices)} samples")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print()
    
    # Step 1: EDA
    print("🔍 Running EDA analysis...")
    eda_service = EDAService()
    eda_results = eda_service.perform_eda(df)
    
    outlier_info = eda_results['outliers']['outliers_by_column']
    print(f"EDA detected outliers in {len(outlier_info)} columns")
    for col, info in outlier_info.items():
        print(f"  - {col}: {info['count']} outliers ({info['percentage']:.1f}%)")
    
    # Step 2: Adaptive preprocessing
    print("\n🔧 Running adaptive preprocessing...")
    preprocessor = AdaptivePreprocessor(eda_results=eda_results)
    
    results = preprocessor.fit_transform(
        df=df,
        target_column='price',
        model_type='linear_regression',  # Linear model (should use robust scaling)
        test_size=0.2
    )
    
    # Display results
    print_quality_metrics(results)
    print_decision_summary(results)
    
    return preprocessor, results


def example_3_unsupervised_clustering():
    """
    Example 3: Unsupervised learning (clustering)
    Dataset: Customer segmentation
    """
    print_section("Example 3: Unsupervised Learning (Clustering)")
    
    # Create synthetic customer dataset
    print("📁 Creating dataset: Customer Segmentation (unsupervised)")
    np.random.seed(42)
    
    n_customers = 300
    df = pd.DataFrame({
        'customer_id': [f'CUST_{i:05d}' for i in range(n_customers)],  # ID column
        'age': np.random.randint(18, 75, n_customers),
        'income': np.random.normal(50000, 20000, n_customers),
        'spending_score': np.random.randint(1, 100, n_customers),
        'loyalty_years': np.random.randint(0, 15, n_customers),
        'preferred_category': np.random.choice(['electronics', 'fashion', 'home', 'sports'], n_customers),
        'membership_tier': np.random.choice(['bronze', 'silver', 'gold', 'platinum'], n_customers),
    })
    
    print(f"Dataset shape: {df.shape}")
    print(f"No target column (unsupervised)")
    print()
    
    # Step 1: EDA
    print("🔍 Running EDA analysis...")
    eda_service = EDAService()
    eda_results = eda_service.perform_eda(df)
    
    # Step 2: Adaptive preprocessing WITHOUT target column
    print("\n🔧 Running adaptive preprocessing (unsupervised)...")
    preprocessor = AdaptivePreprocessor(eda_results=eda_results)
    
    results = preprocessor.fit_transform(
        df=df,
        target_column=None,  # No target for unsupervised
        model_type=None,  # No specific model
        test_size=0  # No train/test split
    )
    
    # Display results
    print_quality_metrics(results)
    print_decision_summary(results)
    
    print("\n🎯 Ready for clustering:")
    print(f"Features shape: {results['X_train'].shape}")
    print(f"Can now use: KMeans, DBSCAN, Hierarchical Clustering, etc.")
    
    return preprocessor, results


def example_4_high_missing_data():
    """
    Example 4: Dataset with high missing data
    Demonstrates intelligent column dropping
    """
    print_section("Example 4: High Missing Data Handling")
    
    # Create dataset with varying missing rates
    print("📁 Creating dataset with high missing data")
    np.random.seed(42)
    
    n_samples = 100
    df = pd.DataFrame({
        'good_feature_1': np.random.normal(0, 1, n_samples),
        'good_feature_2': np.random.choice(['A', 'B', 'C'], n_samples),
        'medium_missing': np.random.normal(0, 1, n_samples),  # 35% missing
        'high_missing': np.random.normal(0, 1, n_samples),  # 70% missing
        'constant_feature': ['constant'] * n_samples,  # Constant column
        'target': np.random.choice([0, 1], n_samples)
    })
    
    # Add missing values
    medium_missing_idx = np.random.choice(n_samples, 35, replace=False)
    df.loc[medium_missing_idx, 'medium_missing'] = np.nan
    
    high_missing_idx = np.random.choice(n_samples, 70, replace=False)
    df.loc[high_missing_idx, 'high_missing'] = np.nan
    
    print(f"Dataset shape: {df.shape}")
    print("\nMissing value rates:")
    for col in df.columns:
        missing_pct = df[col].isna().sum() / len(df) * 100
        print(f"  - {col}: {missing_pct:.1f}% missing")
    print()
    
    # Step 1: EDA
    print("🔍 Running EDA analysis...")
    eda_service = EDAService()
    eda_results = eda_service.perform_eda(df)
    
    # Step 2: Adaptive preprocessing
    print("\n🔧 Running adaptive preprocessing...")
    preprocessor = AdaptivePreprocessor(eda_results=eda_results)
    
    # Custom threshold for demo
    preprocessor.MISSING_DROP_THRESHOLD = 0.6  # Drop if >60% missing
    
    results = preprocessor.fit_transform(
        df=df,
        target_column='target',
        model_type='logistic_regression',
        test_size=0.2
    )
    
    # Display results
    print_quality_metrics(results)
    print_decision_summary(results)
    
    print("\n💡 Notice:")
    print("  - 'high_missing' was dropped (>60% missing)")
    print("  - 'constant_feature' was dropped (no variance)")
    print("  - 'medium_missing' was kept and imputed (35% missing)")
    
    return preprocessor, results


def example_5_comparison_linear_vs_tree():
    """
    Example 5: Compare preprocessing for different model families
    Same dataset, different model types
    """
    print_section("Example 5: Preprocessing Comparison (Linear vs Tree-Based)")
    
    # Create dataset
    print("📁 Creating dataset for comparison")
    np.random.seed(42)
    
    df = pd.DataFrame({
        'num_feature_1': np.random.normal(100, 50, 200),
        'num_feature_2': np.random.normal(0, 1, 200),
        'cat_feature_1': np.random.choice(['A', 'B', 'C', 'D', 'E'], 200),
        'cat_feature_2': np.random.choice(['X', 'Y'], 200),
        'target': np.random.choice(['class_0', 'class_1'], 200)
    })
    
    print(f"Dataset shape: {df.shape}")
    print()
    
    # EDA once
    eda_service = EDAService()
    eda_results = eda_service.perform_eda(df)
    
    # --- Scenario A: Linear Model ---
    print("\n🔵 Scenario A: Logistic Regression (Linear Model)")
    print("-" * 80)
    
    preprocessor_linear = AdaptivePreprocessor(eda_results=eda_results)
    results_linear = preprocessor_linear.fit_transform(
        df=df.copy(),
        target_column='target',
        model_type='logistic_regression',
        test_size=0.2
    )
    
    print(f"\nLinear Model Decisions:")
    linear_decisions = [d for d in results_linear['decision_log'] if d['category'] in ['scaling', 'encoding']]
    for d in linear_decisions:
        print(f"  - {d['category']}: {d['decision']}")
    
    print(f"\nFeatures generated: {results_linear['X_train'].shape[1]}")
    
    # --- Scenario B: Tree-Based Model ---
    print("\n\n🟢 Scenario B: Random Forest (Tree-Based Model)")
    print("-" * 80)
    
    preprocessor_tree = AdaptivePreprocessor(eda_results=eda_results)
    results_tree = preprocessor_tree.fit_transform(
        df=df.copy(),
        target_column='target',
        model_type='random_forest',
        test_size=0.2
    )
    
    print(f"\nTree-Based Model Decisions:")
    tree_decisions = [d for d in results_tree['decision_log'] if d['category'] in ['scaling', 'encoding']]
    for d in tree_decisions:
        print(f"  - {d['category']}: {d['decision']}")
    
    print(f"\nFeatures generated: {results_tree['X_train'].shape[1]}")
    
    # Comparison
    print("\n\n📊 Comparison:")
    print("-" * 80)
    print(f"Linear Model (Logistic Regression):")
    print(f"  - Scaling: ✅ StandardScaler applied")
    print(f"  - Encoding: OneHotEncoder (creates more features)")
    print(f"  - Features: {results_linear['X_train'].shape[1]}")
    print()
    print(f"Tree-Based Model (Random Forest):")
    print(f"  - Scaling: ❌ No scaling (not needed)")
    print(f"  - Encoding: OrdinalEncoder (keeps features compact)")
    print(f"  - Features: {results_tree['X_train'].shape[1]}")
    print()
    print("💡 Key Insight: Same data, different preprocessing based on model type!")


def save_preprocessing_report(preprocessor, results, filename="preprocessing_report.json"):
    """Save preprocessing report to file"""
    report = {
        "summary": preprocessor.get_preprocessing_summary(),
        "quality_metrics": results['quality_metrics'],
        "preprocessing_metadata": results['preprocessing_metadata'],
        "decision_log": results['decision_log']
    }
    
    with open(filename, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n💾 Report saved to: {filename}")


def main():
    """Run all examples"""
    print("\n" + "🎯" * 40)
    print("ADAPTIVE PREPROCESSING PIPELINE - DEMONSTRATION")
    print("🎯" * 40)
    
    examples = [
        ("Classification (Tree-Based)", example_1_classification_with_eda),
        ("Regression with Outliers", example_2_regression_with_outliers),
        ("Unsupervised Learning", example_3_unsupervised_clustering),
        ("High Missing Data", example_4_high_missing_data),
        ("Linear vs Tree Comparison", example_5_comparison_linear_vs_tree),
    ]
    
    print("\nAvailable Examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\n" + "-" * 80)
    choice = input("\nSelect example (1-5, or 'all' to run all): ").strip().lower()
    
    if choice == 'all':
        for name, example_func in examples:
            try:
                preprocessor, results = example_func()
                input("\nPress Enter to continue to next example...")
            except Exception as e:
                print(f"\n❌ Error in example '{name}': {e}")
                import traceback
                traceback.print_exc()
    elif choice.isdigit() and 1 <= int(choice) <= len(examples):
        idx = int(choice) - 1
        name, example_func = examples[idx]
        preprocessor, results = example_func()
        
        # Offer to save report
        save = input("\nSave preprocessing report? (y/n): ").strip().lower()
        if save == 'y':
            filename = f"report_{name.lower().replace(' ', '_')}.json"
            save_preprocessing_report(preprocessor, results, filename)
    else:
        print("Invalid choice. Exiting.")
    
    print("\n" + "🎯" * 40)
    print("DEMONSTRATION COMPLETE")
    print("🎯" * 40 + "\n")


if __name__ == "__main__":
    main()
