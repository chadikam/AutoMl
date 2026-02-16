"""
Adaptive Preprocessing Demo
============================

This script demonstrates the adaptive preprocessing pipeline in action with:
1. Multiple dataset types (classification, regression, unsupervised)
2. Different model families (linear, tree-based)
3. Various data quality issues (missing values, outliers, ID columns)
4. Comprehensive logging and reporting
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from app.services.adaptive_preprocessing import AdaptivePreprocessor
from app.services.eda_service import EDAService

# Set random seed for reproducibility
np.random.seed(42)


def create_sample_classification_dataset():
    """Create a sample classification dataset with various data issues"""
    n_samples = 1000
    
    df = pd.DataFrame({
        # ID column (should be dropped)
        'customer_id': range(1, n_samples + 1),
        
        # Numerical features
        'age': np.random.normal(40, 12, n_samples),
        'income': np.random.lognormal(11, 0.5, n_samples),  # Has outliers
        'credit_score': np.random.normal(680, 80, n_samples),
        'years_employed': np.random.exponential(5, n_samples),
        
        # Categorical features
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples, p=[0.4, 0.35, 0.2, 0.05]),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_samples),
        'job_category': np.random.choice(['Tech', 'Finance', 'Healthcare', 'Education', 'Other'], n_samples),
        
        # Constant column (should be dropped)
        'country': 'USA',
        
        # High missing column (should be dropped)
        'optional_field': np.random.choice([np.nan, 'Value'], n_samples, p=[0.7, 0.3]),
        
        # Target variable
        'churn': np.random.choice(['No', 'Yes'], n_samples, p=[0.7, 0.3])
    })
    
    # Introduce some missing values in numerical columns
    df.loc[np.random.choice(df.index, 50), 'age'] = np.nan
    df.loc[np.random.choice(df.index, 30), 'credit_score'] = np.nan
    
    # Introduce missing values in categorical columns
    df.loc[np.random.choice(df.index, 40), 'education'] = np.nan
    
    return df


def create_sample_regression_dataset():
    """Create a sample regression dataset"""
    n_samples = 800
    
    df = pd.DataFrame({
        # ID column
        'house_id': [f'H{i:05d}' for i in range(1, n_samples + 1)],
        
        # Numerical features
        'sqft': np.random.normal(2000, 800, n_samples),
        'bedrooms': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.05, 0.25, 0.40, 0.25, 0.05]),
        'bathrooms': np.random.choice([1, 1.5, 2, 2.5, 3], n_samples, p=[0.10, 0.20, 0.40, 0.20, 0.10]),
        'age_years': np.random.exponential(15, n_samples),
        'lot_size': np.random.lognormal(8, 0.5, n_samples),
        
        # Categorical features
        'location': np.random.choice(['Downtown', 'Suburb', 'Rural'], n_samples, p=[0.3, 0.5, 0.2]),
        'style': np.random.choice(['Modern', 'Traditional', 'Colonial', 'Ranch'], n_samples),
        
        # Constant
        'status': 'Active',
        
        # Target (price)
        'price': 100000 + np.random.lognormal(12, 0.5, n_samples)
    })
    
    # Add missing values
    df.loc[np.random.choice(df.index, 40), 'age_years'] = np.nan
    df.loc[np.random.choice(df.index, 25), 'bathrooms'] = np.nan
    
    return df


def create_sample_unsupervised_dataset():
    """Create a sample dataset for unsupervised learning (clustering)"""
    n_samples = 500
    
    df = pd.DataFrame({
        'customer_id': [f'C{i:06d}' for i in range(1, n_samples + 1)],
        
        # Purchase behavior
        'total_purchases': np.random.poisson(15, n_samples),
        'avg_purchase_value': np.random.lognormal(4, 0.8, n_samples),
        'days_since_last_purchase': np.random.exponential(30, n_samples),
        
        # Demographics
        'age': np.random.normal(35, 15, n_samples),
        'income_bracket': np.random.choice(['Low', 'Medium', 'High'], n_samples, p=[0.3, 0.5, 0.2]),
        
        # Engagement
        'website_visits': np.random.poisson(8, n_samples),
        'email_opens': np.random.poisson(5, n_samples),
        
        # Constant
        'version': 'v2'
    })
    
    # Add missing values
    df.loc[np.random.choice(df.index, 30), 'income_bracket'] = np.nan
    
    return df


def print_section(title, width=80):
    """Print a formatted section header"""
    print("\n" + "=" * width)
    print(f" {title} ".center(width, "="))
    print("=" * width + "\n")


def run_classification_demo():
    """Demonstrate classification preprocessing"""
    print_section("CLASSIFICATION TASK DEMO: Customer Churn Prediction")
    
    # Create dataset
    df = create_sample_classification_dataset()
    print(f"📊 Dataset shape: {df.shape}")
    print(f"📋 Columns: {list(df.columns)}")
    print(f"🎯 Target: 'churn'")
    print(f"🏷️  Target distribution:\n{df['churn'].value_counts()}\n")
    
    # Run EDA
    print("🔍 Running EDA analysis...")
    eda_service = EDAService()
    eda_results = eda_service.perform_eda(df)
    print("✅ EDA complete\n")
    
    # Initialize preprocessor
    print("🤖 Initializing Adaptive Preprocessor...")
    preprocessor = AdaptivePreprocessor(eda_results=eda_results)
    
    # Test 1: Random Forest (Tree-Based)
    print("\n" + "-" * 80)
    print("TEST 1: Random Forest (Tree-Based Model)")
    print("-" * 80)
    
    results_rf = preprocessor.fit_transform(
        df=df,
        target_column='churn',
        model_type='random_forest',
        test_size=0.2
    )
    
    print(f"\n✅ Preprocessing Complete!")
    print(f"   Training set: {results_rf['X_train'].shape}")
    print(f"   Test set: {results_rf['X_test'].shape}")
    print(f"   Features: {len(results_rf['feature_names'])}")
    print(f"   Quality Score: {results_rf['quality_metrics']['quality_score']:.1f}/100")
    
    # Show decision log
    print("\n📝 Key Decisions Made:")
    for decision in results_rf['decision_log'][:10]:  # Show first 10
        print(f"   [{decision['impact'].upper():8}] {decision['category']:25} | {decision['decision']}")
    
    print(f"\n   ... and {len(results_rf['decision_log']) - 10} more decisions")
    
    # Test 2: Logistic Regression (Linear Model)
    print("\n" + "-" * 80)
    print("TEST 2: Logistic Regression (Linear Model)")
    print("-" * 80)
    
    preprocessor2 = AdaptivePreprocessor(eda_results=eda_results)
    results_lr = preprocessor2.fit_transform(
        df=df,
        target_column='churn',
        model_type='logistic_regression',
        test_size=0.2
    )
    
    print(f"\n✅ Preprocessing Complete!")
    print(f"   Training set: {results_lr['X_train'].shape}")
    print(f"   Features: {len(results_lr['feature_names'])}")
    print(f"   Quality Score: {results_lr['quality_metrics']['quality_score']:.1f}/100")
    
    # Compare strategies
    print("\n📊 Strategy Comparison:")
    print(f"   Model Family: RF={results_rf['preprocessing_metadata']['model_family']} vs LR={results_lr['preprocessing_metadata']['model_family']}")
    
    # Check scaling difference
    rf_decisions = {d['category']: d['decision'] for d in results_rf['decision_log']}
    lr_decisions = {d['category']: d['decision'] for d in results_lr['decision_log']}
    
    print(f"   Scaling: RF={rf_decisions.get('scaling', 'N/A')}")
    print(f"   Scaling: LR={lr_decisions.get('scaling', 'N/A')}")
    
    return results_rf


def run_regression_demo():
    """Demonstrate regression preprocessing"""
    print_section("REGRESSION TASK DEMO: House Price Prediction")
    
    # Create dataset
    df = create_sample_regression_dataset()
    print(f"📊 Dataset shape: {df.shape}")
    print(f"📋 Columns: {list(df.columns)}")
    print(f"🎯 Target: 'price'")
    print(f"💰 Price range: ${df['price'].min():,.0f} - ${df['price'].max():,.0f}\n")
    
    # Run EDA
    print("🔍 Running EDA analysis...")
    eda_service = EDAService()
    eda_results = eda_service.perform_eda(df)
    print("✅ EDA complete\n")
    
    # Initialize preprocessor
    print("🤖 Initializing Adaptive Preprocessor...")
    preprocessor = AdaptivePreprocessor(eda_results=eda_results)
    
    # Preprocess for linear regression
    print("\n" + "-" * 80)
    print("TEST: Linear Regression")
    print("-" * 80)
    
    results = preprocessor.fit_transform(
        df=df,
        target_column='price',
        model_type='linear_regression',
        test_size=0.25
    )
    
    print(f"\n✅ Preprocessing Complete!")
    print(f"   Training set: {results['X_train'].shape}")
    print(f"   Test set: {results['X_test'].shape}")
    print(f"   Features: {len(results['feature_names'])}")
    print(f"   Quality Score: {results['quality_metrics']['quality_score']:.1f}/100")
    
    # Show preprocessing summary
    metadata = results['preprocessing_metadata']
    print(f"\n📊 Preprocessing Summary:")
    print(f"   Task Type: {results['task_type']}")
    print(f"   Model Family: {results['model_family']}")
    print(f"   Numerical features: {metadata['numerical_features']}")
    print(f"   Categorical features: {metadata['categorical_features']}")
    print(f"   ID columns removed: {metadata['id_columns_removed']}")
    print(f"   Constant columns removed: {metadata['constant_columns_removed']}")
    print(f"   Total columns removed: {metadata['total_columns_removed']}")
    
    return results


def run_unsupervised_demo():
    """Demonstrate unsupervised learning preprocessing"""
    print_section("UNSUPERVISED LEARNING DEMO: Customer Segmentation")
    
    # Create dataset
    df = create_sample_unsupervised_dataset()
    print(f"📊 Dataset shape: {df.shape}")
    print(f"📋 Columns: {list(df.columns)}")
    print(f"🎯 Target: None (unsupervised)\n")
    
    # Run EDA
    print("🔍 Running EDA analysis...")
    eda_service = EDAService()
    eda_results = eda_service.perform_eda(df)
    print("✅ EDA complete\n")
    
    # Initialize preprocessor
    print("🤖 Initializing Adaptive Preprocessor...")
    preprocessor = AdaptivePreprocessor(eda_results=eda_results)
    
    # Preprocess without target
    print("\n" + "-" * 80)
    print("TEST: Unsupervised Preprocessing (No Target)")
    print("-" * 80)
    
    results = preprocessor.fit_transform(
        df=df,
        target_column=None,  # Unsupervised
        model_type=None,
        test_size=0  # No split for unsupervised
    )
    
    print(f"\n✅ Preprocessing Complete!")
    print(f"   Dataset: {results['X_train'].shape}")
    print(f"   Features: {len(results['feature_names'])}")
    print(f"   Quality Score: {results['quality_metrics']['quality_score']:.1f}/100")
    
    # Show summary
    print(f"\n📊 Preprocessing Summary:")
    print(f"   Task Type: {results['task_type']}")
    print(f"   Model Family: {results['model_family']}")
    print(f"   Decisions Made: {len(results['decision_log'])}")
    
    # Show decision categories
    from collections import Counter
    categories = Counter(d['category'] for d in results['decision_log'])
    print(f"\n   Decisions by Category:")
    for cat, count in categories.most_common():
        print(f"      {cat}: {count}")
    
    return results


def demonstrate_pipeline_persistence():
    """Demonstrate saving and loading preprocessing pipelines"""
    print_section("PIPELINE PERSISTENCE DEMO")
    
    # Create and preprocess dataset
    df = create_sample_classification_dataset()
    eda_service = EDAService()
    eda_results = eda_service.perform_eda(df)
    
    # Fit preprocessor
    print("🔧 Fitting preprocessor...")
    preprocessor = AdaptivePreprocessor(eda_results=eda_results)
    results = preprocessor.fit_transform(
        df=df,
        target_column='churn',
        model_type='random_forest',
        test_size=0.2
    )
    
    # Save pipeline
    save_path = 'temp_preprocessing_pipeline.pkl'
    print(f"💾 Saving pipeline to {save_path}...")
    preprocessor.save_pipeline(save_path)
    print("✅ Pipeline saved\n")
    
    # Load pipeline
    print(f"📂 Loading pipeline from {save_path}...")
    new_preprocessor = AdaptivePreprocessor()
    new_preprocessor.load_pipeline(save_path)
    print("✅ Pipeline loaded\n")
    
    # Transform new data
    print("🔄 Transforming new data with loaded pipeline...")
    new_df = create_sample_classification_dataset()
    new_df = new_df.drop(columns=['churn'])  # Remove target
    
    X_new = new_preprocessor.transform(new_df)
    print(f"✅ Transformed: {X_new.shape}")
    print(f"   Features match: {X_new.shape[1] == results['X_train'].shape[1]}")
    
    # Clean up
    import os
    os.remove(save_path)
    print(f"\n🧹 Cleaned up temporary file")


def generate_comparison_report(results_list, names):
    """Generate a comparison report for multiple preprocessing runs"""
    print_section("COMPARISON REPORT")
    
    print(f"{'Metric':<30} | " + " | ".join(f"{name:^15}" for name in names))
    print("-" * (30 + 3 * len(names) * 16))
    
    # Task type
    row = f"{'Task Type':<30} | "
    row += " | ".join(f"{r['task_type']:^15}" for r in results_list)
    print(row)
    
    # Model family
    row = f"{'Model Family':<30} | "
    row += " | ".join(f"{r['model_family']:^15}" for r in results_list)
    print(row)
    
    # Features
    row = f"{'Final Features':<30} | "
    row += " | ".join(f"{len(r['feature_names']):^15}" for r in results_list)
    print(row)
    
    # Quality score
    row = f"{'Quality Score':<30} | "
    row += " | ".join(f"{r['quality_metrics']['quality_score']:^15.1f}" for r in results_list)
    print(row)
    
    # Columns removed
    row = f"{'Columns Removed':<30} | "
    row += " | ".join(f"{r['preprocessing_metadata']['total_columns_removed']:^15}" for r in results_list)
    print(row)
    
    # Decisions made
    row = f"{'Decisions Made':<30} | "
    row += " | ".join(f"{len(r['decision_log']):^15}" for r in results_list)
    print(row)


def main():
    """Run all demonstrations"""
    print("\n" + "=" * 80)
    print(" ADAPTIVE PREPROCESSING PIPELINE - COMPREHENSIVE DEMONSTRATION ".center(80, "="))
    print("=" * 80)
    
    results = []
    names = []
    
    # Classification
    result_class = run_classification_demo()
    results.append(result_class)
    names.append("Classification")
    
    # Regression
    result_reg = run_regression_demo()
    results.append(result_reg)
    names.append("Regression")
    
    # Unsupervised
    result_unsup = run_unsupervised_demo()
    results.append(result_unsup)
    names.append("Unsupervised")
    
    # Pipeline persistence
    demonstrate_pipeline_persistence()
    
    # Comparison report
    generate_comparison_report(results, names)
    
    print_section("DEMONSTRATION COMPLETE")
    print("✅ All tests passed successfully!")
    print("📚 Review the decision logs above to see intelligent preprocessing in action")
    print("💡 The pipeline automatically adapted to different tasks and model types")


if __name__ == "__main__":
    main()
