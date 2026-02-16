"""
Unit Tests for Adaptive Preprocessing Pipeline
Tests all intelligent decision-making capabilities
"""
import unittest
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from app.services.adaptive_preprocessing import (
    AdaptivePreprocessor, TaskType, ModelFamily
)
from app.services.eda_service import EDAService


class TestTaskDetection(unittest.TestCase):
    """Test automatic task type detection"""
    
    def test_classification_categorical_target(self):
        """Test classification detection with categorical target"""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'target': ['A', 'B', 'A', 'B', 'A']
        })
        
        preprocessor = AdaptivePreprocessor()
        task_type = preprocessor.detect_task_type(df, 'target')
        
        self.assertEqual(task_type, TaskType.CLASSIFICATION)
        self.assertEqual(preprocessor.task_type, TaskType.CLASSIFICATION)
    
    def test_classification_discrete_numerical(self):
        """Test classification detection with discrete numerical target"""
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'target': np.random.choice([0, 1, 2], 100)  # Only 3 unique values
        })
        
        preprocessor = AdaptivePreprocessor()
        task_type = preprocessor.detect_task_type(df, 'target')
        
        self.assertEqual(task_type, TaskType.CLASSIFICATION)
    
    def test_regression_continuous_target(self):
        """Test regression detection with continuous target"""
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'target': np.random.randn(100) * 100  # Many unique values
        })
        
        preprocessor = AdaptivePreprocessor()
        task_type = preprocessor.detect_task_type(df, 'target')
        
        self.assertEqual(task_type, TaskType.REGRESSION)
    
    def test_unsupervised_no_target(self):
        """Test unsupervised detection when no target specified"""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [5, 4, 3, 2, 1]
        })
        
        preprocessor = AdaptivePreprocessor()
        task_type = preprocessor.detect_task_type(df, None)
        
        self.assertEqual(task_type, TaskType.UNSUPERVISED)


class TestModelFamilyMapping(unittest.TestCase):
    """Test model family detection and mapping"""
    
    def test_linear_models(self):
        """Test linear model family detection"""
        preprocessor = AdaptivePreprocessor()
        
        preprocessor.set_model_family('logistic_regression')
        self.assertEqual(preprocessor.model_family, ModelFamily.LINEAR)
        
        preprocessor.set_model_family('linear_regression')
        self.assertEqual(preprocessor.model_family, ModelFamily.LINEAR)
    
    def test_tree_based_models(self):
        """Test tree-based model family detection"""
        preprocessor = AdaptivePreprocessor()
        
        for model in ['random_forest', 'decision_tree', 'gradient_boosting', 'xgboost']:
            preprocessor.set_model_family(model)
            self.assertEqual(preprocessor.model_family, ModelFamily.TREE_BASED)
    
    def test_unknown_model(self):
        """Test unknown model handling"""
        preprocessor = AdaptivePreprocessor()
        preprocessor.set_model_family('some_unknown_model')
        
        self.assertEqual(preprocessor.model_family, ModelFamily.UNKNOWN)


class TestIDColumnDetection(unittest.TestCase):
    """Test ID column detection logic"""
    
    def test_id_by_name(self):
        """Test ID detection by column name"""
        df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'user_id': [101, 102, 103, 104, 105],
            'feature': [10, 20, 30, 40, 50]
        })
        
        preprocessor = AdaptivePreprocessor()
        
        self.assertTrue(preprocessor._is_id_column(df, 'id'))
        self.assertTrue(preprocessor._is_id_column(df, 'user_id'))
        self.assertFalse(preprocessor._is_id_column(df, 'feature'))
    
    def test_id_by_uniqueness(self):
        """Test ID detection by high uniqueness"""
        df = pd.DataFrame({
            'unique_col': range(1000),  # 100% unique
            'normal_col': np.random.choice([1, 2, 3], 1000)  # Low uniqueness
        })
        
        preprocessor = AdaptivePreprocessor()
        
        self.assertTrue(preprocessor._is_id_column(df, 'unique_col'))
        self.assertFalse(preprocessor._is_id_column(df, 'normal_col'))
    
    def test_id_by_sequence(self):
        """Test ID detection by sequential pattern"""
        df = pd.DataFrame({
            'sequential': list(range(1, 101)),  # 1, 2, 3, ..., 100
            'non_sequential': [1, 5, 2, 8, 3, 10, 4, 15] + list(range(8, 100))
        })
        
        preprocessor = AdaptivePreprocessor()
        
        self.assertTrue(preprocessor._is_id_column(df, 'sequential'))


class TestConstantColumnDetection(unittest.TestCase):
    """Test constant column detection"""
    
    def test_single_value_column(self):
        """Test detection of columns with only one value"""
        df = pd.DataFrame({
            'constant': ['same'] * 100,
            'variable': np.random.choice(['A', 'B', 'C'], 100)
        })
        
        preprocessor = AdaptivePreprocessor()
        
        self.assertTrue(preprocessor._is_constant_column(df, 'constant'))
        self.assertFalse(preprocessor._is_constant_column(df, 'variable'))
    
    def test_all_nan_column(self):
        """Test detection of all-NaN columns"""
        df = pd.DataFrame({
            'all_nan': [np.nan] * 100,
            'some_values': [1, 2, 3] + [np.nan] * 97
        })
        
        preprocessor = AdaptivePreprocessor()
        
        # All NaN = only 1 unique value (NaN)
        self.assertTrue(preprocessor._is_constant_column(df, 'all_nan'))


class TestMissingDataHandling(unittest.TestCase):
    """Test missing data detection and handling"""
    
    def test_missing_ratio_calculation(self):
        """Test missing value ratio calculation"""
        df = pd.DataFrame({
            'no_missing': [1, 2, 3, 4, 5],
            'half_missing': [1, np.nan, 3, np.nan, 5],
            'all_missing': [np.nan] * 5
        })
        
        preprocessor = AdaptivePreprocessor()
        
        self.assertEqual(preprocessor._get_missing_ratio(df, 'no_missing'), 0.0)
        self.assertEqual(preprocessor._get_missing_ratio(df, 'half_missing'), 0.4)
        self.assertEqual(preprocessor._get_missing_ratio(df, 'all_missing'), 1.0)
    
    def test_high_missing_columns_dropped(self):
        """Test that columns with >60% missing are dropped"""
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'high_missing': [1] + [np.nan] * 4,  # 80% missing
            'target': [0, 1, 0, 1, 0]
        })
        
        preprocessor = AdaptivePreprocessor()
        df_cleaned = preprocessor.detect_and_remove_problematic_columns(df, 'target')
        
        self.assertNotIn('high_missing', df_cleaned.columns)
        self.assertIn('feature1', df_cleaned.columns)
        self.assertIn('high_missing', preprocessor.high_missing_columns)


class TestFeatureCategorization(unittest.TestCase):
    """Test feature type categorization"""
    
    def test_numerical_features(self):
        """Test numerical feature detection"""
        df = pd.DataFrame({
            'int_col': [1, 2, 3, 4, 5],
            'float_col': [1.1, 2.2, 3.3, 4.4, 5.5],
            'target': [0, 1, 0, 1, 0]
        })
        
        preprocessor = AdaptivePreprocessor()
        preprocessor.categorize_features(df, 'target')
        
        self.assertEqual(len(preprocessor.numerical_features), 2)
        self.assertIn('int_col', preprocessor.numerical_features)
        self.assertIn('float_col', preprocessor.numerical_features)
    
    def test_categorical_features(self):
        """Test categorical feature detection"""
        df = pd.DataFrame({
            'string_col': ['A', 'B', 'C', 'A', 'B'],
            'category_col': pd.Categorical(['X', 'Y', 'Z', 'X', 'Y']),
            'target': [0, 1, 0, 1, 0]
        })
        
        preprocessor = AdaptivePreprocessor()
        preprocessor.categorize_features(df, 'target')
        
        self.assertEqual(len(preprocessor.categorical_features), 2)
        self.assertIn('string_col', preprocessor.categorical_features)
    
    def test_binary_numerical_as_categorical(self):
        """Test that binary (0,1) columns are treated as categorical"""
        df = pd.DataFrame({
            'binary_col': [0, 1, 0, 1, 0, 1, 0, 1],
            'continuous_col': [1.5, 2.7, 3.2, 4.1, 5.5, 6.3, 7.8, 8.2],
            'target': [0, 1, 0, 1, 0, 1, 0, 1]
        })
        
        preprocessor = AdaptivePreprocessor()
        preprocessor.categorize_features(df, 'target')
        
        self.assertIn('binary_col', preprocessor.categorical_features)
        self.assertIn('continuous_col', preprocessor.numerical_features)


class TestScalingStrategy(unittest.TestCase):
    """Test intelligent scaling strategy selection"""
    
    def test_no_scaling_for_tree_based(self):
        """Test that tree-based models skip scaling"""
        preprocessor = AdaptivePreprocessor()
        preprocessor.model_family = ModelFamily.TREE_BASED
        preprocessor.numerical_features = ['feature1', 'feature2']
        
        strategy = preprocessor.choose_scaling_strategy()
        
        self.assertEqual(strategy, 'none')
    
    def test_standard_scaling_for_linear_no_outliers(self):
        """Test standard scaling for linear models without outliers"""
        preprocessor = AdaptivePreprocessor()
        preprocessor.model_family = ModelFamily.LINEAR
        preprocessor.numerical_features = ['feature1', 'feature2']
        # No EDA results = no outliers detected
        
        strategy = preprocessor.choose_scaling_strategy()
        
        self.assertEqual(strategy, 'standard')
    
    def test_minmax_scaling_for_neural_networks(self):
        """Test minmax scaling for neural networks"""
        preprocessor = AdaptivePreprocessor()
        preprocessor.model_family = ModelFamily.NEURAL_NETWORK
        preprocessor.numerical_features = ['feature1', 'feature2']
        
        strategy = preprocessor.choose_scaling_strategy()
        
        self.assertEqual(strategy, 'minmax')


class TestEncodingStrategy(unittest.TestCase):
    """Test intelligent encoding strategy selection"""
    
    def test_ordinal_for_tree_based(self):
        """Test ordinal encoding for tree-based models"""
        preprocessor = AdaptivePreprocessor()
        preprocessor.model_family = ModelFamily.TREE_BASED
        
        # Mock EDA results with low cardinality
        preprocessor.eda_results = {
            'categorical_analysis': {
                'categorical_features': {
                    'color': {'unique_count': 5}
                }
            }
        }
        
        strategy = preprocessor.choose_encoding_strategy('color')
        
        self.assertEqual(strategy, 'ordinal')
    
    def test_onehot_for_linear_low_cardinality(self):
        """Test one-hot encoding for linear models with low cardinality"""
        preprocessor = AdaptivePreprocessor()
        preprocessor.model_family = ModelFamily.LINEAR
        
        preprocessor.eda_results = {
            'categorical_analysis': {
                'categorical_features': {
                    'color': {'unique_count': 5}
                }
            }
        }
        
        strategy = preprocessor.choose_encoding_strategy('color')
        
        self.assertEqual(strategy, 'onehot')
    
    def test_ordinal_for_high_cardinality(self):
        """Test ordinal encoding for high cardinality features"""
        preprocessor = AdaptivePreprocessor()
        preprocessor.model_family = ModelFamily.LINEAR
        
        preprocessor.eda_results = {
            'categorical_analysis': {
                'categorical_features': {
                    'zipcode': {'unique_count': 200}  # High cardinality
                }
            }
        }
        
        strategy = preprocessor.choose_encoding_strategy('zipcode')
        
        self.assertEqual(strategy, 'ordinal')


class TestEndToEndPreprocessing(unittest.TestCase):
    """Test complete preprocessing workflows"""
    
    def test_classification_workflow(self):
        """Test complete classification preprocessing"""
        # Create test dataset
        np.random.seed(42)
        df = pd.DataFrame({
            'id': range(100),  # Should be removed
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'category': np.random.choice(['A', 'B', 'C'], 100),
            'target': np.random.choice([0, 1], 100)
        })
        
        # Run EDA
        eda_service = EDAService()
        eda_results = eda_service.perform_eda(df)
        
        # Preprocess
        preprocessor = AdaptivePreprocessor(eda_results=eda_results)
        results = preprocessor.fit_transform(
            df=df,
            target_column='target',
            model_type='random_forest',
            test_size=0.2
        )
        
        # Assertions
        self.assertIsNotNone(results['X_train'])
        self.assertIsNotNone(results['X_test'])
        self.assertIsNotNone(results['y_train'])
        self.assertIsNotNone(results['y_test'])
        self.assertEqual(results['task_type'], TaskType.CLASSIFICATION.value)
        self.assertEqual(results['model_family'], ModelFamily.TREE_BASED.value)
        
        # Check ID column was removed
        self.assertIn('id', preprocessor.id_columns)
        
        # Check train/test split
        self.assertEqual(len(results['X_train']) + len(results['X_test']), 100)
        self.assertGreater(len(results['X_train']), len(results['X_test']))
    
    def test_unsupervised_workflow(self):
        """Test complete unsupervised preprocessing"""
        df = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
            'category': np.random.choice(['A', 'B'], 100)
        })
        
        eda_service = EDAService()
        eda_results = eda_service.perform_eda(df)
        
        preprocessor = AdaptivePreprocessor(eda_results=eda_results)
        results = preprocessor.fit_transform(
            df=df,
            target_column=None,  # Unsupervised
            model_type=None,
            test_size=0
        )
        
        # No train/test split for unsupervised
        self.assertIsNone(results['X_test'])
        self.assertIsNone(results['y_train'])
        self.assertIsNone(results['y_test'])
        self.assertEqual(results['task_type'], TaskType.UNSUPERVISED.value)


class TestPipelinePersistence(unittest.TestCase):
    """Test saving and loading pipelines"""
    
    def test_save_and_load_pipeline(self):
        """Test pipeline serialization"""
        df = pd.DataFrame({
            'feature1': np.random.randn(50),
            'category': np.random.choice(['A', 'B'], 50),
            'target': np.random.choice([0, 1], 50)
        })
        
        # Create and fit pipeline
        preprocessor = AdaptivePreprocessor()
        results = preprocessor.fit_transform(
            df=df,
            target_column='target',
            model_type='random_forest',
            test_size=0.2
        )
        
        # Save pipeline
        filepath = 'test_pipeline.pkl'
        preprocessor.save_pipeline(filepath)
        
        # Load pipeline
        new_preprocessor = AdaptivePreprocessor()
        new_preprocessor.load_pipeline(filepath)
        
        # Check attributes match
        self.assertEqual(preprocessor.feature_names, new_preprocessor.feature_names)
        self.assertEqual(preprocessor.numerical_features, new_preprocessor.numerical_features)
        self.assertEqual(preprocessor.categorical_features, new_preprocessor.categorical_features)
        self.assertEqual(preprocessor.task_type, new_preprocessor.task_type)
        
        # Clean up
        import os
        if os.path.exists(filepath):
            os.remove(filepath)


class TestDecisionLogging(unittest.TestCase):
    """Test decision logging functionality"""
    
    def test_decisions_are_logged(self):
        """Test that preprocessing decisions are logged"""
        df = pd.DataFrame({
            'id': range(50),
            'feature1': np.random.randn(50),
            'target': np.random.choice([0, 1], 50)
        })
        
        preprocessor = AdaptivePreprocessor()
        results = preprocessor.fit_transform(
            df=df,
            target_column='target',
            model_type='logistic_regression',
            test_size=0.2
        )
        
        # Check decision log exists and is populated
        self.assertGreater(len(results['decision_log']), 0)
        
        # Check decision structure
        first_decision = results['decision_log'][0]
        self.assertIn('timestamp', first_decision)
        self.assertIn('category', first_decision)
        self.assertIn('decision', first_decision)
        self.assertIn('reason', first_decision)
        self.assertIn('impact', first_decision)


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestTaskDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestModelFamilyMapping))
    suite.addTests(loader.loadTestsFromTestCase(TestIDColumnDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestConstantColumnDetection))
    suite.addTests(loader.loadTestsFromTestCase(TestMissingDataHandling))
    suite.addTests(loader.loadTestsFromTestCase(TestFeatureCategorization))
    suite.addTests(loader.loadTestsFromTestCase(TestScalingStrategy))
    suite.addTests(loader.loadTestsFromTestCase(TestEncodingStrategy))
    suite.addTests(loader.loadTestsFromTestCase(TestEndToEndPreprocessing))
    suite.addTests(loader.loadTestsFromTestCase(TestPipelinePersistence))
    suite.addTests(loader.loadTestsFromTestCase(TestDecisionLogging))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*80)
    
    return result


if __name__ == '__main__':
    result = run_tests()
    sys.exit(0 if result.wasSuccessful() else 1)
