"""
Test multicollinearity detection and handling
"""
import pandas as pd
import numpy as np
from app.services.adaptive_preprocessing import AdaptivePreprocessor, ModelFamily

# Create a sample dataset with correlated features
np.random.seed(42)
n_samples = 100

# Create base features
year_built = np.random.randint(1950, 2020, n_samples)
garage_year_built = year_built + np.random.randint(-5, 10, n_samples)  # Highly correlated

garage_cars = np.random.randint(0, 4, n_samples)
garage_area = garage_cars * 250 + np.random.randn(n_samples) * 50  # Highly correlated

lot_area = np.random.randint(5000, 20000, n_samples)
lot_frontage = lot_area / 100 + np.random.randn(n_samples) * 10  # Moderately correlated

# Independent features
total_rooms = np.random.randint(4, 10, n_samples)
bedrooms = np.random.randint(2, 5, n_samples)

# Target
sale_price = (
    year_built * 100 + 
    garage_area * 50 + 
    lot_area * 0.5 + 
    total_rooms * 5000 + 
    np.random.randn(n_samples) * 10000
)

# Create DataFrame
df = pd.DataFrame({
    'year_built': year_built,
    'garage_year_built': garage_year_built,
    'garage_cars': garage_cars,
    'garage_area': garage_area,
    'lot_area': lot_area,
    'lot_frontage': lot_frontage,
    'total_rooms': total_rooms,
    'bedrooms': bedrooms,
    'sale_price': sale_price
})

print("=" * 80)
print("Testing Multicollinearity Detection and Handling")
print("=" * 80)

# Test with Linear model (should combine features)
print("\n1. LINEAR MODEL (Severe Impact - Should Combine Features)")
print("-" * 80)
preprocessor_linear = AdaptivePreprocessor()
preprocessor_linear.model_family = ModelFamily.LINEAR
correlated_pairs = preprocessor_linear.detect_high_correlation(df.drop(columns=['sale_price']), threshold=0.9)
print(f"\nDetected {len(correlated_pairs)} highly correlated pairs:")
for pair in correlated_pairs:
    print(f"  {pair['feature1']} <-> {pair['feature2']}")
    print(f"    Correlation: {pair['correlation']:.3f}")
    print(f"    Semantic relationship: {pair['semantic_relationship']}")
    print(f"    Can combine: {pair['can_combine']}")
    print()

df_linear = df.drop(columns=['sale_price']).copy()
df_linear = preprocessor_linear.handle_multicollinearity(df_linear, correlated_pairs)
print(f"\nAfter handling (LINEAR):")
print(f"  Original columns: {len(df.columns) - 1}")
print(f"  Final columns: {len(df_linear.columns)}")
print(f"  New columns: {[c for c in df_linear.columns if c not in df.columns]}")
print(f"  Dropped columns: {[c for c in df.columns if c not in df_linear.columns and c != 'sale_price']}")

metadata = preprocessor_linear.preprocessing_metadata.get('multicollinearity_handled', {})
print(f"\nMetadata:")
print(f"  Pairs detected: {metadata.get('pairs_detected', 0)}")
print(f"  Features dropped: {metadata.get('features_dropped', [])}")
print(f"  Features combined: {len(metadata.get('features_combined', []))}")
for combined in metadata.get('features_combined', []):
    print(f"    - {combined['new_feature']}: {combined['method']} ({combined['reason']})")

# Test with Tree-based model (should keep features)
print("\n" + "=" * 80)
print("2. TREE-BASED MODEL (Moderate Impact - Should Keep Features)")
print("-" * 80)
preprocessor_tree = AdaptivePreprocessor()
preprocessor_tree.model_family = ModelFamily.TREE_BASED
correlated_pairs_tree = preprocessor_tree.detect_high_correlation(df.drop(columns=['sale_price']), threshold=0.9)
print(f"\nDetected {len(correlated_pairs_tree)} highly correlated pairs (same as linear)")

df_tree = df.drop(columns=['sale_price']).copy()
df_tree = preprocessor_tree.handle_multicollinearity(df_tree, correlated_pairs_tree)
print(f"\nAfter handling (TREE-BASED):")
print(f"  Original columns: {len(df.columns) - 1}")
print(f"  Final columns: {len(df_tree.columns)}")
print(f"  Columns dropped: {len(df.columns) - 1 - len(df_tree.columns)}")
print(f"  (Tree-based models keep correlated features)")

# Test with Distance-based model (should drop features)
print("\n" + "=" * 80)
print("3. DISTANCE-BASED MODEL (Medium Impact - Should Drop Features)")
print("-" * 80)
preprocessor_distance = AdaptivePreprocessor()
preprocessor_distance.model_family = ModelFamily.DISTANCE_BASED
correlated_pairs_distance = preprocessor_distance.detect_high_correlation(df.drop(columns=['sale_price']), threshold=0.9)
print(f"\nDetected {len(correlated_pairs_distance)} highly correlated pairs")

df_distance = df.drop(columns=['sale_price']).copy()
df_distance = preprocessor_distance.handle_multicollinearity(df_distance, correlated_pairs_distance)
print(f"\nAfter handling (DISTANCE-BASED):")
print(f"  Original columns: {len(df.columns) - 1}")
print(f"  Final columns: {len(df_distance.columns)}")
print(f"  Dropped columns: {[c for c in df.columns if c not in df_distance.columns and c != 'sale_price']}")

print("\n" + "=" * 80)
print("✅ All tests completed!")
print("=" * 80)
