"""
Test script to figure out how to extract categorical value mappings from a fitted pipeline.
"""
import joblib
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# This script will help us understand how to extract categorical mappings
# from the preprocessor pipeline that's saved during training

def extract_categorical_mappings(pipeline_path):
    """Extract categorical feature to encoded value mappings from a fitted pipeline."""
    try:
        # Load the pipeline
        preprocessor = joblib.load(pipeline_path)
        
        # The preprocessor is a ColumnTransformer with transformers
        # Get the categorical pipeline
        for name, transformer, columns in preprocessor.transformers_:
            if name == 'cat':
                # This is the categorical pipeline
                # It's a Pipeline with steps like [('imputer', ...), ('encoder', ...)]
                encoder = transformer.named_steps['encoder']
                
                # Check if it's OrdinalEncoder or OneHotEncoder
                if isinstance(encoder, OrdinalEncoder):
                    # OrdinalEncoder has categories_ attribute
                    # It's a list of arrays, one per feature
                    mappings = {}
                    for i, col in enumerate(columns):
                        categories = encoder.categories_[i]
                        # Create mapping: encoded_value -> original_value
                        mappings[col] = {
                            int(j): str(cat) for j, cat in enumerate(categories)
                        }
                    return mappings
                
                elif isinstance(encoder, OneHotEncoder):
                    # OneHotEncoder also has categories_
                    mappings = {}
                    for i, col in enumerate(columns):
                        categories = encoder.categories_[i]
                        mappings[col] = {
                            int(j): str(cat) for j, cat in enumerate(categories)
                        }
                    return mappings
        
        return {}
    except Exception as e:
        print(f"Error extracting mappings: {e}")
        return {}

# Example usage (commented out, this is just for reference)
# mappings = extract_categorical_mappings('path/to/preprocessor.pkl')
# print(mappings)
# Output example:
# {
#   'workclass': {0: 'Private', 1: 'Self-emp-not-inc', 2: 'State-gov', ...},
#   'education': {0: 'Bachelors', 1: 'HS-grad', 2: 'Masters', ...},
# }
