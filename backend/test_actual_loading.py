import pandas as pd
import sys
sys.path.insert(0, 'app')

from routes.datasets import read_csv_smart, MISSING_VALUE_INDICATORS

# Load the file the exact same way the backend does
file_path = 'uploads/d27e6a83-b9eb-4dc3-b02e-ca4cb4646e27_Ames Housing (many features, missing values).csv'

print("Testing how backend actually loads the file...\n")

# Load using the same method as backend
df = read_csv_smart(file_path, 'utf-8')

print(f"Shape: {df.shape}")
print(f"Columns: {len(df.columns)}")

# Check the three problematic columns
for col in ['Alley', 'Mas Vnr Type', 'Fence']:
    if col in df.columns:
        print(f"\n{'='*60}")
        print(f"Column: {col}")
        print(f"dtype: {df[col].dtype}")
        
        vc = df[col].value_counts(dropna=False)
        print(f"\nValue counts:")
        print(vc.head(10))
        
        most_common = vc.index[0]
        count = int(vc.iloc[0])
        pct = (count / len(df) * 100)
        
        print(f"\nMost common value:")
        print(f"  Value: {repr(most_common)}")
        print(f"  Type: {type(most_common)}")
        print(f"  Is NaN: {pd.isna(most_common)}")
        print(f"  Equals '': {most_common == '' if not pd.isna(most_common) else 'N/A'}")
        print(f"  Count: {count}")
        print(f"  Percentage: {pct:.1f}%")

