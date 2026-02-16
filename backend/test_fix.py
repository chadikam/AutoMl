import pandas as pd
from app.routes.datasets import read_csv_smart

# Test 1: Read preprocessed file WITHOUT treating -1 as missing
print("=" * 60)
print("TEST 1: Reading preprocessed file (na_values=[], keep_default_na=False)")
print("=" * 60)
df1 = read_csv_smart(
    'uploads/preprocessed_06c96e9f-9725-42d5-8308-32fa92b5e530_Heavy Missing Values_1.csv',
    na_values=[],
    keep_default_na=False,
    nrows=10
)
print(f"Last column dtype: {df1.iloc[:, -1].dtype}")
print(f"Last column values:\n{df1.iloc[:, -1]}")
print(f"Null count: {df1.iloc[:, -1].isna().sum()}")
print(f"Unique values: {df1.iloc[:, -1].unique()}")

# Test 2: Simulate JSON serialization
print("\n" + "=" * 60)
print("TEST 2: JSON serialization of last column")
print("=" * 60)
data = df1.to_dict('records')
print(f"First 3 rows, last column values:")
for i in range(min(3, len(data))):
    last_col_name = list(data[i].keys())[-1]
    print(f"  Row {i}: {data[i][last_col_name]}")
