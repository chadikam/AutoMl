import pandas as pd

# Test if pandas matches -1.0 when looking for -1 in na_values
df = pd.read_csv(
    'uploads/preprocessed_06c96e9f-9725-42d5-8308-32fa92b5e530_Heavy Missing Values_1.csv',
    na_values=['-1'],
    keep_default_na=True,
    nrows=10
)

print("With na_values=['-1']:")
print(f"Last column dtypes: {df.iloc[:, -1].dtype}")
print(f"Last column values:\n{df.iloc[:, -1]}")
print(f"\nNull count: {df.iloc[:, -1].isna().sum()}")
