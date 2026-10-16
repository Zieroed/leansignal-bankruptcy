import pandas as pd
from scipy.io import arff
from pathlib import Path

PATH_ROOT = Path(__file__).resolve().parent.parent.parent

data, meta = arff.loadarff(f'{PATH_ROOT}/data/raw/3year.arff')

df = pd.DataFrame(data)
shape = df.shape
dtypes = df.dtypes

# print("First 5 rows:")
# print(df.head(5))

print("Shape of DataFrame:")
print(shape)

print("Data types of each column:")
print(dtypes)

print("Total number of missing values per column:")
print(df.isna().sum())
print("Total number of missing values in the DataFrame:")
print(df.isna().sum().sum())
print("Columns with missing values:")
missing_cols_list = df.columns[df.isna().any()].tolist()
print(missing_cols_list)
print("Count of columns with missing values:")
print(len(missing_cols_list))

print("Count per class:")
print(df.iloc[:, -1].value_counts())
print("Percentage per class:")
print(df.iloc[:, -1].value_counts(normalize=True).round(4) * 100)