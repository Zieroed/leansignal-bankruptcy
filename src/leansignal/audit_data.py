def describe_data(df):
    print(df.shape)
    print(df.dtypes)

def print_first_five_rows(df):
    print("First 5 rows:")
    print(df.head(5))

def print_missing_values_info(df):
    print("Total number of missing values per column:")
    print(df.isna().sum())
    print("Total number of missing values in the DataFrame:")
    print(df.isna().sum().sum())
    print("Columns with missing values:")
    missing_cols_list = df.columns[df.isna().any()].tolist()
    print(missing_cols_list)
    print("Count of columns with missing values:")
    print(len(missing_cols_list))

def print_class_distribution(df):
    print("Count per class:")
    print(df.iloc[:, -1].value_counts())
    print("Percentage per class:")
    print(df.iloc[:, -1].value_counts(normalize=True).round(4) * 100)