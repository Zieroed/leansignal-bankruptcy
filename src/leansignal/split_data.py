from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import numpy as np

def split_data(df, test_size=0.2, random_state=42):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    return X_train, X_test, y_train, y_test

def print_class_distribution(y_train, y_test):
    print("Train Percentage per class:")
    print(y_train.value_counts(normalize=True).round(4) * 100)
    print("Test Percentage per class:")
    print(y_test.value_counts(normalize=True).round(4) * 100)
    return

def impute_missing_values(X_train, X_test):
    imp_median = SimpleImputer(strategy='median')
    imp_median.fit(X_train)
    X_train_imputed = imp_median.transform(X_train)
    X_test_imputed = imp_median.transform(X_test)
    print("Total number of missing values per column:")
    print(f"Train missing values:{np.isnan(X_train_imputed).sum()}")
    print(f"Test missing values:{np.isnan(X_test_imputed).sum()}")
    return X_train_imputed, X_test_imputed
