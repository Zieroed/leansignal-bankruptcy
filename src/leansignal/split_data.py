from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import numpy as np

def split_data(df, test_size=0.2, random_state=2026):
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    X_dev, X_test, y_dev, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    return X_dev, X_test, y_dev, y_test

def split_train_test(X_dev, y_dev, val_size=0.2, random_state=42):
    X_train, X_val, y_train, y_val = train_test_split(X_dev, y_dev, test_size=val_size, random_state=random_state, stratify=y_dev)
    return X_train, X_val, y_train, y_val

def print_per_class_distribution(y_dev, y_test):
    print("Train Percentage per class:")
    print(y_dev.value_counts(normalize=True).round(4) * 100)
    print("Test Percentage per class:")
    print(y_test.value_counts(normalize=True).round(4) * 100)
    return

def impute_missing_values(X_dev, X_test):
    imp_median = SimpleImputer(strategy='median')
    imp_median.fit(X_dev)
    X_dev_imputed = imp_median.transform(X_dev)
    X_test_imputed = imp_median.transform(X_test)
    return X_dev_imputed, X_test_imputed
