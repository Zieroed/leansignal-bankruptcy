import numpy as np
import pandas as pd

from src.leansignal.split_data import impute_missing_values

def test_impute_missing_values_on_both_splits():
    X_train = pd.DataFrame(
        {
            "feature_1": [1, 2, np.nan, 4],
            "feature_2": [np.nan, 1, 2, 3],
        }
    )
    X_test = pd.DataFrame(
        {
            "feature_1": [2, np.nan, np.nan, 4],
            "feature_2": [np.nan, 1, 2, np.nan],
        }
    )

    X_train_imputed, X_test_imputed = impute_missing_values(X_train, X_test)

    assert not np.isnan(X_train_imputed).any()
    assert not np.isnan(X_test_imputed).any()