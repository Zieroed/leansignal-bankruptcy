from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from split_data import impute_missing_values

def feature_selection(X_train_imputed, X_val_imputed, y_train):
    selector = SelectFromModel(RandomForestClassifier(random_state=42), threshold="median").fit(X_train_imputed, y_train)
    # print(f"selector.threshold_: {selector.threshold_}")
    # print(f"selector.getsupport(): {selector.get_support()}")
    X_train_selected = selector.transform(X_train_imputed)
    X_val_selected = selector.transform(X_val_imputed)
    print(f"X_train_feature_selected.shape: {X_train_selected.shape}")
    return X_train_selected, X_val_selected

def train_and_predict_random_forest(X_train_imputed, X_val_imputed, y_train):
    rf_clf = RandomForestClassifier(random_state=42)
    rf_clf.fit(X_train_imputed, y_train)
    predictions = rf_clf.predict(X_val_imputed)
    probability_predictions = rf_clf.predict_proba(X_val_imputed)
    X_train_selected, X_val_selected = feature_selection(X_train_imputed, X_val_imputed, y_train)
    rf_clf_selected = RandomForestClassifier(random_state=42).fit(X_train_selected, y_train)
    predictions_selected = rf_clf_selected.predict(X_val_selected)
    probability_selected = rf_clf_selected.predict_proba(X_val_selected)
    return predictions, probability_predictions, predictions_selected, probability_selected, X_train_selected.shape[1]

def train_final_rf(X_dev, y_dev, X_test, y_test):
    # Preprocess
    X_dev_imputed, X_test_imputed = impute_missing_values(X_dev, X_test)

    # Train RF Model and Feature Selector
    final_rf = RandomForestClassifier(random_state=42)
    selector = SelectFromModel(final_rf, threshold="median").fit(X_dev_imputed, y_dev)

    # Feature Selection
    X_dev_selected = selector.transform(X_dev_imputed)
    X_test_selected = selector.transform(X_test_imputed)
    print(f"Development shape before selection: {X_dev.shape}")
    print(f"Development shape after selection: {X_dev_selected.shape}")
    feature_count = X_dev_selected.shape[1]
    print(f"Selected feature count: {feature_count}")

    # Prediction
    final_rf.fit(X_dev_selected, y_dev)
    final_predictions = final_rf.predict(X_test_selected)
    final_probability_predictions = final_rf.predict_proba(X_test_selected)
    return X_dev_selected, final_predictions, final_probability_predictions, feature_count

 