from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

def feature_selection(X_train_imputed, X_val_imputed, y_train):
    selector = SelectFromModel(RandomForestClassifier(), threshold="median").fit(X_train_imputed, y_train)
    print(f"selector.threshold_: {selector.threshold_}")
    print(f"selector.getsupport(): {selector.get_support()}")
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
    return predictions, probability_predictions, predictions_selected, probability_selected

 