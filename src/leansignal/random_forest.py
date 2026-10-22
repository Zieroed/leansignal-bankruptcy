from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

def feature_selection(rf_clf, X_train_imputed, y_train):
    selector = SelectFromModel(RandomForestClassifier(), threshold="median").fit(X_train_imputed, y_train)
    print(f"selector.threshold_: {selector.threshold_}")
    print(f"selector.getsupport(): {selector.get_support()}")
    X_train_feature_selected = selector.transform(X_train_imputed)
    print(f"X_train_feature_selected.shape: {X_train_feature_selected.shape}")
    return

def train_and_predict_random_forest(X_train_imputed, X_val_imputed, y_train):
    rf_clf = RandomForestClassifier(random_state=42)
    rf_clf.fit(X_train_imputed, y_train)
    predictions = rf_clf.predict(X_val_imputed)
    probability_predictions = rf_clf.predict_proba(X_val_imputed)
    feature_selection(rf_clf, X_train_imputed, y_train)
    return predictions, probability_predictions

 