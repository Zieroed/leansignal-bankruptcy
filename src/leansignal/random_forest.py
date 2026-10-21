from sklearn.ensemble import RandomForestClassifier

def train_and_predict_random_forest(X_train_imputed, X_val_imputed, y_train):
    rf_clf = RandomForestClassifier(random_state=42)
    rf_clf.fit(X_train_imputed, y_train)
    predictions = rf_clf.predict(X_val_imputed)
    probability_predictions = rf_clf.predict_proba(X_val_imputed)
    return predictions, probability_predictions