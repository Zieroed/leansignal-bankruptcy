from sklearn.linear_model import LogisticRegression

def train_and_predict_logistic_regression(X_train_imputed, X_val_imputed, y_train):
    lr_clf = LogisticRegression(random_state=0, max_iter=100).fit(X_train_imputed, y_train)
    predictions = lr_clf.predict(X_val_imputed)
    probability_predictions = lr_clf.predict_proba(X_val_imputed)
    print(f"Classes: {lr_clf.classes_}")
    print(f"Predictions: {predictions}")
    print(f"Probability Predictions Shape: {probability_predictions.shape}")
    print(f"Probability Predictions: {probability_predictions[:5]}")
    return
  