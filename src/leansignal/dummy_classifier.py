from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd

def predict_most_frequent(X_train_imputed, X_test_imputed, y_train, y_test):
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(X_train_imputed, y_train)
    dummy_predictions = dummy_clf.predict(X_test_imputed)
    return dummy_predictions

