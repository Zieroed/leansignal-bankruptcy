from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
import pandas as pd

def predict_most_frequent(X_train_imputed, X_test_imputed, y_train, y_test):
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(X_train_imputed, y_train)
    dummy_predictions = dummy_clf.predict(X_test_imputed)
    confusion_mat = confusion_matrix(y_test, dummy_predictions, labels=[0, 1])
    print("Dummy Classifier Predictions (Most Frequent):")
    print(dummy_predictions)
    print("Confusion Matrix:")
    class_names = ['Non-Bankrupt', 'Bankrupt']
    confusion_display = pd.DataFrame(confusion_mat, index=[f'T: {x}' for x in class_names], columns=[f'P: {x}' for x in class_names])
    print(confusion_display)
    return dummy_predictions, confusion_mat