from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix

def predict_most_frequent(X_train_imputed, X_test_imputed, y_train, y_test):
    dummy_clf = DummyClassifier(strategy="most_frequent")
    dummy_clf.fit(X_train_imputed, y_train)
    dummy_predictions = dummy_clf.predict(X_test_imputed)
    print("Dummy Classifier Predictions (Most Frequent):")
    print(dummy_predictions)
    confusion_mat = confusion_matrix(y_test, dummy_predictions, labels=[0, 1])
    print("Confusion Matrix:")
    print(confusion_mat)
    return dummy_predictions
