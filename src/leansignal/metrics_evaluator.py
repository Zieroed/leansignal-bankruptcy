from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, accuracy_score
import numpy as np

def compute_precision(y_val, dummy_predictions):
    return precision_score(y_val, dummy_predictions, average='binary', pos_label=1, zero_division=0.0)

def compute_recall(y_val, dummy_predictions):
    return recall_score(y_val, dummy_predictions, average='binary', pos_label=1)

def compute_f1_score(y_val, dummy_predictions):
    return f1_score(y_val, dummy_predictions, average='binary', pos_label=1)

def compute_roc_auc(y_val, dummy_predictions):
    return roc_auc_score(y_val, dummy_predictions)

def compute_average_precision(y_val, dummy_predictions):
    return average_precision_score(y_val, dummy_predictions)

def compute_accuracy(y_val, dummy_predictions):
    return accuracy_score(y_val, dummy_predictions)