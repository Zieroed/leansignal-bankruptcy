from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, accuracy_score
import numpy as np

def compute_precision(y_true, y_pred):
    return round(precision_score(y_true, y_pred, average='binary', pos_label=1, zero_division=0.0)* 100, 2) 

def compute_recall(y_true, y_pred):
    return round(recall_score(y_true, y_pred, average='binary', pos_label=1)* 100, 2) 

def compute_f1_score(y_true, y_pred):
    return round(f1_score(y_true, y_pred, average='binary', pos_label=1)* 100, 2) 

def compute_roc_auc(y_true, y_pred):
    return round(roc_auc_score(y_true, y_pred)* 100, 2) 

def compute_average_precision(y_true, y_pred):
    return round(average_precision_score(y_true, y_pred)* 100, 2) 

def compute_accuracy(y_true, y_pred):
    return round(accuracy_score(y_true, y_pred)* 100, 2) 

def compute_metrics(y_true, y_probabilites, y_predictions):
    y_probabilites = y_probabilites[:, -1] if y_probabilites.ndim == 2 else y_probabilites
    return {
        "PR-AUC": compute_average_precision(y_true, y_probabilites),
        "ROC-AUC": compute_roc_auc(y_true, y_probabilites),
        "Precision": compute_precision(y_true, y_predictions),
        "Recall": compute_recall(y_true, y_predictions),
        "F1": compute_f1_score(y_true, y_predictions),
        "Accuracy": compute_accuracy(y_true, y_predictions),
    }