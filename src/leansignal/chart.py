import matplotlib.pyplot as plt
from load_data import PATH_ROOT
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_bar_chart(metrics_dictionary):
    models = []
    prauc = []
    for model, metrics in metrics_dictionary.items():
        models.append(model)
        prauc.append(metrics["PR-AUC"])

    plt.bar(models, prauc)
    plt.title("PR-AUC Comparison")
    plt.xlabel("Models")
    plt.ylabel("PR-AUC Score")

    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    plt.savefig(f"{PATH_ROOT}/results/pr_auc_comparison.png", dpi=300)

def save_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Bankrupt', 'Non-Bankrupt'])
    disp.plot(cmap=plt.cm.Blues)

    plt.savefig(f'{PATH_ROOT}/results/final_model_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()