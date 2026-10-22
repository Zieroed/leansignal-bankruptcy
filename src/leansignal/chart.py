import matplotlib.pyplot as plt
from load_data import PATH_ROOT

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

    