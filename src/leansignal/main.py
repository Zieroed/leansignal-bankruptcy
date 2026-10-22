from load_data import load_data, save_data
from audit_data import describe_data, print_first_five_rows, print_missing_values_info, print_class_distribution
from split_data import split_data, split_train_test, print_per_class_distribution, impute_missing_values
from dummy_classifier import predict_most_frequent
from metrics_evaluator import compute_precision, compute_recall, compute_f1_score, compute_roc_auc, compute_average_precision, compute_accuracy, compute_metrics
from logistic_regression import train_and_predict_logistic_regression
from random_forest import train_and_predict_random_forest
from sklearn.metrics import confusion_matrix
import pandas as pd
from chart import plot_bar_chart


def print_x_y_shapes(X_dev, X_test, y_dev, y_test, x, y):
    print(f"{x} shape: {X_dev.shape}")
    print(f"{y} shape: {X_test.shape}")
    # print("y_dev shape:", y_dev.shape)
    # print("y_test shape:", y_test.shape)
    return

def print_line_separator():
    print("=" * 50)
    return

def print_confusion_matrix(model_name, y_val, predictions, labels=[1,0]):
    confusion_mat = confusion_matrix(y_val, predictions, labels=labels)
    class_names = ['Bankrupt', 'Non-Bankrupt']
    confusion_display = pd.DataFrame(confusion_mat, index=[f'T: {x}' for x in class_names], columns=[f'P: {x}' for x in class_names])
    print(f"{model_name} Confusion Matrix:")
    print(confusion_display)
    return

def main():
    print("Loading data..")
    df = load_data()
    print("Data loaded successfully.")
    repeat = True
    while repeat:
        print("\nSelect an option:")
        print("1. Describe data")
        print("2. Print first 5 rows")
        print("3. Print missing values info")
        print("4. Print class distribution")
        print("5. Split data into train and test sets")
        print("0. Exit")

        choice = '5' #input("Enter your choice: ")
        if choice == '1':
            describe_data(df)
        elif choice == '2':
            print_first_five_rows(df)
        elif choice == '3':
            print_missing_values_info(df)
        elif choice == '4':
            print_class_distribution(df)
        elif choice == '5':
            print_line_separator()
            print("Spliting development and test sets from the original dataset...")
            X_dev, X_test, y_dev, y_test = split_data(df)
            print_x_y_shapes(X_dev, X_test, y_dev, y_test, "Dev", "Test")
            print_per_class_distribution(y_dev, y_test) #80/20 bankrupt/non-bankrupt for dev and test

            print_line_separator()
            print("Splitting train and validation sets from the development set...")
            X_train, X_val, y_train, y_val = split_train_test(X_dev, y_dev)
            print_x_y_shapes(X_train, X_val, y_train, y_val , "Train", "Validation")
            print_per_class_distribution(y_train, y_val)

            print_line_separator()
            print("Imputing missing values in the training and validation sets...")
            X_train_imputed, X_val_imputed = impute_missing_values(X_train, X_val)

            print_line_separator()
            print("Running dummy classifier on the imputed training and validation sets...")
            dummy_predictions = predict_most_frequent(X_train_imputed, X_val_imputed, y_train, y_val)
            print_confusion_matrix("Dummy", y_val, dummy_predictions)

            print_line_separator()
            print("Training Logistic Regression Model")
            lr_predictions, lr_proba = train_and_predict_logistic_regression(X_train_imputed, X_val_imputed, y_train)
            print_confusion_matrix("Logistic Regression", y_val, lr_predictions)

            print_line_separator()
            print("Training Random Forest Classifier")
            rf_predictions, rf_proba, predictions_selected, probability_selected, rf_feature_count = train_and_predict_random_forest(X_train_imputed, X_val_imputed, y_train)
            print_confusion_matrix("Random_Forest", y_val, rf_predictions)

            print_line_separator()
            print("Reducing Features and Evaluating using Random Forest")
            print_confusion_matrix("Random_Forest", y_val, predictions_selected)

            print_line_separator()    

        elif choice == '0':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")
        repeat = False

        metrics_by_model = {
            "Dummy": {
                "Features": X_train_imputed.shape[1],
                **compute_metrics(y_val, dummy_predictions, dummy_predictions),
            },
            "Logistic Regression": {
                "Features": X_train_imputed.shape[1],
                **compute_metrics(y_val, lr_proba, lr_predictions),
            },
            "Random Forest": {
                "Features": X_train_imputed.shape[1],
                **compute_metrics(y_val, rf_proba, rf_predictions),
            },
            "Random Forest (32 Features)": {
                "Features": rf_feature_count,
                **compute_metrics(y_val, probability_selected, predictions_selected),
            },
        }
        results = pd.DataFrame.from_dict(metrics_by_model, orient="index")
        results.index.name = "Model"
        results = results.reset_index()
        print(results.to_string(index=False))
        save_data(results, "baseline_results.csv")
        plot_bar_chart(metrics_by_model)

if __name__ == "__main__":
    main()