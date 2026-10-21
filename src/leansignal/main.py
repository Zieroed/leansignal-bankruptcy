from load_data import load_data
from audit_data import describe_data, print_first_five_rows, print_missing_values_info, print_class_distribution
from split_data import split_data, split_train_test, print_per_class_distribution, impute_missing_values
from dummy_classifier import predict_most_frequent
from metrics_evaluator import compute_precision, compute_recall, compute_f1_score, compute_roc_auc, compute_average_precision, compute_accuracy
from logistic_regression import train_and_predict_logistic_regression
from random_forest import train_and_predict_random_forest

def print_x_y_shapes(X_dev, X_test, y_dev, y_test, x, y):
    print(f"{x} shape: {X_dev.shape}")
    print(f"{y} shape: {X_test.shape}")
    # print("y_dev shape:", y_dev.shape)
    # print("y_test shape:", y_test.shape)
    return

def print_line_separator():
    print("=" * 50)
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
            dummy_predictions, confusion_mat = predict_most_frequent(X_train_imputed, X_val_imputed, y_train, y_val)
            dummy_prauc = compute_average_precision(y_val, dummy_predictions)
            dummy_rocauc = compute_roc_auc(y_val, dummy_predictions)
            dummy_precision = compute_precision(y_val, dummy_predictions)
            dummy_recall = compute_recall(y_val, dummy_predictions)
            dummy_f1 = compute_f1_score(y_val, dummy_predictions)
            dummy_accuracy = compute_accuracy(y_val, dummy_predictions)

            print_line_separator()
            print("Training Logistic Regression Model")
            lr_predictions, lr_proba = train_and_predict_logistic_regression(X_train_imputed, X_val_imputed, y_train)
            lr_prauc = compute_average_precision(y_val, lr_proba[:, -1])
            lr_rocauc = compute_roc_auc(y_val, lr_proba[:, -1])
            lr_precision = compute_precision(y_val, lr_predictions)
            lr_recall = compute_recall(y_val, lr_predictions)
            lr_f1 = compute_f1_score(y_val, lr_predictions)
            lr_accuracy = compute_accuracy(y_val, lr_predictions)

            print_line_separator()
            print("Training Random Forest Classifier")
            rf_predictions, rf_proba = train_and_predict_random_forest(X_train_imputed, X_val_imputed, y_train)
            rf_prauc = compute_average_precision(y_val, rf_proba[:, -1])
            rf_rocauc = compute_roc_auc(y_val, rf_proba[:, -1])
            rf_precision = compute_precision(y_val, rf_predictions)
            rf_recall = compute_recall(y_val, rf_predictions)
            rf_f1 = compute_f1_score(y_val, rf_predictions)
            rf_accuracy = compute_accuracy(y_val, rf_predictions)

            print_line_separator()
            metrics = [
                ("Metric", "Dummy", "LR", "RF"),
                ("PR-AUC", dummy_prauc, lr_prauc, rf_prauc),
                ("ROC-AUC", dummy_rocauc, lr_rocauc, rf_rocauc),
                ("Precision", dummy_precision, lr_precision, rf_precision),
                ("Recall", dummy_recall, lr_recall, rf_recall),
                ("F1", dummy_f1, lr_f1, rf_f1),
                ("Accuracy", dummy_accuracy, lr_accuracy, lr_accuracy),
            ]
            print("Metrics Comparison:")
            for metric, dummy, lr, rf in metrics:
                print(f"{metric:<12} {dummy:<12} {lr:<12} {rf:<12}")
            
        elif choice == '0':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")
        repeat = False

if __name__ == "__main__":
    main()