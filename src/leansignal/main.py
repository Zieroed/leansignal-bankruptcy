from load_data import load_data
from audit_data import describe_data, print_first_five_rows, print_missing_values_info, print_class_distribution
from split_data import split_data, split_train_test, print_per_class_distribution, impute_missing_values
from dummy_classifier import predict_most_frequent
from metrics_evaluator import compute_precision, compute_recall, compute_f1_score, compute_roc_auc, compute_average_precision, compute_accuracy
from logistic_regression import train_and_predict_logistic_regression

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

            print_line_separator()
            print("Metrics for the dummy classifier:")
            print(f"Precision score: {round(compute_precision(y_val, dummy_predictions) * 100, 2)}%")
            print(f"Recall score: {round(compute_recall(y_val, dummy_predictions) * 100, 2)}%")
            print(f"F1 score: {round(compute_f1_score(y_val, dummy_predictions) * 100, 2)}%")
            print(f"ROC AUC score: {round(compute_roc_auc(y_val, dummy_predictions) * 100, 2)}%")
            print(f"Average Precision score: {round(compute_average_precision(y_val, dummy_predictions) * 100, 2)}%")
            print(f"Accuracy score: {round(compute_accuracy(y_val, dummy_predictions) * 100, 2)}%")

            print_line_separator()
            train_and_predict_logistic_regression(X_train_imputed, X_val_imputed, y_train)

        elif choice == '0':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")
        repeat = False

if __name__ == "__main__":
    main()