from load_data import load_data
from audit_data import describe_data, print_first_five_rows, print_missing_values_info, print_class_distribution
from split_data import split_data, print_class_distribution, impute_missing_values

def main():
    print("Loading data..")
    df = load_data()
    print("Data loaded successfully.")

    while True:
        print("\nSelect an option:")
        print("1. Describe data")
        print("2. Print first 5 rows")
        print("3. Print missing values info")
        print("4. Print class distribution")
        print("5. Split data into train and test sets")
        print("0. Exit")

        choice = input("Enter your choice: ")
        if choice == '1':
            describe_data(df)
        elif choice == '2':
            print_first_five_rows(df)
        elif choice == '3':
            print_missing_values_info(df)
        elif choice == '4':
            print_class_distribution(df)
        elif choice == '5':
            X_train, X_test, y_train, y_test = split_data(df)
            print("X_train shape:", X_train.shape)
            print("X_test shape:", X_test.shape)
            print("y_train shape:", y_train.shape)
            print("y_test shape:", y_test.shape)
            print_class_distribution(y_train, y_test)
            X_train_imputed, X_test_imputed = impute_missing_values(X_train, X_test)
        elif choice == '0':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")

if __name__ == "__main__":
    main()