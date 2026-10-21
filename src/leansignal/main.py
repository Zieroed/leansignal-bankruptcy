from load_data import load_data
from audit_data import describe_data, print_first_five_rows, print_missing_values_info, print_class_distribution
from split_data import split_data, split_train_test, print_per_class_distribution, impute_missing_values

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

        elif choice == '0':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")

if __name__ == "__main__":
    main()