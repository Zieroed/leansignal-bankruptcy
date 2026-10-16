from load_data import load_data
from audit_data import describe_data, print_first_five_rows, print_missing_values_info, print_class_distribution

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
        print("5. Exit")

        choice = input("Enter your choice (1-5): ")
        if choice == '1':
            describe_data(df)
        elif choice == '2':
            print_first_five_rows(df)
        elif choice == '3':
            print_missing_values_info(df)
        elif choice == '4':
            print_class_distribution(df)
        elif choice == '5':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 5.")

if __name__ == "__main__":
    main()