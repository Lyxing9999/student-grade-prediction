from src.data_loader import load_dataset
from src.preprocessing import (
    prepare_dataset,
    split_dataset,
)


def main() -> None:
    df = load_dataset()

    X, y = prepare_dataset(df)
    split = split_dataset(X, y)

    print("\n=== SHARED DATA PREPARATION ===")
    print(f"Full dataset: {len(df):,}")
    print(f"Training records: {len(split.X_train):,}")
    print(f"Testing records: {len(split.X_test):,}")

    print("\n=== INPUT FEATURES ===")
    for column in X.columns:
        print(f"- {column}")

    print("\n=== TARGET ===")
    print("- grade")

    print("\n=== TRAINING GRADE DISTRIBUTION ===")
    print(split.y_train.value_counts().sort_index())

    print("\n=== TESTING GRADE DISTRIBUTION ===")
    print(split.y_test.value_counts().sort_index())


if __name__ == "__main__":
    main()
