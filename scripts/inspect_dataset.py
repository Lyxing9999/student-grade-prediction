from pathlib import Path

import pandas as pd


DATASET_PATH = Path("data/raw/student_performance.csv")


def main() -> None:
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH.resolve()}"
        )

    df = pd.read_csv(DATASET_PATH)

    print("\n=== DATASET FILE ===")
    print(DATASET_PATH.resolve())

    print("\n=== DATASET SHAPE ===")
    print(f"Rows: {df.shape[0]:,}")
    print(f"Columns: {df.shape[1]}")

    print("\n=== COLUMN NAMES ===")
    for index, column in enumerate(df.columns, start=1):
        print(f"{index}. {column}")

    print("\n=== FIRST 5 ROWS ===")
    print(df.head().to_string(index=False))

    print("\n=== DATA TYPES ===")
    print(df.dtypes)

    print("\n=== MISSING VALUES ===")
    print(df.isnull().sum())

    print("\n=== DUPLICATE ROWS ===")
    print(df.duplicated().sum())

    print("\n=== NUMERICAL SUMMARY ===")
    print(df.describe().to_string())

    if "grade" in df.columns:
        print("\n=== GRADE DISTRIBUTION ===")
        print(df["grade"].value_counts(dropna=False).sort_index())

    if "total_score" in df.columns and "grade" in df.columns:
        print("\n=== SCORE RANGE BY GRADE ===")
        print(
            df.groupby("grade")["total_score"]
            .agg(["min", "max", "mean", "count"])
            .sort_index()
        )


if __name__ == "__main__":
    main()

