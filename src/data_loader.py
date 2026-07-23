import pandas as pd

from src.config import (
    DATA_PATH,
    FEATURE_COLUMNS,
    TARGET_COLUMN,
)


def load_dataset() -> pd.DataFrame:
    """Load and validate the student-performance dataset."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = set(
        FEATURE_COLUMNS + [TARGET_COLUMN]
    )

    missing_columns = (
        required_columns - set(df.columns)
    )

    if missing_columns:
        raise ValueError(
            f"Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    return df
