from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    FEATURE_COLUMNS,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)


@dataclass
class DatasetSplit:
    """Shared training and testing data."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def prepare_dataset(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Create features X and target y."""

    X = df[FEATURE_COLUMNS].copy()
    y = (
        df[TARGET_COLUMN]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    return X, y


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
) -> DatasetSplit:
    """Create one reproducible stratified split."""

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=y,
        )
    )

    return DatasetSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )
