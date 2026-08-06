"""Train, evaluate, and save the selected Random Forest model."""

from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import joblib
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from src.data_loader import load_dataset
from src.final_model import (
    FEATURE_COLUMNS,
    GRADE_ORDER,
    create_final_model,
)
from src.preprocessing import prepare_dataset, split_dataset


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "final_random_forest.joblib"
)


def main() -> None:
    print("Loading dataset...")

    dataset = load_dataset()
    features, target = prepare_dataset(dataset)
    split = split_dataset(features, target)

    print(f"Training rows: {len(split.X_train):,}")
    print(f"Testing rows:  {len(split.X_test):,}")

    print("\nTraining tuned Random Forest...")

    model = create_final_model()

    started_at = perf_counter()
    model.fit(split.X_train, split.y_train)
    fit_time_seconds = perf_counter() - started_at

    predictions = model.predict(split.X_test)

    metrics = {
        "accuracy": accuracy_score(
            split.y_test,
            predictions,
        ),
        "macro_precision": precision_score(
            split.y_test,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "macro_recall": recall_score(
            split.y_test,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "macro_f1": f1_score(
            split.y_test,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "fit_time_seconds": fit_time_seconds,
    }

    artifact = {
        "model": model,
        "model_name": "Tuned Random Forest",
        "feature_columns": FEATURE_COLUMNS,
        "grade_order": GRADE_ORDER,
        "metrics": metrics,
        "trained_at": datetime.now(
            timezone.utc
        ).isoformat(),
    }

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        artifact,
        MODEL_PATH,
        compress=3,
    )

    print("\nFinal model saved successfully.")
    print(f"Path: {MODEL_PATH}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(
        "Macro Precision: "
        f"{metrics['macro_precision']:.4f}"
    )
    print(
        "Macro Recall: "
        f"{metrics['macro_recall']:.4f}"
    )
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(
        "Fit time: "
        f"{metrics['fit_time_seconds']:.2f} seconds"
    )


if __name__ == "__main__":
    main()

