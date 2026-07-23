from time import perf_counter

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.data_loader import load_dataset
from src.models.random_forest import create_random_forest
from src.preprocessing import prepare_dataset, split_dataset


GRADE_LABELS = ["A", "B", "C", "D", "F"]


def main() -> None:
    print("\n=== LOADING DATA ===")

    df = load_dataset()
    X, y = prepare_dataset(df)
    split = split_dataset(X, y)

    print(f"Training rows: {len(split.X_train):,}")
    print(f"Testing rows: {len(split.X_test):,}")

    model = create_random_forest()

    print("\n=== TRAINING RANDOM FOREST ===")

    started_at = perf_counter()

    model.fit(
        split.X_train,
        split.y_train,
    )

    training_time = perf_counter() - started_at

    print("\n=== MAKING PREDICTIONS ===")

    predictions = model.predict(split.X_test)

    train_accuracy = model.score(
        split.X_train,
        split.y_train,
    )

    test_accuracy = accuracy_score(
        split.y_test,
        predictions,
    )

    precision, recall, macro_f1, _ = (
        precision_recall_fscore_support(
            split.y_test,
            predictions,
            average="macro",
            zero_division=0,
        )
    )

    print("\n=== BASELINE RESULTS ===")
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Testing accuracy:  {test_accuracy:.4f}")
    print(f"Macro precision:   {precision:.4f}")
    print(f"Macro recall:      {recall:.4f}")
    print(f"Macro F1-score:    {macro_f1:.4f}")
    print(f"Training time:     {training_time:.2f} seconds")

    print("\n=== CONFUSION MATRIX ===")
    print(
        confusion_matrix(
            split.y_test,
            predictions,
            labels=GRADE_LABELS,
        )
    )

    print("\n=== CLASSIFICATION REPORT ===")
    print(
        classification_report(
            split.y_test,
            predictions,
            labels=GRADE_LABELS,
            zero_division=0,
        )
    )

    forest = model.named_steps["model"]

    print("\n=== FOREST INFORMATION ===")
    print(f"Number of trees: {forest.n_estimators}")
    print(f"Maximum depth setting: {forest.max_depth}")

    print("\n=== FEATURE IMPORTANCE ===")
    for feature, importance in zip(
        split.X_train.columns,
        forest.feature_importances_,
        strict=True,
    ):
        print(f"{feature}: {importance:.4f}")


if __name__ == "__main__":
    main()
