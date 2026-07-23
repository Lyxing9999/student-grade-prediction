from time import perf_counter

from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    train_test_split,
)

from src.data_loader import load_dataset
from src.models.logistic_regression import (
    create_logistic_regression,
)
from src.preprocessing import (
    prepare_dataset,
    split_dataset,
)


GRADE_LABELS = ["A", "B", "C", "D", "F"]

# Use part of the training data for faster parameter search.
TUNING_SAMPLE_SIZE = 200_000


def main() -> None:
    print("\n=== LOADING SHARED DATA ===")

    df = load_dataset()
    X, y = prepare_dataset(df)
    split = split_dataset(X, y)

    print(f"Full training rows: {len(split.X_train):,}")
    print(f"Testing rows:       {len(split.X_test):,}")

    X_tune, _, y_tune, _ = train_test_split(
        split.X_train,
        split.y_train,
        train_size=TUNING_SAMPLE_SIZE,
        random_state=42,
        stratify=split.y_train,
    )

    print(f"Tuning sample rows: {len(X_tune):,}")

    pipeline = create_logistic_regression()

    parameter_space = {
        "model__C": [
            0.01,
            0.1,
            1.0,
            10.0,
            100.0,
        ],
        "model__class_weight": [
            None,
            "balanced",
        ],
        "model__solver": [
            "lbfgs",
        ],
    }

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameter_space,
        n_iter=8,
        scoring="f1_macro",
        cv=3,
        random_state=42,
        n_jobs=-1,
        verbose=1,
        refit=True,
        return_train_score=True,
    )

    print("\n=== SEARCHING HYPERPARAMETERS ===")

    search_started = perf_counter()
    search.fit(X_tune, y_tune)
    search_time = perf_counter() - search_started

    print("\n=== BEST SEARCH RESULT ===")
    print(f"Best parameters: {search.best_params_}")
    print(f"Best CV Macro F1: {search.best_score_:.4f}")
    print(f"Search time: {search_time:.2f} seconds")

    # Refit the selected configuration using all training records.
    final_model = clone(search.best_estimator_)

    print("\n=== REFITTING ON FULL TRAINING DATA ===")

    fit_started = perf_counter()
    final_model.fit(split.X_train, split.y_train)
    final_fit_time = perf_counter() - fit_started

    predictions = final_model.predict(split.X_test)

    train_accuracy = final_model.score(
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

    print("\n=== TUNED LOGISTIC REGRESSION RESULTS ===")
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Testing accuracy:  {test_accuracy:.4f}")
    print(f"Macro precision:   {precision:.4f}")
    print(f"Macro recall:      {recall:.4f}")
    print(f"Macro F1-score:    {macro_f1:.4f}")
    print(f"Final fit time:    {final_fit_time:.2f} seconds")

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


if __name__ == "__main__":
    main()

