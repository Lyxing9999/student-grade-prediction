from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


OUTPUT_METRICS = Path("outputs/metrics")
OUTPUT_FIGURES = Path("outputs/figures")


def main() -> None:
    OUTPUT_METRICS.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)

    results = pd.DataFrame(
        [
            {
                "model": "Logistic Regression",
                "train_accuracy": 0.6958,
                "test_accuracy": 0.6958,
                "macro_precision": 0.4685,
                "macro_recall": 0.4374,
                "macro_f1": 0.4496,
                "training_time_seconds": 2.77,
            },
            {
                "model": "Decision Tree",
                "train_accuracy": 0.9594,
                "test_accuracy": 0.6089,
                "macro_precision": 0.3979,
                "macro_recall": 0.3942,
                "macro_f1": 0.3960,
                "training_time_seconds": 2.53,
            },
            {
                "model": "Random Forest",
                "train_accuracy": 0.7370,
                "test_accuracy": 0.6926,
                "macro_precision": 0.4642,
                "macro_recall": 0.4371,
                "macro_f1": 0.4479,
                "training_time_seconds": 3.69,
            },
            {
                "model": "AdaBoost",
                "train_accuracy": 0.6803,
                "test_accuracy": 0.6804,
                "macro_precision": 0.4631,
                "macro_recall": 0.4348,
                "macro_f1": 0.4200,
                "training_time_seconds": 65.67,
            },
        ]
    )

    results["train_test_gap"] = (
        results["train_accuracy"]
        - results["test_accuracy"]
    )

    results = results.sort_values(
        by="macro_f1",
        ascending=False,
    )

    results.to_csv(
        OUTPUT_METRICS / "baseline_model_comparison.csv",
        index=False,
    )

    print("\n=== BASELINE MODEL COMPARISON ===")
    print(results.to_string(index=False))

    ax = results.plot(
        x="model",
        y=["test_accuracy", "macro_f1"],
        kind="bar",
        figsize=(10, 6),
    )

    ax.set_title("Baseline Model Performance")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.savefig(
        OUTPUT_FIGURES / "baseline_model_comparison.png",
        dpi=200,
    )
    plt.close()

    print("\nSaved:")
    print(
        OUTPUT_METRICS
        / "baseline_model_comparison.csv"
    )
    print(
        OUTPUT_FIGURES
        / "baseline_model_comparison.png"
    )


if __name__ == "__main__":
    main()
