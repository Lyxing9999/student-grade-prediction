from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay


METRICS_DIR = Path("outputs/metrics")
FIGURES_DIR = Path("outputs/figures")

GRADE_LABELS = ["A", "B", "C", "D", "F"]


BASELINE_RESULTS = [
    {
        "model": "Logistic Regression",
        "accuracy": 0.6958,
        "macro_precision": 0.4685,
        "macro_recall": 0.4374,
        "macro_f1": 0.4496,
        "fit_time_seconds": 2.77,
    },
    {
        "model": "Decision Tree",
        "accuracy": 0.6089,
        "macro_precision": 0.3979,
        "macro_recall": 0.3942,
        "macro_f1": 0.3960,
        "fit_time_seconds": 2.53,
    },
    {
        "model": "Random Forest",
        "accuracy": 0.6926,
        "macro_precision": 0.4642,
        "macro_recall": 0.4371,
        "macro_f1": 0.4479,
        "fit_time_seconds": 3.69,
    },
    {
        "model": "AdaBoost",
        "accuracy": 0.6804,
        "macro_precision": 0.4631,
        "macro_recall": 0.4348,
        "macro_f1": 0.4200,
        "fit_time_seconds": 65.67,
    },
]


TUNED_RESULTS = [
    {
        "model": "Logistic Regression",
        "accuracy": 0.6442,
        "macro_precision": 0.4380,
        "macro_recall": 0.5573,
        "macro_f1": 0.4535,
        "f_recall": 0.72,
        "fit_time_seconds": 3.07,
    },
    {
        "model": "Decision Tree",
        "accuracy": 0.6460,
        "macro_precision": 0.4377,
        "macro_recall": 0.5571,
        "macro_f1": 0.4503,
        "f_recall": 0.75,
        "fit_time_seconds": 2.61,
    },
    {
        "model": "Random Forest",
        "accuracy": 0.6325,
        "macro_precision": 0.4411,
        "macro_recall": 0.5535,
        "macro_f1": 0.4552,
        "f_recall": 0.66,
        "fit_time_seconds": 7.39,
    },
    {
        "model": "AdaBoost",
        "accuracy": 0.6387,
        "macro_precision": 0.4341,
        "macro_recall": 0.5539,
        "macro_f1": 0.4496,
        "f_recall": 0.69,
        "fit_time_seconds": 278.33,
    },
]


TUNED_CONFUSION_MATRICES = {
    "Logistic Regression": np.array(
        [
            [85180, 22314, 2120, 108, 7],
            [9752, 27850, 11964, 1812, 257],
            [548, 7508, 11933, 5746, 2661],
            [5, 365, 2105, 2992, 3532],
            [0, 1, 79, 270, 891],
        ]
    ),
    "Decision Tree": np.array(
        [
            [86116, 21096, 2430, 76, 11],
            [10250, 26548, 13110, 1413, 314],
            [600, 7032, 13104, 4672, 2988],
            [6, 329, 2409, 2495, 3760],
            [0, 1, 88, 224, 928],
        ]
    ),
    "Random Forest": np.array(
        [
            [80074, 27676, 1843, 131, 5],
            [7418, 31084, 10781, 2143, 209],
            [347, 8186, 11036, 6523, 2304],
            [2, 419, 1940, 3497, 3141],
            [0, 3, 70, 351, 817],
        ]
    ),
    "AdaBoost": np.array(
        [
            [83964, 23708, 1898, 153, 6],
            [9113, 29067, 10690, 2532, 233],
            [488, 7871, 10351, 7171, 2515],
            [4, 398, 1718, 3505, 3374],
            [0, 2, 57, 329, 853],
        ]
    ),
}


def save_comparison_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline = pd.DataFrame(BASELINE_RESULTS)
    tuned = pd.DataFrame(TUNED_RESULTS)

    baseline.to_csv(
        METRICS_DIR / "baseline_results.csv",
        index=False,
    )

    tuned = tuned.sort_values(
        "macro_f1",
        ascending=False,
    )

    tuned.to_csv(
        METRICS_DIR / "tuned_results.csv",
        index=False,
    )

    combined = baseline[
        ["model", "accuracy", "macro_f1"]
    ].merge(
        tuned[["model", "accuracy", "macro_f1"]],
        on="model",
        suffixes=("_baseline", "_tuned"),
    )

    combined["macro_f1_improvement"] = (
        combined["macro_f1_tuned"]
        - combined["macro_f1_baseline"]
    )

    combined.to_csv(
        METRICS_DIR / "baseline_vs_tuned.csv",
        index=False,
    )

    return tuned, combined


def save_tuned_comparison_chart(tuned: pd.DataFrame) -> None:
    chart_data = tuned.set_index("model")[
        ["accuracy", "macro_f1"]
    ]

    ax = chart_data.plot(
        kind="bar",
        figsize=(10, 6),
    )

    ax.set_title("Tuned Model Performance")
    ax.set_xlabel("Model")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "tuned_model_comparison.png",
        dpi=200,
    )
    plt.close()


def save_improvement_chart(combined: pd.DataFrame) -> None:
    chart_data = combined.set_index("model")[
        ["macro_f1_baseline", "macro_f1_tuned"]
    ]

    ax = chart_data.plot(
        kind="bar",
        figsize=(10, 6),
    )

    ax.set_title("Baseline vs Tuned Macro F1")
    ax.set_xlabel("Model")
    ax.set_ylabel("Macro F1-score")
    ax.set_ylim(0, 0.55)
    ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "baseline_vs_tuned_macro_f1.png",
        dpi=200,
    )
    plt.close()


def save_confusion_matrices() -> None:
    for model_name, matrix in TUNED_CONFUSION_MATRICES.items():
        display = ConfusionMatrixDisplay(
            confusion_matrix=matrix,
            display_labels=GRADE_LABELS,
        )

        display.plot(
            values_format="d",
        )

        plt.title(f"{model_name} — Tuned Confusion Matrix")
        plt.tight_layout()

        safe_name = model_name.lower().replace(" ", "_")

        plt.savefig(
            FIGURES_DIR
            / f"{safe_name}_tuned_confusion_matrix.png",
            dpi=200,
        )

        plt.close()


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    tuned, combined = save_comparison_tables()
    save_tuned_comparison_chart(tuned)
    save_improvement_chart(combined)
    save_confusion_matrices()

    winner = tuned.iloc[0]

    print("\n=== FINAL TUNED MODEL COMPARISON ===")
    print(tuned.to_string(index=False))

    print("\n=== SELECTED MODEL ===")
    print(f"Model: {winner['model']}")
    print(f"Macro F1: {winner['macro_f1']:.4f}")
    print(f"Accuracy: {winner['accuracy']:.4f}")

    print("\nResults and figures saved successfully.")


if __name__ == "__main__":
    main()
