from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.config import FEATURE_COLUMNS
from src.data_loader import load_dataset


FIGURES_DIR = Path("outputs/figures")
METRICS_DIR = Path("outputs/metrics")

GRADE_ORDER = ["A", "B", "C", "D", "F"]

FEATURE_LABELS = {
    "weekly_self_study_hours": "Weekly Self-Study Hours",
    "attendance_percentage": "Attendance Percentage",
    "class_participation": "Class Participation",
}


def save_grade_distribution(df: pd.DataFrame) -> None:
    grade_counts = (
        df["grade"]
        .value_counts()
        .reindex(GRADE_ORDER, fill_value=0)
    )

    ax = grade_counts.plot(
        kind="bar",
        figsize=(8, 5),
    )

    ax.set_title("Distribution of Student Grades")
    ax.set_xlabel("Grade")
    ax.set_ylabel("Number of Students")
    ax.tick_params(axis="x", rotation=0)

    plt.tight_layout()
    plt.savefig(
        FIGURES_DIR / "grade_distribution.png",
        dpi=200,
    )
    plt.close()


def save_feature_histograms(df: pd.DataFrame) -> None:
    for feature in FEATURE_COLUMNS:
        ax = df[feature].plot(
            kind="hist",
            bins=30,
            figsize=(8, 5),
        )

        label = FEATURE_LABELS[feature]

        ax.set_title(f"Distribution of {label}")
        ax.set_xlabel(label)
        ax.set_ylabel("Number of Students")

        plt.tight_layout()
        plt.savefig(
            FIGURES_DIR / f"{feature}_distribution.png",
            dpi=200,
        )
        plt.close()


def save_feature_boxplots(df: pd.DataFrame) -> None:
    # A sample is enough for visualization and keeps plotting fast.
    sample_size = min(50_000, len(df))

    sample = df.sample(
        n=sample_size,
        random_state=42,
    ).copy()

    sample["grade"] = pd.Categorical(
        sample["grade"],
        categories=GRADE_ORDER,
        ordered=True,
    )

    sample = sample.sort_values("grade")

    for feature in FEATURE_COLUMNS:
        label = FEATURE_LABELS[feature]

        sample.boxplot(
            column=feature,
            by="grade",
            figsize=(8, 5),
        )

        plt.title(f"{label} by Grade")
        plt.suptitle("")
        plt.xlabel("Grade")
        plt.ylabel(label)

        plt.tight_layout()
        plt.savefig(
            FIGURES_DIR / f"{feature}_by_grade.png",
            dpi=200,
        )
        plt.close()


def save_summary_tables(df: pd.DataFrame) -> None:
    grade_summary = (
        df.groupby("grade", observed=True)[FEATURE_COLUMNS]
        .agg(["mean", "median", "std", "min", "max"])
        .reindex(GRADE_ORDER)
    )

    grade_summary.to_csv(
        METRICS_DIR / "feature_summary_by_grade.csv"
    )

    grade_distribution = (
        df["grade"]
        .value_counts()
        .reindex(GRADE_ORDER, fill_value=0)
        .rename("count")
        .to_frame()
    )

    grade_distribution["percentage"] = (
        grade_distribution["count"]
        / len(df)
        * 100
    )

    grade_distribution.to_csv(
        METRICS_DIR / "grade_distribution.csv"
    )

    correlation = df[
        [
            "weekly_self_study_hours",
            "attendance_percentage",
            "class_participation",
            "total_score",
        ]
    ].corr()

    correlation.to_csv(
        METRICS_DIR / "feature_correlation.csv"
    )

    print("\n=== GRADE DISTRIBUTION ===")
    print(grade_distribution)

    print("\n=== MEAN FEATURE VALUES BY GRADE ===")
    print(
        df.groupby(
            "grade",
            observed=True,
        )[FEATURE_COLUMNS]
        .mean()
        .reindex(GRADE_ORDER)
    )

    print("\n=== CORRELATION MATRIX ===")
    print(correlation)


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset()

    save_grade_distribution(df)
    save_feature_histograms(df)
    save_feature_boxplots(df)
    save_summary_tables(df)

    print("\nEDA completed successfully.")
    print(f"Figures saved to: {FIGURES_DIR.resolve()}")
    print(f"Tables saved to: {METRICS_DIR.resolve()}")


if __name__ == "__main__":
    main()
