"""Definition of the final selected machine-learning model."""

from typing import Final

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


FEATURE_COLUMNS: Final[list[str]] = [
    "weekly_self_study_hours",
    "attendance_percentage",
    "class_participation",
]

GRADE_ORDER: Final[list[str]] = ["A", "B", "C", "D", "F"]


def create_final_model() -> Pipeline:
    """Create the tuned Random Forest selected by Macro F1-score."""

    classifier = RandomForestClassifier(
        n_estimators=150,
        max_depth=12,
        min_samples_split=2,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline(
        steps=[
            ("model", classifier),
        ]
    )


def create_student_input(
    *,
    weekly_self_study_hours: float,
    attendance_percentage: float,
    class_participation: float,
) -> pd.DataFrame:
    """Create one correctly ordered model input row."""

    return pd.DataFrame(
        [
            {
                "weekly_self_study_hours": weekly_self_study_hours,
                "attendance_percentage": attendance_percentage,
                "class_participation": class_participation,
            }
        ],
        columns=FEATURE_COLUMNS,
    )
