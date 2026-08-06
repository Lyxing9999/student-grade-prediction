"""Prediction logic shared by the user interface."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib

from src.final_model import (
    GRADE_ORDER,
    create_student_input,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "final_random_forest.joblib"
)


@dataclass(frozen=True)
class GradePrediction:
    """One student-grade prediction result."""

    predicted_grade: str
    probabilities: dict[str, float]
    model_name: str
    metrics: dict[str, float]


class GradePredictionService:
    """Load the final model and produce grade predictions."""

    def __init__(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
    ) -> None:
        if not model_path.exists():
            raise FileNotFoundError(
                "Final model not found. Run:\n"
                "python -m scripts.train_final_model"
            )

        artifact: dict[str, Any] = joblib.load(
            model_path
        )

        self._model = artifact["model"]
        self._model_name = artifact["model_name"]
        self._metrics = artifact.get("metrics", {})

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def metrics(self) -> dict[str, float]:
        return dict(self._metrics)

    def predict(
        self,
        *,
        weekly_self_study_hours: float,
        attendance_percentage: float,
        class_participation: float,
    ) -> GradePrediction:
        input_frame = create_student_input(
            weekly_self_study_hours=(
                weekly_self_study_hours
            ),
            attendance_percentage=(
                attendance_percentage
            ),
            class_participation=(
                class_participation
            ),
        )

        predicted_grade = str(
            self._model.predict(input_frame)[0]
        )

        probability_values = self._model.predict_proba(
            input_frame
        )[0]

        classifier = self._model.named_steps["model"]

        probability_lookup = {
            str(grade): float(probability)
            for grade, probability in zip(
                classifier.classes_,
                probability_values,
                strict=True,
            )
        }

        ordered_probabilities = {
            grade: probability_lookup.get(grade, 0.0)
            for grade in GRADE_ORDER
        }

        return GradePrediction(
            predicted_grade=predicted_grade,
            probabilities=ordered_probabilities,
            model_name=self._model_name,
            metrics=dict(self._metrics),
        )
