from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_logistic_regression() -> Pipeline:
    """Create a Logistic Regression classification pipeline."""

    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    random_state=42,
                ),
            ),
        ]
    )