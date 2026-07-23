from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline


def create_random_forest() -> Pipeline:
    """Create a resource-safe Random Forest baseline."""

    return Pipeline(
        steps=[
            (
                "model",
                RandomForestClassifier(
                    n_estimators=50,
                    max_depth=20,
                    min_samples_leaf=5,
                    max_features="sqrt",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
