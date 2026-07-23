from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


def create_decision_tree() -> Pipeline:
    """Create the baseline Decision Tree pipeline."""

    return Pipeline(
        steps=[
            (
                "model",
                DecisionTreeClassifier(
                    random_state=42,
                ),
            ),
        ]
    )
