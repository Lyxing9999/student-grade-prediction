from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


def create_adaboost() -> Pipeline:
    """Create the baseline AdaBoost classification pipeline."""

    weak_learner = DecisionTreeClassifier(
        max_depth=1,
        random_state=42,
    )

    return Pipeline(
        steps=[
            (
                "model",
                AdaBoostClassifier(
                    estimator=weak_learner,
                    n_estimators=50,
                    learning_rate=1.0,
                    random_state=42,
                ),
            ),
        ]
    )