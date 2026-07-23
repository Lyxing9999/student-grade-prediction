from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "student_performance.csv"
)

FEATURE_COLUMNS = [
    "weekly_self_study_hours",
    "attendance_percentage",
    "class_participation",
]

TARGET_COLUMN = "grade"

EXCLUDED_COLUMNS = [
    "student_id",
    "total_score",
]

RANDOM_STATE = 42
TEST_SIZE = 0.20
