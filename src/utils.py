from pathlib import Path
import json
import joblib

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "employee_data.csv"
MODEL_PATH = ROOT_DIR / "models" / "best_model.pkl"
METRICS_PATH = ROOT_DIR / "models" / "model_metrics.json"

FEATURES = [
    "Age", "Department", "MonthlyIncome", "YearsAtCompany",
    "JobSatisfaction", "WorkLifeBalance", "PerformanceRating",
    "OverTime", "DistanceFromHome"
]
CATEGORICAL_FEATURES = ["Department", "OverTime"]
NUMERICAL_FEATURES = [
    "Age", "MonthlyIncome", "YearsAtCompany", "JobSatisfaction",
    "WorkLifeBalance", "PerformanceRating", "DistanceFromHome"
]
DEPARTMENTS = ["Sales", "Engineering", "HR", "Marketing", "Finance"]
BINARY_YES_NO = ["Yes", "No"]

def save_json(payload, path=METRICS_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))

def load_model(path=MODEL_PATH):
    return joblib.load(path)
