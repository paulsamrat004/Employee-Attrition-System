import sys
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.utils import FEATURES, MODEL_PATH, DATA_PATH, DEPARTMENTS, load_model


def get_employee_by_id(employee_id):
    """Return the stored employee profile for an EmployeeID, or None."""
    employee_id = str(employee_id).strip()
    if not employee_id:
        raise ValueError("Employee ID is required.")

    if not DATA_PATH.exists():
        raise FileNotFoundError("Employee dataset not found.")

    df = pd.read_csv(DATA_PATH)
    matches = df[df["EmployeeID"].astype(str).str.strip() == employee_id]
    if matches.empty:
        return None

    row = matches.iloc[0]
    profile = {feature: row[feature] for feature in FEATURES}
    # Convert NumPy/Pandas scalar values to normal JSON-friendly Python values.
    for key in ["Age", "MonthlyIncome", "YearsAtCompany", "DistanceFromHome"]:
        profile[key] = float(profile[key])
    for key in ["JobSatisfaction", "WorkLifeBalance", "PerformanceRating"]:
        profile[key] = int(float(profile[key]))
    profile["Department"] = str(profile["Department"])
    profile["OverTime"] = str(profile["OverTime"])

    actual = str(row["Attrition"]) if "Attrition" in df.columns else None
    return {"employee_id": employee_id, "profile": profile, "actual_attrition": actual}


def validate_employee_id(employee_id):
    employee_id = str(employee_id).strip()
    if not employee_id:
        raise ValueError("Employee ID is required.")
    if len(employee_id) > 50:
        raise ValueError("Employee ID must be 50 characters or fewer.")
    return employee_id


def validate_input(data):
    employee_id = validate_employee_id(data.get("EmployeeID", ""))
    missing = [f for f in FEATURES if f not in data]
    if missing:
        raise ValueError(f"Missing fields: {missing}")

    age = float(data["Age"])
    income = float(data["MonthlyIncome"])
    years = float(data["YearsAtCompany"])
    distance = float(data["DistanceFromHome"])
    for label, value in [("Age", age), ("MonthlyIncome", income), ("YearsAtCompany", years), ("DistanceFromHome", distance)]:
        if value < 0:
            raise ValueError(f"{label} cannot be negative.")
    if not 16 <= age <= 100:
        raise ValueError("Age must be between 16 and 100.")
    if not 1 <= int(data["JobSatisfaction"]) <= 5:
        raise ValueError("JobSatisfaction must be between 1 and 5.")
    if not 1 <= int(data["WorkLifeBalance"]) <= 5:
        raise ValueError("WorkLifeBalance must be between 1 and 5.")
    if not 1 <= int(data["PerformanceRating"]) <= 5:
        raise ValueError("PerformanceRating must be between 1 and 5.")
    if data["Department"] not in DEPARTMENTS:
        raise ValueError(f"Department must be one of: {', '.join(DEPARTMENTS)}")
    if data["OverTime"] not in {"Yes", "No"}:
        raise ValueError("OverTime must be Yes or No.")
    return employee_id


def predict_attrition(data):
    """Predict an employee and associate the prediction with EmployeeID.

    If the EmployeeID exists in the supplied dataset, its stored profile is used
    automatically. This also returns the stored Attrition value so the UI can
    compare the model prediction with the dataset's actual label.

    For a new EmployeeID, all model feature fields must be supplied manually.
    EmployeeID itself is never used as a model feature.
    """
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Trained model not found. Run: python src/train_model.py")

    employee_id = validate_employee_id(data.get("EmployeeID", ""))
    stored = get_employee_by_id(employee_id)

    if stored is not None:
        # ID determines which stored employee profile is evaluated.
        profile = stored["profile"]
        actual_attrition = stored["actual_attrition"]
    else:
        # New IDs are allowed, but the model still needs the employee profile.
        validate_input(data)
        profile = {key: data[key] for key in FEATURES}
        actual_attrition = None

    row = dict(profile)
    row["Age"] = float(row["Age"])
    row["MonthlyIncome"] = float(row["MonthlyIncome"])
    row["YearsAtCompany"] = float(row["YearsAtCompany"])
    row["JobSatisfaction"] = int(row["JobSatisfaction"])
    row["WorkLifeBalance"] = int(row["WorkLifeBalance"])
    row["PerformanceRating"] = int(row["PerformanceRating"])
    row["DistanceFromHome"] = float(row["DistanceFromHome"])
    X = pd.DataFrame([row], columns=FEATURES)

    model = load_model()
    probability = float(model.predict_proba(X)[0, 1])
    label = "Yes" if probability >= 0.5 else "No"

    result = {
        "employee_id": employee_id,
        "attrition": label,
        "probability": round(probability, 4),
        "profile": profile,
        "employee_found": stored is not None,
    }
    if actual_attrition is not None:
        result["actual_attrition"] = actual_attrition
        result["prediction_correct"] = label == actual_attrition
    else:
        result["actual_attrition"] = None
        result["prediction_correct"] = None
    return result
