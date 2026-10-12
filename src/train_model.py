import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, RocCurveDisplay
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
import joblib

# Allow: python src/train_model.py from project root
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.preprocessing import build_preprocessor, validate_dataframe
from src.utils import DATA_PATH, MODEL_PATH, METRICS_PATH, FEATURES, save_json

RANDOM_STATE = 42

def load_data():
    df = pd.read_csv(DATA_PATH)
    validate_dataframe(df)
    return df

def evaluate(model, X_test, y_test):
    pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "precision": round(precision_score(y_test, pred, zero_division=0), 4),
        "recall": round(recall_score(y_test, pred, zero_division=0), 4),
        "f1_score": round(f1_score(y_test, pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_test, proba), 4),
    }

def save_eda_plots(df, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 1. Attrition rate by department
    rate = pd.crosstab(df["Department"], df["Attrition"], normalize="index")["Yes"].sort_values(ascending=False)
    plt.figure(figsize=(9, 5))
    rate.mul(100).plot(kind="bar")
    plt.ylabel("Attrition rate (%)")
    plt.xlabel("Department")
    plt.title("Attrition Rate by Department")
    plt.tight_layout()
    plt.savefig(output_dir / "01_attrition_by_department.png", dpi=160)
    plt.close()

    # Additional EDA: overtime and job satisfaction, as required by the brief.
    overtime_rate = pd.crosstab(df["OverTime"], df["Attrition"], normalize="index")["Yes"].sort_values(ascending=False)
    plt.figure(figsize=(7, 5))
    overtime_rate.mul(100).plot(kind="bar")
    plt.ylabel("Attrition rate (%)")
    plt.xlabel("OverTime")
    plt.title("Attrition Rate by Overtime Status")
    plt.tight_layout()
    plt.savefig(output_dir / "02_attrition_by_overtime.png", dpi=160)
    plt.close()

    satisfaction_rate = pd.crosstab(df["JobSatisfaction"], df["Attrition"], normalize="index")["Yes"].sort_index()
    plt.figure(figsize=(7, 5))
    satisfaction_rate.mul(100).plot(kind="bar")
    plt.ylabel("Attrition rate (%)")
    plt.xlabel("Job Satisfaction (1–5)")
    plt.title("Attrition Rate by Job Satisfaction")
    plt.tight_layout()
    plt.savefig(output_dir / "03_attrition_by_job_satisfaction.png", dpi=160)
    plt.close()

    # 4. Monthly income vs attrition
    plot_df = df.copy()
    plot_df["MonthlyIncome"] = plot_df["MonthlyIncome"].fillna(plot_df["MonthlyIncome"].median())
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=plot_df, x="Attrition", y="MonthlyIncome")
    plt.title("Monthly Income vs Attrition")
    plt.tight_layout()
    plt.savefig(output_dir / "04_income_vs_attrition.png", dpi=160)
    plt.close()

    # 5. Correlation heatmap, including binary Attrition for direct correlation inspection.
    numeric = df.select_dtypes(include="number").copy()
    numeric["AttritionBinary"] = df["Attrition"].map({"No": 0, "Yes": 1})
    plt.figure(figsize=(9, 6))
    sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Correlation Heatmap of Numerical Features")
    plt.tight_layout()
    plt.savefig(output_dir / "05_correlation_heatmap.png", dpi=160)
    plt.close()

def save_feature_importance(model, X_test, y_test, output_dir):
    result = permutation_importance(
        model, X_test, y_test, scoring="roc_auc",
        n_repeats=10, random_state=RANDOM_STATE
    )
    imp = pd.Series(result.importances_mean, index=X_test.columns).sort_values()
    plt.figure(figsize=(9, 5))
    imp.tail(10).plot(kind="barh")
    plt.xlabel("Mean decrease in ROC-AUC")
    plt.title("Feature Importance (Permutation Importance)")
    plt.tight_layout()
    plt.savefig(output_dir / "06_feature_importance.png", dpi=160)
    plt.close()
    return imp.sort_values(ascending=False).to_dict()

def train():
    df = load_data()
    X = df[FEATURES]
    y = df["Attrition"].map({"No": 0, "Yes": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    candidates = {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
            {"model__C": [0.1, 1, 10]}
        ),
        "Random Forest": (
            RandomForestClassifier(
                class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
            ),
            {
                "model__n_estimators": [150, 300],
                "model__max_depth": [None, 8, 14],
                "model__min_samples_leaf": [1, 3]
            }
        )
    }

    results = {}
    fitted = {}
    for name, (estimator, params) in candidates.items():
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("model", estimator),
        ])
        search = GridSearchCV(
            pipeline, params, scoring="roc_auc",
            cv=5, n_jobs=-1, refit=True
        )
        search.fit(X_train, y_train)
        results[name] = {
            "metrics": evaluate(search.best_estimator_, X_test, y_test),
            "best_params": search.best_params_
        }
        fitted[name] = search.best_estimator_

    best_name = max(results, key=lambda name: results[name]["metrics"]["roc_auc"])
    best_model = fitted[best_name]
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODEL_PATH)

    feature_importance = save_feature_importance(
        best_model, X_test, y_test, ROOT_DIR / "visuals"
    )
    save_eda_plots(df, ROOT_DIR / "visuals")

    # ROC curve
    plt.figure(figsize=(7, 5))
    for name, model in fitted.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, name=name)
    plt.title("ROC-AUC Comparison")
    plt.tight_layout()
    plt.savefig(ROOT_DIR / "visuals" / "07_roc_curve_comparison.png", dpi=160)
    plt.close()

    metrics = {
        "best_model": best_name,
        "test_size": 0.20,
        "random_state": RANDOM_STATE,
        "class_distribution": y.value_counts().to_dict(),
        "models": results,
        "feature_importance": feature_importance,
        "features_used": FEATURES,
    }
    save_json(metrics)
    print(json.dumps(metrics, indent=2))
    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")

if __name__ == "__main__":
    train()
