# Employee Attrition Prediction System

## 1. Project overview
A production-style machine learning project that predicts whether an employee is likely to leave a company from workplace, performance and demographic factors. It includes data preprocessing, exploratory data analysis, model comparison/tuning, evaluation, a reusable saved model, and a browser-based prediction interface.

## 2. Core project context
The project brief requires:
- Python + Pandas, NumPy and Scikit-learn
- HR-style dataset with EmployeeID, Age, Department, MonthlyIncome, YearsAtCompany, JobSatisfaction, WorkLifeBalance, PerformanceRating, OverTime, DistanceFromHome and Attrition
- Missing-value handling, categorical encoding, numerical scaling and class-imbalance handling
- EDA for department, overtime, job satisfaction, correlations and important factors
- At least four plots: department attrition, income vs attrition, numerical correlation heatmap, and model feature importance
- At least two ML models, hyperparameter tuning, and evaluation with Accuracy, Precision, Recall, F1-score and ROC-AUC
- Model persistence with Joblib/Pickle
- Input validation and graceful error handling
- Modular code instead of one giant script
- README documentation and GitHub-ready structure
- An interactive prediction interface

This implementation uses Logistic Regression and Random Forest, `class_weight="balanced"`, a leakage-safe Scikit-learn Pipeline, GridSearchCV, and a Flask API serving a custom HTML/CSS/JavaScript UI.

## 3. Dataset
`data/employee_data.csv` is a reproducible synthetic HR-style dataset generated for this project. It contains 1,800 records and the exact requested feature names. A small amount of missing data is intentionally included so the imputation step is exercised.

> Synthetic data is for project demonstration/testing and should not be treated as real HR evidence or used for employment decisions.

## 4. Project structure
```text
employee-attrition-prediction/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── employee_data.csv
├── models/
│   └── (generated after training)
├── visuals/
│   └── (generated after training)
├── notebooks/
│   └── EDA_and_Modeling.ipynb
├── src/
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── train_model.py
│   ├── predict.py
│   └── utils.py
└── frontend/
    ├── index.html
    ├── style.css
    └── script.js
```

## 5. Data preprocessing
- Numerical missing values: median imputation.
- Categorical missing values: most-frequent imputation.
- Department and OverTime: One-Hot Encoding.
- Numerical features: StandardScaler.
- Class imbalance: balanced class weights.
- EmployeeID is retained in the dataset as a unique identifier but is excluded from model features.
- Train/test split is stratified 80/20.
- Preprocessing is inside the model Pipeline, preventing test-set leakage.

## 6. EDA
The training script creates:
1. `visuals/01_attrition_by_department.png`
2. `visuals/02_attrition_by_overtime.png`
3. `visuals/03_attrition_by_job_satisfaction.png`
4. `visuals/04_income_vs_attrition.png`
5. `visuals/05_correlation_heatmap.png`
6. `visuals/06_feature_importance.png`
7. `visuals/07_roc_curve_comparison.png`

## 7. Models and tuning
### Logistic Regression
Grid searched over `C = [0.1, 1, 10]`.

### Random Forest
Grid searched over:
- `n_estimators = [150, 300]`
- `max_depth = [None, 8, 14]`
- `min_samples_leaf = [1, 3]`

Both use 5-fold cross-validation and ROC-AUC as the tuning objective.

## 8. Evaluation
The held-out test set is used for:
- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

The model with the highest test ROC-AUC is saved as `models/best_model.pkl`. The exact results are written to `models/model_metrics.json`.

## 9. Employee ID evaluation
The web interface now includes an **Employee ID** section. Employee ID is treated as a unique identifier, not as a machine-learning feature.

When an existing Employee ID from `data/employee_data.csv` is entered:
- The matching employee profile is loaded automatically.
- The trained model evaluates that employee profile.
- The prediction and probability are displayed with the Employee ID.
- If the dataset contains an `Attrition` label for that employee, the UI also displays the actual dataset label and whether the prediction matches it.

For a new Employee ID that is not in the dataset, the user can enter the profile fields manually. In that case there is no stored actual label to compare against.

Employee ID is deliberately excluded from `FEATURES`, so the model does not learn from the identifier itself.

## 10. Web interface
The Flask application serves a responsive HR analytics interface. The user enters:
- Employee ID
- Age
- Department
- Monthly Income
- Years at Company
- Job Satisfaction
- Work-Life Balance
- Performance Rating
- OverTime
- Distance From Home

The UI returns the Employee ID, predicted Attrition (Yes/No), probability, and—when the ID exists in the dataset—the stored actual Attrition label and prediction-match status.

## 11. Run the project

### Windows PowerShell
```powershell
cd employee-attrition-prediction
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/train_model.py
python app.py
```

Open:
```text
http://127.0.0.1:5000
```

### Windows Command Prompt
```bat
cd employee-attrition-prediction
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python src/train_model.py
python app.py
```

### macOS/Linux
```bash
cd employee-attrition-prediction
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python src/train_model.py
python app.py
```

## 12. Optional notebook
```bash
jupyter notebook notebooks/EDA_and_Modeling.ipynb
```

## 13. Submission checklist
- [x] Proper folder structure
- [x] Clean modular source code
- [x] Synthetic HR-style dataset
- [x] Missing-value handling
- [x] Categorical encoding
- [x] Numerical scaling
- [x] Class imbalance handling
- [x] EDA
- [x] Four required plot types
- [x] Two ML models
- [x] Hyperparameter tuning
- [x] Accuracy, Precision, Recall, F1 and ROC-AUC
- [x] Saved model with Joblib
- [x] Input validation
- [x] Graceful model/file errors
- [x] Interactive HTML/CSS/JavaScript UI
- [x] README
- [x] Notebook
