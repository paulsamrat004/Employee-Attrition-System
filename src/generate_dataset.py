"""Generate the reproducible synthetic HR dataset used by this project."""
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "employee_data.csv"

def generate(n=1800, seed=42):
    rng = np.random.default_rng(seed)
    age = rng.integers(21, 61, n)
    department = rng.choice(["Sales","Engineering","HR","Marketing","Finance"], n, p=[.25,.30,.15,.15,.15])
    income = np.clip(rng.normal(65000, 22000, n), 25000, 150000).round().astype(int)
    tenure = np.clip(rng.gamma(2.2, 3, n), .2, 20).round(1)
    job_sat = rng.integers(1, 6, n)
    work_life = rng.integers(1, 6, n)
    performance = rng.integers(1, 6, n)
    overtime = rng.choice(["Yes","No"], n, p=[.32,.68])
    distance = np.clip(rng.gamma(2.2, 4, n), 1, 40).round(1)
    score = (0.45 + .75*(overtime=="Yes") + .28*(job_sat<=2)
             + .20*(work_life<=2) + .16*(performance<=2) + .025*distance
             + .000012*(80000-income) + .10*(tenure<2) - .015*(age-35)
             + .12*(department=="Sales") - .12*(department=="Engineering"))
    p = 1/(1+np.exp(-score))
    attrition = np.where(rng.random(n) < p, "Yes", "No")
    df = pd.DataFrame({
        "EmployeeID":[f"E{100000+i}" for i in range(n)],
        "Age":age, "Department":department, "MonthlyIncome":income,
        "YearsAtCompany":tenure, "JobSatisfaction":job_sat,
        "WorkLifeBalance":work_life, "PerformanceRating":performance,
        "OverTime":overtime, "DistanceFromHome":distance, "Attrition":attrition
    })
    for col in ["MonthlyIncome","JobSatisfaction","WorkLifeBalance","DistanceFromHome"]:
        idx=rng.choice(n, size=round(n*.015), replace=False)
        df.loc[idx,col]=np.nan
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT,index=False)
    return df

if __name__ == "__main__":
    print(generate().head())
    print(f"Saved {OUT}")
