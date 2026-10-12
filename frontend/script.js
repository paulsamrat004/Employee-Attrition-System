const form = document.getElementById("predictionForm");
const errorBox = document.getElementById("error");
const resultLabel = document.getElementById("resultLabel");
const probBar = document.getElementById("probBar");
const probText = document.getElementById("probText");
const predictBtn = document.getElementById("predictBtn");
const employeeId = document.getElementById("employeeId");
const lookupBtn = document.getElementById("lookupBtn");
const lookupStatus = document.getElementById("lookupStatus");
const employeeResultId = document.getElementById("employeeResultId");
const comparisonText = document.getElementById("comparisonText");

const numericFields = ["Age","MonthlyIncome","YearsAtCompany","JobSatisfaction","WorkLifeBalance","PerformanceRating","DistanceFromHome"];

function fillProfile(profile){
  Object.entries(profile).forEach(([key, value]) => {
    if (form.elements[key]) form.elements[key].value = value;
  });
}

async function loadEmployee(){
  const id = employeeId.value.trim();
  errorBox.textContent = "";
  lookupStatus.textContent = "";
  if (!id) { lookupStatus.textContent = "Enter an Employee ID first."; return; }

  lookupBtn.disabled = true;
  lookupBtn.textContent = "Loading…";
  try {
    const response = await fetch(`/api/employee/${encodeURIComponent(id)}`);
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Employee ID not found.");
    fillProfile(data.profile);
    lookupStatus.textContent = `Employee ${data.employee_id} loaded successfully. Click Evaluate Employee to get the model result.`;
    lookupStatus.className = "lookup-status success-text";
  } catch (err) {
    lookupStatus.textContent = `${err.message} For a new Employee ID, enter the profile fields manually.`;
    lookupStatus.className = "lookup-status warning-text";
  } finally {
    lookupBtn.disabled = false;
    lookupBtn.textContent = "Load Employee";
  }
}

function formDataToObject(form){
  const data = Object.fromEntries(new FormData(form).entries());
  numericFields.forEach(k => data[k] = Number(data[k]));
  return data;
}

lookupBtn.addEventListener("click", loadEmployee);
employeeId.addEventListener("change", () => { if (employeeId.value.trim()) loadEmployee(); });

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  errorBox.textContent = "";
  predictBtn.disabled = true;
  predictBtn.textContent = "Evaluating…";
  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(formDataToObject(form))
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "Prediction failed.");

    const pct = data.probability * 100;
    employeeResultId.textContent = `Employee ID: ${data.employee_id}`;
    resultLabel.textContent = data.attrition === "Yes" ? "Likely to Leave" : "Likely to Stay";
    probBar.style.width = `${pct}%`;
    probText.textContent = `Predicted attrition: ${data.attrition} • Probability: ${pct.toFixed(1)}%`;

    if (data.employee_found && data.actual_attrition !== null) {
      const actual = data.actual_attrition === "Yes" ? "Attrition: Yes" : "Attrition: No";
      const match = data.prediction_correct ? "Prediction matches the dataset label." : "Prediction differs from the dataset label.";
      comparisonText.textContent = `Actual dataset result: ${actual} • ${match}`;
      comparisonText.className = data.prediction_correct ? "comparison-text match" : "comparison-text mismatch";
    } else {
      comparisonText.textContent = "New Employee ID: no actual attrition label is available for comparison.";
      comparisonText.className = "comparison-text";
    }
  } catch (err) {
    errorBox.textContent = err.message;
  } finally {
    predictBtn.disabled = false;
    predictBtn.textContent = "Evaluate Employee";
  }
});

document.getElementById("demoBtn").addEventListener("click", () => {
  employeeId.value = "E100000";
  loadEmployee();
});

async function loadMetrics(){
  const box = document.getElementById("metrics");
  try{
    const response = await fetch("/api/metrics");
    const data = await response.json();
    if(!response.ok) throw new Error(data.error);
    const m = data.models[data.best_model].metrics;
    const items = [["Accuracy",m.accuracy],["Precision",m.precision],["Recall",m.recall],["F1 Score",m.f1_score],["ROC-AUC",m.roc_auc]];
    box.innerHTML = items.map(([label,val]) => `<div class="metric"><span>${label}</span><b>${(val*100).toFixed(1)}%</b></div>`).join("");
  } catch(e){ box.innerHTML = `<div class="loading">${e.message}</div>`; }
}
loadMetrics();
