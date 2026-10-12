from pathlib import Path
import json
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from src.predict import predict_attrition
from src.utils import METRICS_PATH

ROOT = Path(__file__).resolve().parent
app = Flask(__name__, static_folder="frontend", static_url_path="")
CORS(app)

@app.get("/")
def home():
    return send_from_directory(ROOT / "frontend", "index.html")

@app.get("/api/health")
def health():
    return jsonify({"status": "ok"})

@app.get("/api/metrics")
def metrics():
    if not METRICS_PATH.exists():
        return jsonify({"error": "Metrics not found. Train the model first."}), 404
    return jsonify(json.loads(METRICS_PATH.read_text()))

@app.get("/api/employee/<employee_id>")
def employee(employee_id):
    from src.predict import get_employee_by_id
    try:
        result = get_employee_by_id(employee_id)
        if result is None:
            return jsonify({"error": "Employee ID not found in the dataset."}), 404
        return jsonify(result)
    except (ValueError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 400

@app.post("/api/predict")
def predict():
    try:
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "Request body must be valid JSON."}), 400
        result = predict_attrition(payload)
        return jsonify(result)
    except (ValueError, TypeError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        return jsonify({"error": "Unexpected server error. Check the server logs."}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
