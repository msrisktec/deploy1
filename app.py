from flask import Flask, request, jsonify
import numpy as np
import joblib
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

try:
    model = joblib.load("risk_model.pkl")  # Load the ML model
except:
    model = None  # If model is not found, use random predictions

def compute_risk_score(inputs):
    if model:
        risk_score = model.predict([inputs])[0]
    else:
        risk_score = np.random.choice(["Low", "Medium", "High"])
    return risk_score

@app.route("/assess-risk", methods=["POST"])
def assess_risk():
    try:
        data = request.json
        inputs = [
            data["financial_stability"],
            data["market_volatility"],
            data["cybersecurity_risk"],
            data["operational_risk"]
        ]
        risk_level = compute_risk_score(inputs)

        recommendations = {
            "Low": "Your risk level is low. Maintain stability and monitor trends.",
            "Medium": "Your risk level is moderate. Consider diversifying your risk portfolio.",
            "High": "Your risk level is high! Take immediate action to mitigate risks."
        }

        return jsonify({"risk_level": risk_level, "recommendation": recommendations[risk_level]})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Get PORT from Render, default to 5000
    app.run(host="0.0.0.0", port=port, debug=True)
