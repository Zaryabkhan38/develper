import os
import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load model and preprocessors
model   = joblib.load("best_model.joblib")
scaler  = joblib.load("scaler.joblib")
le_dict = joblib.load("label_encoders.joblib")

SCALED_MODELS = ["Logistic Regression","KNN","SVM","Naive Bayes","Neural Network"]

# Detect best model name from file (stored as attribute or fallback)
try:
    best_model_name = model.__class__.__name__
except:
    best_model_name = "Unknown"

@app.route("/")
def home():
    return {"status": "API is running"}, 200

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        age               = float(data["age"])
        gender            = data["gender"]
        department        = data["department"]
        monthly_income    = float(data["monthly_income"])
        total_experience  = float(data["total_experience"])
        tenure_at_co      = float(data["tenure_at_co"])
        overtime_eligible = data["overtime_eligible"]

        # Encode categoricals using saved LabelEncoders
        gender_enc    = le_dict["Gender"].transform([gender])[0]
        dept_enc      = le_dict["Department"].transform([department])[0]
        overtime_enc  = le_dict["Overtime_Eligible"].transform([overtime_eligible])[0]

        features = np.array([[age, gender_enc, dept_enc,
                               monthly_income, total_experience,
                               tenure_at_co, overtime_enc]])

        # Scale if needed
        if best_model_name in SCALED_MODELS:
            features = scaler.transform(features)

        prediction = model.predict(features)[0]
        probability = model.predict_proba(features)[0].tolist() if hasattr(model,"predict_proba") else [0.5, 0.5]

        result_label = le_dict["Attrition_Status"].inverse_transform([prediction])[0]

        return jsonify({
            "prediction": result_label,
            "probability_no":  round(probability[0]*100, 2),
            "probability_yes": round(probability[1]*100, 2),
            "status": "success"
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
