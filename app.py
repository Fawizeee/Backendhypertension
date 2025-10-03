from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib
import os
import pandas as pd

app = Flask(__name__)
CORS(app)

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'hypertension_model.pkl')

# Load the pre-trained model
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        print(f"Model loaded successfully from {MODEL_PATH}")
        print(f"Model type: {type(model)}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Initialize the model
model = load_model()

if model is None:
    print("Warning: Could not load the model. Using fallback prediction.")
    model = "fallback"
else:
    print("Model ready for predictions!")

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "message": "Hypertension Prediction API is running"})

@app.route('/api/predict', methods=['POST'])
def predict_hypertension():
    try:
        data = request.get_json()
        
        # Required fields based on user's example
        required_fields = ['Age', 'Salt_Intake', 'Stress_Score', 'BP_History', 'Sleep_Duration', 'BMI', 'Medication', 'Family_History', 'Exercise_Level', 'Smoking_Status']
        
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Encoding mappings for categorical variables
        bp_history_map = {'Normal': 0, 'High': 1, 'Low': 2}
        medication_map = {'None': 0, 'Yes': 1}
        family_history_map = {'No': 0, 'Yes': 1}
        exercise_level_map = {'Low': 0, 'Moderate': 1, 'High': 2}
        smoking_status_map = {'Non-Smoker': 0, 'Smoker': 1}
        
        # Prepare encoded data
        encoded_data = {
            'Age': float(data['Age']),
            'Salt_Intake': float(data['Salt_Intake']),
            'Stress_Score': float(data['Stress_Score']),
            'BP_History': bp_history_map.get(data['BP_History'], 0),
            'Sleep_Duration': float(data['Sleep_Duration']),
            'BMI': float(data['BMI']),
            'Medication': medication_map.get(data['Medication'], 0),
            'Family_History': family_history_map.get(data['Family_History'], 0),
            'Exercise_Level': exercise_level_map.get(data['Exercise_Level'], 0),
            'Smoking_Status': smoking_status_map.get(data['Smoking_Status'], 0)
        }
        
        new_data = pd.DataFrame([encoded_data])
        
        if model != "fallback":
            proba = model.predict_proba(new_data)[:, 1][0]
            pred = int(proba >= 0.5)
        else:
            proba = 0.5
            pred = 0
        
        return jsonify({
            "prediction": pred,
            "probability": round(proba, 3),
            "message": "New Patient Prediction"
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/risk-factors', methods=['GET'])
def get_risk_factors():
    """Return information about risk factors for hypertension"""
    risk_factors = {
        "modifiable": [
            "High BMI (Body Mass Index)",
            "Smoking",
            "Lack of exercise",
            "High stress levels",
            "High cholesterol",
            "Poor diet"
        ],
        "non_modifiable": [
            "Age",
            "Family history of hypertension",
            "Gender (men are at higher risk until age 64)",
            "Race/ethnicity"
        ],
        "medical_conditions": [
            "Diabetes",
            "High blood pressure readings",
            "Kidney disease",
            "Sleep apnea"
        ]
    }
    return jsonify(risk_factors)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
