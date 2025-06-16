from flask import Flask, request, jsonify
from flask_cors import CORS  # 👈 Import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])  # 👈 Allow React frontend

# Load trained model and preprocessing tools
model = joblib.load('trained_data/model_reg.pkl')
scaler = joblib.load('trained_data/scaler.pkl')
label_encoders = joblib.load('trained_data/label_encoders.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Extract and encode inputs
        gender = label_encoders['Gender'].transform([data['gender']])[0]
        internet = label_encoders['Internet_Access_at_Home'].transform([data['internet']])[0]
        extracurricular = label_encoders['Extracurricular_Activities'].transform([data['extracurricular']])[0]

        study_hours = float(data['study_hours_per_week'])
        attendance = float(data['attendance_rate'])
        past_scores = float(data['past_exam_scores'])
        final_exam_score = float(data['final_exam_score'])

        # Create feature array in correct order
        features = np.array([[gender, study_hours, attendance, past_scores,
                              internet, extracurricular, final_exam_score]])

        # Scale and predict
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)[0]
        result = label_encoders['Pass_Fail'].inverse_transform([prediction])[0]

        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
