from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model and preprocessing tools
model = joblib.load('trained_data/model_reg.pkl')
scaler = joblib.load('trained_data/scaler.pkl')
label_encoders = joblib.load('trained_data/label_encoders.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json

        # Extract values
        gender = data['gender']
        study_hours = float(data['study_hours_per_week'])
        attendance = float(data['attendance_rate'])
        past_scores = float(data['past_exam_scores'])
        internet = data['internet']
        extracurricular = data['extracurricular']
        final_exam_score = float(data['final_exam_score'])

        # Encode categorical inputs
        gender_encoded = label_encoders['Gender'].transform([gender])[0]
        internet_encoded = label_encoders['Internet_Access_at_Home'].transform([internet])[0]
        extra_encoded = label_encoders['Extracurricular_Activities'].transform([extracurricular])[0]

        # Combine all features in correct order
        features = np.array([[gender_encoded, study_hours, attendance, past_scores,
                              internet_encoded, extra_encoded, final_exam_score]])

        # Scale features
        features_scaled = scaler.transform(features)

        # Predict
        prediction = model.predict(features_scaled)[0]
        result = label_encoders['Pass_Fail'].inverse_transform([prediction])[0]

        return jsonify({'prediction': result})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
