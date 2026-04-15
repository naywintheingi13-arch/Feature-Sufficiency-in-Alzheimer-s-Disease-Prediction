from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    bundle = pickle.load(f)

pipeline      = bundle['pipeline']
FEATURE_NAMES = bundle['feature_names']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json

    features = pd.DataFrame([{
        'FunctionalAssessment': data['functional_assessment'],
        'ADL':                  data['adl'],
        'MMSE':                 data['mmse'],
        'MemoryComplaints':     int(data['memory_complaints']),
        'BehavioralProblems':   int(data['behavioral_problems']),
    }])

    prob      = pipeline.predict_proba(features)[0][1]
    predicted = int(pipeline.predict(features)[0])

    return jsonify({
        'prediction':   predicted,
        'probability':  round(float(prob) * 100, 1),
        'diagnosis':    'Alzheimer\'s Detected' if predicted == 1 else 'No Alzheimer\'s Detected',
        'risk_level':   'High' if prob >= 0.7 else 'Moderate' if prob >= 0.4 else 'Low',
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)