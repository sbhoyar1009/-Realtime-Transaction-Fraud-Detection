from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the saved model
model = joblib.load('fraud_detection_model.pkl')

# Define a function to preprocess the incoming data
def preprocess(data):
    # Convert JSON data to DataFrame for easier manipulation
    df = pd.DataFrame([data])
    
    # Replicate the same preprocessing steps as during training
    df['hour'] = df['step'] % 24
    df['balance_change'] = df['newbalanceOrig'] - df['oldbalanceOrg']
    df['high_amount'] = df['amount'] > 200000  # Example threshold
    
    # One-hot encode the transaction type
    df = pd.get_dummies(df, columns=['type'], drop_first=True)
    
    # Ensure that the DataFrame has the same columns as the model’s training data
    for col in model.feature_names_in_:
        if col not in df.columns:
            df[col] = 0  # Add missing columns with a default value (e.g., 0)

    return df[model.feature_names_in_].values

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    # Preprocess the input data to match training features
    features = preprocess(data)
    
    # Predict fraud
    prediction = model.predict(features)
    return jsonify({'isFraud': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
