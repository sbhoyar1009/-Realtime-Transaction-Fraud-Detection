# app.py

from flask import Flask, request, jsonify
from kafka import KafkaProducer, KafkaConsumer
import json
import threading
import pandas as pd
import joblib
# Initialize the Flask application
app = Flask(__name__)

# Load the trained model at startup
model = joblib.load('../fraud_model.pkl')
producer = KafkaProducer(bootstrap_servers='localhost:9092',
                         value_serializer=lambda v: json.dumps(v).encode('utf-8'))

# Function to prepare features for prediction
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

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    # Get the transaction data from the request
    transaction = request.json
    
    # Send the transaction data to Kafka
    producer.send('transactions', transaction)
    producer.flush()  # Ensure the message is sent

    return jsonify({'status': 'Transaction sent to Kafka for processing'})

# Function to consume messages from Kafka and process them
def consume_transactions():
    consumer = KafkaConsumer('transactions',
                             bootstrap_servers='localhost:9092',
                             value_deserializer=lambda x: json.loads(x.decode('utf-8')))
    
    for message in consumer:
        transaction = message.value
        print(f"Received transaction: {transaction}")

        # Prepare features for prediction
        features = preprocess(transaction)
        
        # Make prediction
        prediction = model.predict(features)
        
        # Check if the prediction indicates fraud
        if int(prediction[0]) == 1:
            print(f"Fraud detected for transaction: {transaction}")
            # Here you could send an alert or take some action
        else:
            print(f"Fraud not detected for transaction: {transaction}")

# Run the consumer in a separate thread
threading.Thread(target=consume_transactions, daemon=True).start()

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
