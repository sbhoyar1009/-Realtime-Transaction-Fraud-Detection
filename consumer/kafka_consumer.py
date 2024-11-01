from kafka import KafkaConsumer
import pandas as pd
import pickle
import json
import joblib

# Load your trained model
model = joblib.load('../fraud_model.pkl')
# Kafka Consumer setup
consumer = KafkaConsumer(
    'transactions',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='fraud_detection_group',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

def preprocess(data):
    df = pd.DataFrame([data])
    df = pd.get_dummies(df, columns=['type'], prefix='', prefix_sep='')

    # Calculate features
    df['balance_change'] = df['newbalanceOrig'] - df['oldbalanceOrg']
    df['high_amount'] = df['amount'] > 200000
    df['hour'] = df['step'] % 24

    # Fixed columns
    required_columns = [
        'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
        'oldbalanceDest', 'newbalanceDest', 'balance_change',
        'high_amount', 'hour', 'isFlaggedFraud',
        'CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'
    ]

    df = df.reindex(columns=required_columns, fill_value=0)
    return df

def run_consumer():
    print("Consumer is called")
    for message in consumer:
        transaction_data = message.value
        features = preprocess(transaction_data)
        prediction = model.predict(features)

        # Handle the prediction result (e.g., log it or trigger alerts)
        if prediction[0] == 1:
            print(f"Fraud detected for transaction: {transaction_data}")

if __name__ == '__main__':
    run_consumer()
