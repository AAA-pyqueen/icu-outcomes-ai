import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score
import json

df = pd.read_csv('data/processed/icu_data_cleaned.csv')
X = df.drop(columns=['outcome'])
y = df['outcome']

model = joblib.load('models/patient_outcome_model.pkl')
y_pred = model.predict(X)

metrics = {
    "accuracy": accuracy_score(y, y_pred),
    "precision": precision_score(y, y_pred),
    "recall": recall_score(y, y_pred)
}

with open('outputs/evaluation_metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

print("✓ Evaluation metrics saved.")
