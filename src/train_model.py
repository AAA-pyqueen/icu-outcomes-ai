import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv('data/processed/icu_data_cleaned.csv')
X = df.drop(columns=['outcome'])
y = df['outcome']

model = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42)
model.fit(X, y)

joblib.dump(model, 'models/patient_outcome_model.pkl')
print("✓ Model trained and saved.")
