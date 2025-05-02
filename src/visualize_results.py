import matplotlib.pyplot as plt
import pandas as pd
import joblib

model = joblib.load('models/patient_outcome_model.pkl')
X = pd.read_csv('data/processed/icu_data_cleaned.csv').drop(columns=['outcome'])

# Feature importance
importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(10, 6))
plt.barh(features, importances)
plt.title("Feature Importance")
plt.savefig('outputs/feature_importance_plot.png')
print("✓ Feature importance plot saved.")
