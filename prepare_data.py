import pandas as pd
import os

# Load raw data
raw_path = 'data/raw/icu_data.csv'
df = pd.read_csv(raw_path)

# Preprocessing
df.dropna(inplace=True)
df['gender'] = df['gender'].map({'Male': 0, 'Female': 1})

# Save cleaned data
os.makedirs('data/processed', exist_ok=True)
df.to_csv('data/processed/icu_data_cleaned.csv', index=False)
print("✓ Data preprocessing complete.")
