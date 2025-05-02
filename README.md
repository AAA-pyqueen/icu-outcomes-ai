#  Predicting Patient Outcomes in ICUs Using AI
This project uses machine learning models to predict important patient outcomes in intensive care units (ICUs), such as:

- Survival (binary classification)
- Length of stay (regression)
- Risk of complications (multi-class classification)
  
The goal is to develop an informative and research-ready AI tool for exploring ICU data and risk prediction, with potential clinical applications.

**This project is in progress, please stop by for updates! Thank you for supporting my research!**

## 📂 Folder Structure
```bash
icu-outcomes-ai/
├── data/
│   ├── raw/
│   ├── processed/
│   └── metadata.csv
├── notebooks/
│   ├── eda_visualizations.ipynb
│   └── model_prototyping.ipynb
├── models/
│   ├── patient_outcome_model.pkl
│   ├── model_config.yaml
│   └── model_notes.md
├── outputs/
│   ├── evaluation_metrics.json
│   ├── prediction_examples.csv
│   └── feature_importance_plot.png
├── src/
│   ├── train_model.py
│   ├── evaluate_model.py
│   └── visualize_results.py
├── requirements.txt
├── prepare_data.py
├── README.md
└── .gitignore
```
## 🧪 Sample Data
You can use the provided **icu_data.csv** or **icu_extended_data.csv** file in data/raw/ to test the project. The data has been synthesized for educational use.
- outcome: 1 = Survived, 0 = Did not survive
- length_of_stay: In days
- comorbidity_score: 0–5 scale based on existing conditions
