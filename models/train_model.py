import argparse
import yaml
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from datetime import datetime
from utils.update_model_registry import update_model_registry

from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

# -------------- Load Config ----------------
def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

# -------------- Load and Prepare Data ----------------
data = pd.read_csv("data/icu_extended.csv")
X = data.drop(columns=["survival"])
y = data["survival"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------- Random Forest ----------------
def train_random_forest(X_train, X_test, y_train, y_test, config):
    hp = config['hyperparameters']
    model = RandomForestClassifier(
        n_estimators=hp['n_estimators'],
        max_depth=hp['max_depth'],
        min_samples_split=hp['min_samples_split'],
        min_samples_leaf=hp['min_samples_leaf'],
        random_state=hp['random_state']
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    evaluate_model(y_test, y_pred, config['evaluation_metrics'])
    return model

# -------------- Deep Learning Model ----------------
def train_deep_learning(X_train, X_test, y_train, y_test, config):
    hp = config['hyperparameters']
    model = Sequential()
    model.add(Dense(hp['hidden_layers'][0], activation=hp['activation'], input_shape=(X_train.shape[1],)))
    for units in hp['hidden_layers'][1:]:
        model.add(Dense(units, activation=hp['activation']))
        model.add(Dropout(hp['dropout_rate']))
    model.add(Dense(1, activation=hp['output_activation']))

    model.compile(
        loss=hp['loss_function'],
        optimizer=hp['optimizer'],
        metrics=config['evaluation_metrics']
    )

    model.fit(X_train, y_train, epochs=hp['epochs'], batch_size=hp['batch_size'], validation_data=(X_test, y_test))
    y_pred = (model.predict(X_test) > 0.5).astype(int)
    evaluate_model(y_test, y_pred, config['evaluation_metrics'])
    return model

# -------------- Evaluation ----------------
def evaluate_model(y_true, y_pred, metrics):
    print("\n--- Evaluation Metrics ---")
    for metric in metrics:
        if metric == 'accuracy':
            print("Accuracy:", accuracy_score(y_true, y_pred))
        elif metric == 'precision':
            print("Precision:", precision_score(y_true, y_pred))
        elif metric == 'recall':
            print("Recall:", recall_score(y_true, y_pred))
        elif metric == 'f1':
            print("F1 Score:", f1_score(y_true, y_pred))
        elif metric == 'roc_auc':
            print("ROC AUC Score:", roc_auc_score(y_true, y_pred))

# -------------- Main Entry Point ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Path to config YAML file')
    parser.add_argument('--data', required=True, help='Path to CSV data file')
    args = parser.parse_args()

    config = load_config(args.config)
    X_train, X_test, y_train, y_test = load_data(args.data, config['input_features'], config['output_label'])

    if config['model_type'] == 'random_forest':
        model = train_random_forest(X_train, X_test, y_train, y_test, config)
    elif config['model_type'] == 'deep_learning':
        model = train_deep_learning(X_train, X_test, y_train, y_test, config)
    else:
        raise ValueError("Unsupported model type specified in config.")
