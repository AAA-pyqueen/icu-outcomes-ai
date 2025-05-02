import yaml
import os
from datetime import datetime

def update_model_registry(model_entry, registry_path="models/models_config.yaml"):
    # Load existing registry as needed
    if os.path.exists(registry_path):
        with open(registry_path, "r") as file:
            registry = yaml.safe_load(file) or {}
    else:
        registry = {}

    if "models" not in registry:
        registry["models"] = []

    # Check if model name already exists
    for existing_model in registry["models"]:
        if existing_model["name"] == model_entry["name"]:
            existing_model.update(model_entry)
            break
    else:
        registry["models"].append(model_entry)

    # Save updated registry
    with open(registry_path, "w") as file:
        yaml.dump(registry, file, default_flow_style=False)

    print(f"✅ Model '{model_entry['name']}' updated in registry.")

# Example usage (after training a model)
if __name__ == "__main__":
    new_model = {
        "name": "RandomForest ICU Outcome Predictor",
        "version": "1.0",
        "config_file": "config/random_forest_config.yaml",
        "input_features": ["age", "sex", "heart_rate", "blood_pressure", "temperature", "respiratory_rate"],
        "target_variable": "survival",
        "model_file": "models/random_forest_model.pkl",
        "metrics": {
            "accuracy": 0.85,
            "precision": 0.80,
            "recall": 0.78
        },
        "last_trained": datetime.today().strftime("%Y-%m-%d")
    }

    update_model_registry(new_model)
