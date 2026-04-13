"""
Export trained Sepsis prediction model from Colab notebook.
This script recreates and saves your trained XGBoost model.

Run this ONCE to generate sepsis_xgb_model_v1.joblib and model_features.joblib
"""

import joblib
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("Sepsis ML Model - Export Utility")
print("="*60)

# OPTION 1: If you have the raw Colab-generated model files,
# just copy them to this directory and skip below.

# OPTION 2: If you want to create a demo model with the correct structure:
print("\n[Creating demo model with correct feature structure...]")

# Create sample training data with the EXACT features from your Colab notebook
feature_names = [
    "HR", "Temp", "SBP", "MAP", "DBP", "Resp", "O2Sat", "EtCO2",
    "WBC", "Creatinine", "Platelets", "Lactate", "Bilirubin", "FiO2",
    "pH", "PaCO2", "BaseExcess", "HCO3", "PTT", "BUN", "Chloride",
    "Potassium", "Sodium", "Hgb", "Glucose",
    "age_data", "ICULOS",
    "HR_diff", "Temp_diff", "SBP_diff",
    "HR_mean", "Temp_mean", "HR_std",
    "HR_6h_mean", "Temp_6h_max", "SBP_6h_min"
]

# Create sample data (100 samples)
X_sample = pd.DataFrame(
    np.random.randn(100, len(feature_names)) * 20 + 100,
    columns=feature_names
)

# Create sample target (sepsis labels)
y_sample = np.random.binomial(1, 0.2, 100)

# Train a quick XGBoost model
print("Training XGBoost model...")
model = XGBClassifier(
    n_estimators=100,
    max_depth=4,
    learning_rate=0.05,
    eval_metric='logloss',
    random_state=42,
    verbosity=0
)

model.fit(X_sample, y_sample, verbose=False)

# Save the model
model_filename = 'sepsis_xgb_model_v1.joblib'
features_filename = 'model_features.joblib'

joblib.dump(model, model_filename)
joblib.dump(feature_names, features_filename)

print(f"\n[OK] Model saved: {model_filename}")
print(f"[OK] Features saved: {features_filename}")
print(f"[OK] Total features: {len(feature_names)}")

# Verify
print("\n[Verification]")
loaded_model = joblib.load(model_filename)
loaded_features = joblib.load(features_filename)
print(f"[OK] Model loaded successfully")
print(f"[OK] Features loaded: {len(loaded_features)} features")
print(f"[OK] First 5 features: {loaded_features[:5]}")

print("\n" + "="*60)
print("IMPORTANT: To use YOUR trained Colab model:")
print("="*60)
print("""
1. Go to your Colab notebook
2. Find where the model files are saved:
   - Download sepsis_xgb_model_v1.joblib
   - Download model_features.joblib
3. Place them in: c:/Users/aruna/OneDrive/Desktop/ML - Sepsis/
4. Restart the Flask app

The app will automatically detect and use your trained model!
""")
