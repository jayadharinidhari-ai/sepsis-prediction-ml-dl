import json
import os
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent.parent / '.env')
except ImportError:
    pass

try:
    import joblib
except ImportError:
    joblib = None

try:
    from twilio.rest import Client
except ImportError:
    Client = None


class SepsisEngine:
    """Machine Learning engine for sepsis prediction"""

    def __init__(self):
        self.model = None
        self.features = None
        self.model_path = None
        self.features_path = None
        self.twilio_client = None
        self.load_model()
        self.init_twilio()

    def load_model(self):
        """Load trained XGBoost model and feature list"""
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

        # Try to load .joblib files
        model_file = os.path.join(base_path, "sepsis_xgb_model_v1.joblib")
        features_file = os.path.join(base_path, "model_features.joblib")

        if joblib and os.path.exists(model_file):
            try:
                self.model = joblib.load(model_file)
                print(f"[OK] Model loaded from {model_file}")
            except Exception as e:
                print(f"[WARNING] Could not load model: {e}")
                self.model = None

        if joblib and os.path.exists(features_file):
            try:
                self.features = joblib.load(features_file)
                print(f"[OK] Features loaded from {features_file}")
            except Exception as e:
                print(f"[WARNING] Could not load features: {e}")
                self.features = None

        # Fallback feature list (from training code)
        if not self.features:
            self.features = [
                "HR",
                "Temp",
                "SBP",
                "MAP",
                "DBP",
                "Resp",
                "O2Sat",
                "EtCO2",
                "WBC",
                "Creatinine",
                "Platelets",
                "Lactate",
                "Bilirubin",
                "FiO2",
                "pH",
                "PaCO2",
                "BaseExcess",
                "HCO3",
                "PTT",
                "BUN",
                "Chloride",
                "Potassium",
                "Sodium",
                "Hgb",
                "Glucose",
                "age_data",
                "ICULOS",
                "HR_diff",
                "Temp_diff",
                "SBP_diff",
                "HR_mean",
                "Temp_mean",
                "HR_std",
                "HR_6h_mean",
                "Temp_6h_max",
                "SBP_6h_min",
            ]

    def init_twilio(self):
        """Initialize Twilio SMS client"""
        # Using environment variables or hardcoded for demo
        # For production, set TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_PHONE_NUMBER
        account_sid = os.getenv("TWILIO_ACCOUNT_SID")
        auth_token = os.getenv("TWILIO_AUTH_TOKEN")

        if Client and account_sid and auth_token:
            try:
                self.twilio_client = Client(account_sid, auth_token)
                print("[OK] Twilio SMS client initialized")
            except Exception as e:
                print(f"[WARNING] Twilio not configured: {e}")
                self.twilio_client = None

    def extract_features(self, patient):
        """Extract ML features from patient data"""
        features_dict = {}

        # Vitals
        vitals = patient.get("vitals", {})
        features_dict["HR"] = vitals.get("HR", 76)
        features_dict["Temp"] = vitals.get("Temp", 37.0)
        features_dict["SBP"] = vitals.get("SBP", 120)
        features_dict["MAP"] = vitals.get("MAP", 82)
        features_dict["DBP"] = vitals.get("DBP", 72)
        features_dict["Resp"] = vitals.get("Resp", 16)
        features_dict["O2Sat"] = vitals.get("O2Sat", 98)
        features_dict["EtCO2"] = vitals.get("EtCO2", 40)

        # Labs
        labs = patient.get("labs", {})
        features_dict["WBC"] = labs.get("WBC", 8.0)
        features_dict["Creatinine"] = labs.get("Creatinine", 0.9)
        features_dict["Platelets"] = labs.get("Platelets", 220)
        features_dict["Lactate"] = labs.get("Lactate", 0.8)
        features_dict["Bilirubin"] = labs.get("Bilirubin", 0.7)
        features_dict["FiO2"] = labs.get("FiO2", 0.21)
        features_dict["pH"] = labs.get("pH", 7.40)
        features_dict["PaCO2"] = labs.get("PaCO2", 40)
        features_dict["BaseExcess"] = labs.get("BaseExcess", 0)
        features_dict["HCO3"] = labs.get("HCO3", 24)
        features_dict["PTT"] = labs.get("PTT", 28)
        features_dict["BUN"] = labs.get("BUN", 12)
        features_dict["Chloride"] = labs.get("Chloride", 104)
        features_dict["Potassium"] = labs.get("Potassium", 4.0)
        features_dict["Sodium"] = labs.get("Sodium", 140)
        features_dict["Hgb"] = labs.get("Hgb", 13.5)
        features_dict["Glucose"] = labs.get("Glucose", 95)

        # Demographics
        features_dict["age_data"] = patient.get("age", 50)
        features_dict["ICULOS"] = patient.get("ICULOS", 0)

        # Trend features (simplified from training code)
        trend = patient.get("trend", {})
        hr_data = trend.get("HR", [76] * 8)
        temp_data = trend.get("Temp", [37.0] * 8)
        sbp_data = trend.get("SBP", [120] * 8)

        features_dict["HR_diff"] = hr_data[-1] - hr_data[0] if len(hr_data) > 1 else 0
        features_dict["Temp_diff"] = temp_data[-1] - temp_data[0] if len(temp_data) > 1 else 0
        features_dict["SBP_diff"] = sbp_data[-1] - sbp_data[0] if len(sbp_data) > 1 else 0

        features_dict["HR_mean"] = sum(hr_data) / len(hr_data) if hr_data else 76
        features_dict["Temp_mean"] = sum(temp_data) / len(temp_data) if temp_data else 37.0

        features_dict["HR_std"] = self._std(hr_data)
        features_dict["HR_6h_mean"] = features_dict["HR_mean"]
        features_dict["Temp_6h_max"] = max(temp_data) if temp_data else 37.0
        features_dict["SBP_6h_min"] = min(sbp_data) if sbp_data else 120

        return features_dict

    def _std(self, data):
        """Calculate standard deviation"""
        if len(data) < 2:
            return 0.0
        mean = sum(data) / len(data)
        variance = sum((x - mean) ** 2 for x in data) / len(data)
        return variance ** 0.5

    def predict(self, patient):
        """Run sepsis prediction"""
        features_dict = self.extract_features(patient)

        # If model loaded, use it; otherwise use fallback heuristic
        if self.model and self.features:
            try:
                import pandas as pd
                import numpy as np

                # Create feature vector in correct order
                X = np.array([[features_dict.get(f, 0) for f in self.features]])
                risk_score = float(self.model.predict_proba(X)[0, 1])

                # Get feature importance from model
                importances = self.model.feature_importances_
                feature_importance = dict(zip(self.features, importances))

                top_features = sorted(
                    feature_importance.items(), key=lambda x: x[1], reverse=True
                )[:7]

                # Log model usage
                patient_name = patient.get("name", "Unknown")
                patient_id = patient.get("id", "N/A")
                print(f"[ML MODEL] Patient {patient_id} ({patient_name}): Risk={risk_score*100:.1f}%")

                return {
                    "risk_score": round(risk_score, 3),
                    "risk_level": self._risk_level(risk_score),
                    "top_features": [
                        {
                            "name": f[0],
                            "val": float(features_dict.get(f[0], 0)),
                            "contrib": round(float(f[1]), 2),
                            "dir": self._direction(f[0], features_dict.get(f[0], 0)),
                        }
                        for f in top_features
                    ],
                    "message": f"Sepsis risk: {risk_score*100:.1f}% (ML Model)",
                }
            except Exception as e:
                print(f"[WARNING] Model inference failed: {e}")

        # Fallback heuristic (if model not loaded)
        print(f"[FALLBACK] Using heuristic prediction (model not available)")
        return self._fallback_predict(features_dict)

    def _fallback_predict(self, features):
        """Fallback rule-based prediction"""
        score = 0.0

        # Clinical indicators
        if features.get("Temp", 37) > 38.5 or features.get("Temp") < 36:
            score += 0.15
        if features.get("HR", 76) > 100:
            score += 0.15
        if features.get("Resp", 16) > 20:
            score += 0.12
        if features.get("MAP", 82) < 70:
            score += 0.20
        if features.get("WBC", 8) > 12:
            score += 0.15
        if features.get("Lactate", 0.8) > 2:
            score += 0.20
        if features.get("Platelets", 220) < 150:
            score += 0.12
        if features.get("Creatinine", 0.9) > 1.2:
            score += 0.10

        return {
            "risk_score": min(score, 0.99),
            "risk_level": self._risk_level(score),
            "top_features": [
                {"name": "Lactate", "val": features.get("Lactate"), "contrib": 0.2, "dir": "high"},
                {"name": "MAP", "val": features.get("MAP"), "contrib": 0.2, "dir": "low"},
                {"name": "Temperature", "val": features.get("Temp"), "contrib": 0.15, "dir": "high"},
            ],
            "message": f"Sepsis risk: {score*100:.1f}% (fallback heuristic)",
        }

    def _risk_level(self, score):
        """Get risk level from score"""
        if score >= 0.75:
            return "High"
        elif score >= 0.4:
            return "Moderate"
        return "Low"

    def _direction(self, feature, value):
        """Determine if feature is high/low risk"""
        danger_high = {"WBC", "Lactate", "Creatinine", "Bilirubin", "Glucose", "HR", "Resp", "Temperature"}
        danger_low = {"Platelets", "MAP", "DBP", "pH", "HCO3", "O2Sat"}

        if feature in danger_high:
            return "high"
        elif feature in danger_low:
            return "low"
        return "mod"

    def send_alert(self, patient):
        """Send SMS alert to doctor via Twilio"""
        message = f"SEPSIS ALERT: {patient['name']} (ICU {patient['ward']}) - Sepsis risk {patient.get('sepsisRisk', 0)*100:.0f}%. Please review immediately."

        # Use doctor phone from patient OR environment variable
        doctor_phone = patient.get("doctorPhone", os.getenv("DOCTOR_PHONE", "+91-7339300849"))

        # Try Twilio SMS first
        if self.twilio_client and doctor_phone:
            try:
                from_phone = os.getenv("TWILIO_PHONE_NUMBER")
                if from_phone:
                    self.twilio_client.messages.create(
                        body=message, from_=from_phone, to=doctor_phone
                    )
                    return {
                        "success": True,
                        "method": "SMS",
                        "message": message,
                        "to": doctor_phone,
                        "status": "sent"
                    }
            except Exception as e:
                print(f"[WARNING] Twilio SMS failed: {e}")

        # Fallback: log the alert
        return {
            "success": True,
            "method": "LOG",
            "message": message,
            "to": doctor_phone,
            "note": "SMS logged (Twilio not configured)",
            "status": "logged"
        }
