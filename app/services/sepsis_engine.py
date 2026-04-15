import json
import os
from datetime import datetime
from pathlib import Path
import sys

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

# Import LSTM model
try:
    from lstm_model import SepsisLSTMModel
except ImportError:
    SepsisLSTMModel = None
    print("[WARNING] LSTM model not available. Install tensorflow: pip install tensorflow")

# Import human-in-the-loop manager
try:
    from app.services.human_loop_manager import get_human_loop_manager
except ImportError:
    get_human_loop_manager = None


class SepsisEngine:
    """
    Hybrid ML Engine for sepsis prediction
    Combines XGBoost (tree-based) + LSTM (deep learning) for robust predictions
    """

    def __init__(self):
        self.xgb_model = None            # XGBoost model
        self.lstm_model = None           # LSTM deep learning model
        self.features = None
        self.twilio_client = None
        
        # Weights for ensemble (XGBoost 60%, LSTM 40%)
        self.xgb_weight = 0.6
        self.lstm_weight = 0.4
        
        self.load_models()
        self.init_twilio()

    def load_models(self):
        """Load trained XGBoost and LSTM models"""
        base_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

        # ===== LOAD XGBOOST MODEL =====
        xgb_model_file = os.path.join(base_path, "sepsis_xgb_model_v1.joblib")
        features_file = os.path.join(base_path, "model_features.joblib")

        if joblib and os.path.exists(xgb_model_file):
            try:
                self.xgb_model = joblib.load(xgb_model_file)
                print(f"[OK] XGBoost model loaded from {xgb_model_file}")
            except Exception as e:
                print(f"[WARNING] Could not load XGBoost model: {e}")
                self.xgb_model = None

        if joblib and os.path.exists(features_file):
            try:
                self.features = joblib.load(features_file)
                print(f"[OK] Features loaded from {features_file}")
            except Exception as e:
                print(f"[WARNING] Could not load features: {e}")
                self.features = None

        # ===== LOAD LSTM MODEL =====
        lstm_model_file = os.path.join(base_path, "lstm_sepsis_model.h5")
        
        if SepsisLSTMModel:
            self.lstm_model = SepsisLSTMModel(timesteps=8, n_features=25)
            if self.lstm_model.load():
                print(f"[OK] LSTM model loaded from {lstm_model_file}")
            else:
                print("[WARNING] LSTM model not found. Will use XGBoost only.")
                self.lstm_model = None
        else:
            print("[WARNING] TensorFlow not installed. LSTM model unavailable.")

        # Fallback feature list (from training code)
        if not self.features:
            self.features = [
                "HR", "Temp", "SBP", "MAP", "DBP", "Resp", "O2Sat", "EtCO2",
                "WBC", "Creatinine", "Platelets", "Lactate", "Bilirubin", "FiO2",
                "pH", "PaCO2", "BaseExcess", "HCO3", "PTT", "BUN", "Chloride",
                "Potassium", "Sodium", "Hgb", "Glucose",
                "age_data", "ICULOS",
                "HR_diff", "Temp_diff", "SBP_diff",
                "HR_mean", "Temp_mean",
                "HR_std", "HR_6h_mean",
                "Temp_6h_max", "SBP_6h_min",
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
        """
        Run hybrid sepsis prediction using ensemble of XGBoost + LSTM
        
        Ensemble Strategy:
        - XGBoost (60%): Tree-based model, good for feature interactions
        - LSTM (40%): Deep learning model, captures temporal patterns
        - Final: Weighted average of both predictions
        """
        features_dict = self.extract_features(patient)
        patient_name = patient.get("name", "Unknown")
        patient_id = patient.get("id", "N/A")

        xgb_score = None
        lstm_score = None
        ensemble_score = None

        # ===== GET XGBOOST PREDICTION =====
        if self.xgb_model and self.features:
            try:
                import numpy as np

                # Create feature vector in correct order
                X = np.array([[features_dict.get(f, 0) for f in self.features]])
                xgb_score = float(self.xgb_model.predict_proba(X)[0, 1])
                print(f"[XGBOOST] Patient {patient_id} ({patient_name}): Risk={xgb_score*100:.1f}%")
            except Exception as e:
                print(f"[WARNING] XGBoost inference failed: {e}")
                xgb_score = None

        # ===== GET LSTM PREDICTION =====
        if self.lstm_model:
            try:
                lstm_score = self.lstm_model.predict(patient)
                print(f"[LSTM] Patient {patient_id} ({patient_name}): Risk={lstm_score*100:.1f}%")
            except Exception as e:
                print(f"[WARNING] LSTM inference failed: {e}")
                lstm_score = None

        # ===== ENSEMBLE DECISION =====
        if xgb_score is not None and lstm_score is not None:
            # Both models available - use weighted ensemble
            ensemble_score = (self.xgb_weight * xgb_score) + (self.lstm_weight * lstm_score)
            model_info = f"Ensemble (XGB: {xgb_score*100:.1f}% + LSTM: {lstm_score*100:.1f}%)"
            print(f"[ENSEMBLE] Final score: {ensemble_score*100:.1f}% (avg of both models)")

        elif xgb_score is not None:
            # Only XGBoost available
            ensemble_score = xgb_score
            model_info = "XGBoost only"
            print(f"[SINGLE MODEL] Using XGBoost prediction (LSTM unavailable)")

        elif lstm_score is not None:
            # Only LSTM available
            ensemble_score = lstm_score
            model_info = "LSTM only"
            print(f"[SINGLE MODEL] Using LSTM prediction (XGBoost unavailable)")

        else:
            # No models available - use fallback
            print(f"[FALLBACK] No ML models available, using heuristic")
            return self._fallback_predict(features_dict)

        # ===== GET FEATURE IMPORTANCE =====
        top_features = []
        if self.xgb_model and self.features:
            try:
                importances = self.xgb_model.feature_importances_
                feature_importance = dict(zip(self.features, importances))
                top_features_sorted = sorted(
                    feature_importance.items(), key=lambda x: x[1], reverse=True
                )[:7]
                
                top_features = [
                    {
                        "name": f[0],
                        "val": float(features_dict.get(f[0], 0)),
                        "contrib": round(float(f[1]), 2),
                        "dir": self._direction(f[0], features_dict.get(f[0], 0)),
                    }
                    for f in top_features_sorted
                ]
            except Exception as e:
                print(f"[WARNING] Could not extract feature importance: {e}")

        response = {
            "risk_score": round(ensemble_score, 3),
            "risk_level": self._risk_level(ensemble_score),
            "top_features": top_features,
            "xgb_score": round(xgb_score, 3) if xgb_score else None,
            "lstm_score": round(lstm_score, 3) if lstm_score else None,
            "ensemble_score": round(ensemble_score, 3),
            "model_type": model_info,
            "message": f"Sepsis risk: {ensemble_score*100:.1f}% ({model_info})",
        }
        
        # ===== ADD TO HUMAN-IN-THE-LOOP REVIEW QUEUE =====
        try:
            if get_human_loop_manager:
                hlm = get_human_loop_manager()
                prediction_id = hlm.add_prediction(
                    patient_id=patient_id,
                    features=features_dict,
                    lstm_score=lstm_score,
                    xgb_score=xgb_score,
                    ensemble_score=ensemble_score,
                    model_type=model_info,
                    risk_level=response["risk_level"]
                )
                response["prediction_id"] = prediction_id
                if response["risk_level"] in ["HIGH", "CRITICAL"]:
                    response["review_required"] = True
        except Exception as e:
            print(f"[WARNING] Could not add to human-loop queue: {e}")
        
        return response

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
            "xgb_score": None,
            "lstm_score": None,
            "ensemble_score": min(score, 0.99),
            "model_type": "Rule-based fallback",
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
