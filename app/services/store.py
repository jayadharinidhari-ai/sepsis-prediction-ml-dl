import json
import os
from datetime import datetime
from collections import defaultdict


class PatientStore:
    """In-memory patient database"""

    def __init__(self):
        self.patients = {}
        self.alerts_log = []
        self.sms_sent = set()
        self.hitl_feedback = []  # HITL feedback from clinicians
        self.users = {}  # User management
        self._load_demo_patients()
        self._load_demo_users()

    def _load_demo_patients(self):
        """Load demo patients from JSON"""
        demo = [
            {
                "id": 1,
                "name": "Rajesh Kumar",
                "age": 67,
                "gender": "M",
                "ward": "ICU-A2",
                "doctor": "Dr. Priya Nair",
                "doctorPhone": "+91-7339300849",
                "admitted": "2026-04-07",
                "status": "admitted",
                "sepsisRisk": 0.87,
                "riskLevel": "High",
                "ICULOS": 18,
                "vitals": {
                    "HR": 118,
                    "Temp": 38.9,
                    "SBP": 88,
                    "MAP": 58,
                    "DBP": 52,
                    "Resp": 26,
                    "O2Sat": 91,
                    "EtCO2": 32,
                },
                "labs": {
                    "WBC": 18.4,
                    "Creatinine": 2.1,
                    "Platelets": 88,
                    "Lactate": 3.2,
                    "Bilirubin": 2.8,
                    "FiO2": 0.4,
                    "pH": 7.28,
                    "PaCO2": 32,
                    "BaseExcess": -6,
                    "HCO3": 18,
                    "PTT": 48,
                    "BUN": 38,
                    "Chloride": 98,
                    "Potassium": 3.2,
                    "Sodium": 136,
                    "Hgb": 9.2,
                    "Glucose": 162,
                },
                "topFeatures": [
                    {"name": "Lactate", "val": 3.2, "max": 5, "contrib": 0.92, "dir": "high"},
                    {"name": "MAP", "val": 58, "max": 100, "contrib": 0.87, "dir": "low"},
                    {"name": "WBC", "val": 18.4, "max": 25, "contrib": 0.81, "dir": "high"},
                    {
                        "name": "Temperature",
                        "val": 38.9,
                        "max": 41,
                        "contrib": 0.74,
                        "dir": "high",
                    },
                    {
                        "name": "Creatinine",
                        "val": 2.1,
                        "max": 5,
                        "contrib": 0.68,
                        "dir": "high",
                    },
                    {"name": "Resp Rate", "val": 26, "max": 40, "contrib": 0.61, "dir": "high"},
                    {"name": "Platelets", "val": 88, "max": 400, "contrib": 0.55, "dir": "low"},
                ],
                "trend": {
                    "HR": [98, 102, 107, 110, 112, 115, 116, 118],
                    "Temp": [37.2, 37.6, 37.9, 38.2, 38.5, 38.7, 38.8, 38.9],
                    "MAP": [78, 74, 70, 68, 65, 62, 60, 58],
                    "WBC": [11.2, 12.4, 13.8, 15.1, 16.2, 17.1, 17.8, 18.4],
                },
                "alerts": [
                    {
                        "time": "06:42",
                        "msg": "Sepsis probability exceeded 85% threshold. Immediate review required.",
                    },
                    {"time": "05:15", "msg": "MAP dropped below 65 mmHg — possible septic shock."},
                ],
            },
            {
                "id": 2,
                "name": "Meena Subramaniam",
                "age": 54,
                "gender": "F",
                "ward": "ICU-B1",
                "doctor": "Dr. Arvind Rao",
                "doctorPhone": "+91-99887-76655",
                "admitted": "2026-04-08",
                "status": "admitted",
                "sepsisRisk": 0.41,
                "riskLevel": "Moderate",
                "ICULOS": 6,
                "vitals": {
                    "HR": 96,
                    "Temp": 37.8,
                    "SBP": 112,
                    "MAP": 76,
                    "DBP": 68,
                    "Resp": 19,
                    "O2Sat": 96,
                    "EtCO2": 38,
                },
                "labs": {
                    "WBC": 12.1,
                    "Creatinine": 1.1,
                    "Platelets": 185,
                    "Lactate": 1.6,
                    "Bilirubin": 1.2,
                    "FiO2": 0.3,
                    "pH": 7.36,
                    "PaCO2": 38,
                    "BaseExcess": -1,
                    "HCO3": 22,
                    "PTT": 32,
                    "BUN": 22,
                    "Chloride": 102,
                    "Potassium": 3.9,
                    "Sodium": 139,
                    "Hgb": 11.4,
                    "Glucose": 128,
                },
                "topFeatures": [
                    {"name": "WBC", "val": 12.1, "max": 25, "contrib": 0.44, "dir": "high"},
                    {
                        "name": "Temperature",
                        "val": 37.8,
                        "max": 41,
                        "contrib": 0.38,
                        "dir": "high",
                    },
                    {"name": "HR", "val": 96, "max": 150, "contrib": 0.32, "dir": "high"},
                    {"name": "Lactate", "val": 1.6, "max": 5, "contrib": 0.28, "dir": "mod"},
                ],
                "trend": {
                    "HR": [88, 90, 91, 93, 94, 95, 95, 96],
                    "Temp": [37.1, 37.2, 37.4, 37.5, 37.6, 37.7, 37.7, 37.8],
                    "MAP": [82, 80, 79, 78, 78, 77, 76, 76],
                    "WBC": [9.8, 10.2, 10.8, 11.1, 11.4, 11.8, 12.0, 12.1],
                },
                "alerts": [],
            },
            {
                "id": 3,
                "name": "Anbu Selvam",
                "age": 72,
                "gender": "M",
                "ward": "ICU-A1",
                "doctor": "Dr. Priya Nair",
                "doctorPhone": "+91-98765-43210",
                "admitted": "2026-04-06",
                "status": "admitted",
                "sepsisRisk": 0.12,
                "riskLevel": "Low",
                "ICULOS": 30,
                "vitals": {
                    "HR": 78,
                    "Temp": 37.0,
                    "SBP": 132,
                    "MAP": 90,
                    "DBP": 78,
                    "Resp": 16,
                    "O2Sat": 98,
                    "EtCO2": 40,
                },
                "labs": {
                    "WBC": 8.4,
                    "Creatinine": 0.9,
                    "Platelets": 242,
                    "Lactate": 0.9,
                    "Bilirubin": 0.8,
                    "FiO2": 0.21,
                    "pH": 7.42,
                    "PaCO2": 40,
                    "BaseExcess": 1,
                    "HCO3": 25,
                    "PTT": 28,
                    "BUN": 14,
                    "Chloride": 105,
                    "Potassium": 4.1,
                    "Sodium": 141,
                    "Hgb": 13.2,
                    "Glucose": 102,
                },
                "topFeatures": [
                    {"name": "Age", "val": 72, "max": 100, "contrib": 0.18, "dir": "high"},
                    {"name": "ICULOS", "val": 30, "max": 72, "contrib": 0.14, "dir": "high"},
                ],
                "trend": {
                    "HR": [80, 79, 78, 79, 78, 78, 78, 78],
                    "Temp": [37.0, 37.0, 37.1, 37.0, 37.0, 37.0, 37.0, 37.0],
                    "MAP": [92, 91, 90, 91, 90, 90, 91, 90],
                    "WBC": [8.2, 8.3, 8.4, 8.3, 8.4, 8.4, 8.4, 8.4],
                },
                "alerts": [],
            },
        ]

        for p in demo:
            self.patients[p["id"]] = p

    def _load_demo_users(self):
        """Load demo users"""
        self.users = {
            'DR_SMITH': {
                'id': 'DR_SMITH',
                'name': 'Dr. John Smith',
                'role': 'clinician',
                'department': 'ICU',
                'created': datetime.now().isoformat()
            },
            'NURSE_JOHN': {
                'id': 'NURSE_JOHN',
                'name': 'John Nurse',
                'role': 'nurse',
                'ward': 'ICU-A2',
                'created': datetime.now().isoformat()
            },
            'PAT_001': {
                'id': 'PAT_001',
                'name': 'Patient Name',
                'role': 'patient',
                'dob': '1950-01-01',
                'created': datetime.now().isoformat()
            }
        }

    def get_admitted_patients(self):
        """Get all admitted patients"""
        return [p for p in self.patients.values() if p["status"] == "admitted"]

    def get_patient(self, patient_id):
        """Get patient by ID"""
        return self.patients.get(patient_id)

    def admit_patient(self, data):
        """Admit new patient"""
        patient_id = int(datetime.now().timestamp() * 1000)
        patient = {
            "id": patient_id,
            "name": data.get("name", "New Patient"),
            "age": int(data.get("age", 50)),
            "gender": data.get("gender", "M"),
            "ward": data.get("ward", "ICU"),
            "doctor": data.get("doctor", "Dr. Unknown"),
            "doctorPhone": data.get("doctorPhone", "+91-00000-00000"),
            "admitted": datetime.now().strftime("%Y-%m-%d"),
            "status": "admitted",
            "sepsisRisk": 0.08,
            "riskLevel": "Low",
            "ICULOS": 0,
            "vitals": {
                "HR": 76,
                "Temp": 37.0,
                "SBP": 120,
                "MAP": 82,
                "DBP": 72,
                "Resp": 16,
                "O2Sat": 98,
                "EtCO2": 40,
            },
            "labs": {
                "WBC": 8.0,
                "Creatinine": 0.9,
                "Platelets": 220,
                "Lactate": 0.8,
                "Bilirubin": 0.7,
                "FiO2": 0.21,
                "pH": 7.40,
                "PaCO2": 40,
                "BaseExcess": 0,
                "HCO3": 24,
                "PTT": 28,
                "BUN": 12,
                "Chloride": 104,
                "Potassium": 4.0,
                "Sodium": 140,
                "Hgb": 13.5,
                "Glucose": 95,
            },
            "topFeatures": [{"name": "Age", "val": data.get("age", 50), "max": 100, "contrib": 0.1, "dir": "high"}],
            "trend": {
                "HR": [76, 76, 76, 76, 76, 76, 76, 76],
                "Temp": [37.0, 37.0, 37.0, 37.0, 37.0, 37.0, 37.0, 37.0],
                "MAP": [82, 82, 82, 82, 82, 82, 82, 82],
                "WBC": [8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0],
            },
            "alerts": [],
        }
        self.patients[patient_id] = patient
        return patient

    def discharge_patient(self, patient_id):
        """Discharge a patient"""
        if patient_id in self.patients:
            self.patients[patient_id]["status"] = "discharged"
            return True
        return False

    def update_patient_risk(self, patient_id, risk_score, top_features):
        """Update patient risk score and features"""
        if patient_id in self.patients:
            self.patients[patient_id]["sepsisRisk"] = risk_score
            self.patients[patient_id]["topFeatures"] = top_features
            if risk_score >= 0.75:
                self.patients[patient_id]["riskLevel"] = "High"
            elif risk_score >= 0.4:
                self.patients[patient_id]["riskLevel"] = "Moderate"
            else:
                self.patients[patient_id]["riskLevel"] = "Low"

    def log_alert(self, patient_id, message):
        """Log an alert"""
        self.alerts_log.append({
            "patient_id": patient_id,
            "message": message,
            "timestamp": datetime.now().isoformat(),
        })

    def get_alerts(self):
        """Get all alerts"""
        return self.alerts_log

    # ============ HITL FEEDBACK MANAGEMENT ============
    def submit_hitl_feedback(self, clinician_id, patient_id, feedback_data):
        """Submit HITL feedback for patient training"""
        feedback = {
            'id': len(self.hitl_feedback) + 1,
            'clinician_id': clinician_id,
            'patient_id': patient_id,
            'timestamp': feedback_data.get('timestamp', datetime.now().isoformat()),
            'accuracy': feedback_data.get('accuracy'),  # New format from dashboard
            'clinical_notes': feedback_data.get('clinical_notes'),  # New format from dashboard
            'recommended_action': feedback_data.get('recommended_action'),  # New format from dashboard
            'vitals_assessment': feedback_data.get('vitals_assessment'),  # Legacy format
            'labs_assessment': feedback_data.get('labs_assessment'),  # Legacy format
            'clinical_impression': feedback_data.get('clinical_impression'),  # Legacy format
            'risk_assessment': feedback_data.get('risk_assessment'),  # Legacy format
            'notes': feedback_data.get('notes', ''),  # Legacy format
            'status': 'pending'
        }
        self.hitl_feedback.append(feedback)
        return feedback

    def get_hitl_feedback_list(self, clinician_id=None):
        """Get HITL feedback list"""
        if clinician_id:
            return [f for f in self.hitl_feedback if f['clinician_id'] == clinician_id]
        return self.hitl_feedback

    def get_hitl_feedback_count(self, clinician_id=None):
        """Get count of HITL feedback"""
        if clinician_id:
            return len([f for f in self.hitl_feedback if f['clinician_id'] == clinician_id])
        return len(self.hitl_feedback)

    def can_retrain(self):
        """Check if we have 10 or more HITL feedback entries"""
        return len(self.hitl_feedback) >= 10

    def get_patient_summary(self, patient_id):
        """Get patient summary for download"""
        patient = self.get_patient(patient_id)
        if not patient:
            return None
        
        feedbacks = [f for f in self.hitl_feedback if f['patient_id'] == patient_id]
        
        return {
            'patient_info': {
                'id': patient['id'],
                'name': patient['name'],
                'age': patient['age'],
                'gender': patient['gender'],
                'ward': patient['ward'],
                'admitted': patient['admitted'],
            },
            'current_risk': {
                'risk_score': patient.get('sepsisRisk', 0),
                'risk_level': patient.get('riskLevel', 'Unknown'),
                'updating_time': datetime.now().isoformat()
            },
            'current_vitals': patient.get('vitals', {}),
            'current_labs': patient.get('labs', {}),
            'clinical_feedback': feedbacks,
            'feedback_count': len(feedbacks)
        }

    def register_user(self, user_id, user_data):
        """Register a new user"""
        self.users[user_id] = {
            'id': user_id,
            'name': user_data.get('name'),
            'role': user_data.get('role'),
            'created': datetime.now().isoformat(),
            **{k: v for k, v in user_data.items() if k not in ['name', 'role']}
        }
        return self.users[user_id]

    def get_user(self, user_id):
        """Get user by ID"""
        return self.users.get(user_id)

    def get_users_by_role(self, role):
        """Get all users with specific role"""
        return [u for u in self.users.values() if u.get('role') == role]
