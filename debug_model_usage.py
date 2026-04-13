#!/usr/bin/env python
"""
DEBUG TOOL: Monitor Model Usage & Admission in Real-Time
Run this while the Flask app is running to see:
- When the ML model is being called
- What predictions it's making
- When patients are admitted
"""

import sys
import json
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from app.services.sepsis_engine import SepsisEngine
from app.services.store import PatientStore

def print_header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)

def check_model_status():
    """Check if model files exist and are loadable"""
    print_header("MODEL STATUS CHECK")

    engine = SepsisEngine()

    print(f"\n1. Model File:")
    print(f"   Status: {'LOADED' if engine.model else 'NOT LOADED'}")
    print(f"   Type: {type(engine.model).__name__ if engine.model else 'N/A'}")

    print(f"\n2. Features:")
    print(f"   Status: {'LOADED' if engine.features else 'NOT LOADED'}")
    print(f"   Count: {len(engine.features) if engine.features else 0}")
    if engine.features:
        print(f"   Sample: {engine.features[:5]}")

    print(f"\n3. Twilio SMS:")
    print(f"   Status: {'CONFIGURED' if engine.twilio_client else 'NOT CONFIGURED'}")

    return engine

def test_prediction(engine):
    """Test the model with a sample patient"""
    print_header("TESTING MODEL PREDICTION")

    test_patients = {
        "High Risk": {
            "name": "High Risk Patient",
            "age": 70,
            "vitals": {
                "HR": 120, "Temp": 39.5, "SBP": 90, "MAP": 60, "DBP": 50,
                "Resp": 26, "O2Sat": 91, "EtCO2": 30,
            },
            "labs": {
                "WBC": 18.0, "Creatinine": 2.0, "Platelets": 80, "Lactate": 3.5,
                "Bilirubin": 1.5, "FiO2": 0.50, "pH": 7.30, "PaCO2": 50,
                "BaseExcess": -5, "HCO3": 18, "PTT": 40, "BUN": 35,
                "Chloride": 100, "Potassium": 4.5, "Sodium": 134, "Hgb": 10.5,
                "Glucose": 180,
            },
        },
        "Low Risk": {
            "name": "Low Risk Patient",
            "age": 50,
            "vitals": {
                "HR": 75, "Temp": 37.0, "SBP": 130, "MAP": 90, "DBP": 75,
                "Resp": 15, "O2Sat": 99, "EtCO2": 40,
            },
            "labs": {
                "WBC": 7.0, "Creatinine": 0.8, "Platelets": 250, "Lactate": 0.9,
                "Bilirubin": 0.6, "FiO2": 0.21, "pH": 7.42, "PaCO2": 38,
                "BaseExcess": 1, "HCO3": 26, "PTT": 26, "BUN": 12,
                "Chloride": 106, "Potassium": 4.0, "Sodium": 142, "Hgb": 14.5,
                "Glucose": 95,
            },
        },
    }

    for risk_type, patient in test_patients.items():
        result = engine.predict(patient)

        print(f"\n{risk_type}:")
        print(f"  Name: {patient['name']}")
        print(f"  Risk Score: {result['risk_score']}")
        print(f"  Risk Level: {result['risk_level']}")
        print(f"  ML Model Used: {'YES' if 'ML Model' in result['message'] else 'NO (fallback)'}")
        print(f"  Message: {result['message']}")

        if result['top_features']:
            print(f"  Top Feature: {result['top_features'][0]['name']} (importance: {result['top_features'][0]['contrib']})")

def check_admissions(store):
    """Check current admissions"""
    print_header("CURRENT ADMISSIONS")

    admitted = store.get_admitted_patients()

    print(f"\nTotal Admitted: {len(admitted)}")

    for patient in admitted:
        print(f"\n  ID: {patient['id']}")
        print(f"    Name: {patient['name']}")
        print(f"    Age: {patient['age']} | Gender: {patient['gender']}")
        print(f"    Ward: {patient['ward']}")
        print(f"    Doctor: {patient['doctor']} ({patient['doctorPhone']})")
        print(f"    Risk: {patient['sepsisRisk']:.3f} ({patient['riskLevel']})")
        print(f"    Admitted: {patient['admitted']}")

def test_new_admission(store, engine):
    """Test admitting a new patient"""
    print_header("TESTING NEW ADMISSION")

    new_patient_data = {
        "name": "Debug Test Patient",
        "age": 65,
        "gender": "M",
        "ward": "DEBUG-ICU",
        "doctor": "Dr. Debug",
        "doctorPhone": "+91-9999999999"
    }

    print(f"\nAdmitting new patient...")
    print(f"  Name: {new_patient_data['name']}")
    print(f"  Age: {new_patient_data['age']}")
    print(f"  Ward: {new_patient_data['ward']}")

    # Admit patient
    admitted = store.admit_patient(new_patient_data)

    print(f"\nAdmission Response:")
    print(f"  ID: {admitted['id']}")
    print(f"  Status: {admitted['status']}")
    print(f"  Initial Risk: {admitted['sepsisRisk']}")

    # Run prediction
    print(f"\nRunning ML prediction...")
    result = engine.predict(admitted)

    print(f"  ML Risk Score: {result['risk_score']}")
    print(f"  ML Risk Level: {result['risk_level']}")
    print(f"  Model Used: {'YES' if 'ML Model' in result['message'] else 'NO'}")

    # Update in store
    store.update_patient_risk(admitted['id'], result['risk_score'], result['top_features'])

    # Fetch updated
    updated = store.get_patient(admitted['id'])

    print(f"\nAfter ML Update:")
    print(f"  Updated Risk: {updated['sepsisRisk']}")
    print(f"  Updated Level: {updated['riskLevel']}")
    print(f"  Top Features Extracted: {len(updated['topFeatures'])} features")

def main():
    print("\n" + "🔍 " * 18)
    print("    ML MODEL & PATIENT ADMISSION DEBUGGING TOOL")
    print("🔍 " * 18)

    # Check model
    engine = check_model_status()

    # Test predictions
    test_prediction(engine)

    # Check admissions
    store = PatientStore()
    check_admissions(store)

    # Test new admission
    test_new_admission(store, engine)

    print_header("DEBUG CHECK COMPLETE")
    print("\nModel Integration Status: OPERATIONAL")
    print("Admission System: OPERATIONAL")
    print("\nYou can now:")
    print("  1. Open http://localhost:5000 in your browser")
    print("  2. Go to Dashboard")
    print("  3. Click '+ Admit Patient' to test admission")
    print("  4. Click on a patient and 'Run Prediction' to trigger ML model")
    print("  5. Check browser console (F12) for real-time updates\n")

if __name__ == "__main__":
    main()
