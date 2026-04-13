#!/usr/bin/env python
"""
REAL-TIME ACTIVITY MONITOR
Watches for model usage and admissions in real-time
Shows when predictions are made and patients are admitted
"""

import sys
import time
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from app import create_app
from app.services.sepsis_engine import SepsisEngine
from app.services.store import PatientStore

class ActivityMonitor:
    def __init__(self):
        self.app = create_app()
        self.engine = SepsisEngine()
        self.store = PatientStore()
        self.prediction_count = 0
        self.admission_count = len(self.store.get_admitted_patients())
        self.last_check_time = time.time()

    def log(self, level, message):
        """Log with timestamp"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        symbols = {
            "INFO": "ℹ",
            "SUCCESS": "✓",
            "MODEL": "🧠",
            "ADMIT": "🏥",
            "ALERT": "⚠",
            "ERROR": "✗"
        }
        symbol = symbols.get(level, "•")
        print(f"[{timestamp}] {symbol} {message}")

    def check_model_status(self):
        """Check if model is loaded"""
        is_model_loaded = self.engine.model is not None
        is_features_loaded = self.engine.features is not None

        status = "LOADED" if (is_model_loaded and is_features_loaded) else "NOT LOADED"
        level = "SUCCESS" if is_model_loaded else "ERROR"

        if is_model_loaded and is_features_loaded:
            self.log("MODEL", f"Model Healthy - {len(self.engine.features)} features ready")
        else:
            self.log("ERROR", f"Model Status: {status}")

        return is_model_loaded

    def test_prediction(self):
        """Test a prediction"""
        test_patient = {
            "name": "Monitor Test",
            "age": 60,
            "vitals": {
                "HR": 95, "Temp": 37.5, "SBP": 125, "MAP": 85, "DBP": 75,
                "Resp": 18, "O2Sat": 97, "EtCO2": 39,
            },
            "labs": {
                "WBC": 10.0, "Creatinine": 1.0, "Platelets": 200, "Lactate": 1.2,
                "Bilirubin": 0.8, "FiO2": 0.21, "pH": 7.40, "PaCO2": 40,
                "BaseExcess": 0, "HCO3": 24, "PTT": 28, "BUN": 15,
                "Chloride": 104, "Potassium": 4.0, "Sodium": 140, "Hgb": 13.5,
                "Glucose": 100,
            },
        }

        result = self.engine.predict(test_patient)
        self.prediction_count += 1

        model_used = "ML Model" in result['message']
        status = "ML Model" if model_used else "Fallback"

        self.log("MODEL", f"Prediction #{self.prediction_count}: {result['risk_score']:.1%} risk ({status})")

        return model_used

    def test_admission(self):
        """Test patient admission"""
        new_patient = {
            "name": f"Monitor Patient {self.admission_count + 1}",
            "age": 50 + (self.admission_count % 20),
            "gender": "M" if self.admission_count % 2 == 0 else "F",
            "ward": f"ICU-M{self.admission_count + 1}",
            "doctor": "Dr. Monitor",
            "doctorPhone": "+91-9999999999"
        }

        admitted = self.store.admit_patient(new_patient)
        self.admission_count += 1

        self.log("ADMIT", f"Patient #{self.admission_count} admitted: {admitted['name']} (ID: {admitted['id']})")

        return admitted

    def show_statistics(self):
        """Show current statistics"""
        print("\n" + "=" * 60)
        print("  STATISTICS")
        print("=" * 60)

        admitted_patients = self.store.get_admitted_patients()
        self.log("INFO", f"Total Admitted Patients: {len(admitted_patients)}")
        self.log("INFO", f"Predictions Tested: {self.prediction_count}")
        self.log("INFO", f"Admissions Tested: {self.admission_count}")

        if admitted_patients:
            high_risk = len([p for p in admitted_patients if p['sepsisRisk'] >= 0.75])
            moderate_risk = len([p for p in admitted_patients if 0.4 <= p['sepsisRisk'] < 0.75])
            low_risk = len([p for p in admitted_patients if p['sepsisRisk'] < 0.4])

            self.log("INFO", f"Risk Distribution - High: {high_risk} | Moderate: {moderate_risk} | Low: {low_risk}")

        print("=" * 60 + "\n")

    def show_menu(self):
        """Show interactive menu"""
        print("\n" + "=" * 60)
        print("  REAL-TIME ACTIVITY MONITOR - MENU")
        print("=" * 60)
        print("  1. Check Model Status")
        print("  2. Test Prediction (uses ML model)")
        print("  3. Test Patient Admission")
        print("  4. Show Statistics")
        print("  5. Run Full Diagnostic")
        print("  6. Exit")
        print("=" * 60)

    def run_full_diagnostic(self):
        """Run complete diagnostic"""
        print("\n" + "=" * 60)
        print("  RUNNING FULL DIAGNOSTIC")
        print("=" * 60)

        # Test 1: Model Status
        print("\n[1/3] Testing Model Status...")
        model_ok = self.check_model_status()

        # Test 2: Prediction
        print("\n[2/3] Testing Prediction...")
        pred_ok = self.test_prediction()

        # Test 3: Admission
        print("\n[3/3] Testing Admission...")
        admission_ok = self.test_admission() is not None

        # Summary
        print("\n" + "=" * 60)
        print("  DIAGNOSTIC SUMMARY")
        print("=" * 60)

        results = [
            ("Model Loading", "PASS" if model_ok else "FAIL"),
            ("ML Prediction", "PASS" if pred_ok else "FAIL"),
            ("Patient Admission", "PASS" if admission_ok else "FAIL"),
        ]

        for test, result in results:
            symbol = "✓" if result == "PASS" else "✗"
            print(f"  {symbol} {test}: {result}")

        all_pass = all(r == "PASS" for _, r in results)
        print("\n" + "=" * 60)
        if all_pass:
            print("  ALL SYSTEMS OPERATIONAL")
        else:
            print("  SOME ISSUES DETECTED - See above")
        print("=" * 60 + "\n")

    def interactive_mode(self):
        """Run interactive menu"""
        print("\n" + "🎯 " * 15)
        print("  REAL-TIME ACTIVITY MONITOR - INTERACTIVE MODE")
        print("🎯 " * 15)

        while True:
            self.show_menu()
            choice = input("Enter choice (1-6): ").strip()

            if choice == "1":
                self.check_model_status()
            elif choice == "2":
                self.test_prediction()
            elif choice == "3":
                self.test_admission()
            elif choice == "4":
                self.show_statistics()
            elif choice == "5":
                self.run_full_diagnostic()
            elif choice == "6":
                print("\nExiting monitor. Goodbye!")
                break
            else:
                self.log("ERROR", "Invalid choice. Please try again.")

    def auto_mode(self, duration=60):
        """Auto-test mode - runs tests for specified duration"""
        print("\n" + "⏱ " * 15)
        print("  AUTO-TEST MODE (60 seconds)")
        print("⏱ " * 15)

        start_time = time.time()
        test_count = 0

        while time.time() - start_time < duration:
            elapsed = int(time.time() - start_time)
            remaining = duration - elapsed

            if test_count % 3 == 0:
                self.log("INFO", f"Running tests ({remaining}s remaining)...")
                self.check_model_status()

            if test_count % 3 == 1:
                self.test_prediction()

            if test_count % 3 == 2:
                if test_count % 6 == 5:  # Admit every other cycle
                    self.test_admission()

            test_count += 1
            time.sleep(3)

        print("\n" + "=" * 60)
        self.show_statistics()
        print("Auto-test complete!")


def main():
    monitor = ActivityMonitor()

    print("\n" + "🔍 " * 18)
    print("    REAL-TIME ACTIVITY MONITOR")
    print("🔍 " * 18)

    mode = input("\nSelect mode:\n  (1) Interactive Menu\n  (2) Auto-Test (60s)\n  (3) Quick Diagnostic\n\nChoice (1-3): ").strip()

    if mode == "1":
        monitor.interactive_mode()
    elif mode == "2":
        monitor.auto_mode(60)
    elif mode == "3":
        monitor.run_full_diagnostic()
    else:
        print("Invalid choice. Running quick diagnostic...")
        monitor.run_full_diagnostic()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nMonitor stopped by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
