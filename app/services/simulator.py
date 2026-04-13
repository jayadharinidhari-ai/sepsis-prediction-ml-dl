import threading
import time
import random
from datetime import datetime, timedelta


class PatientSimulator:
    """Simulates real-time patient vital sign changes"""

    def __init__(self, store):
        self.store = store
        self.running = False
        self.thread = None
        self.last_alert_time = {}
        self.sepsis_progression = {}  # Track sepsis progression for patients

    def start(self):
        """Start the simulation thread"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._simulate_loop, daemon=True)
            self.thread.start()
            print("[OK] Patient simulator started")

    def stop(self):
        """Stop the simulation"""
        self.running = False

    def _simulate_loop(self):
        """Main simulation loop - updates every 5 seconds"""
        while self.running:
            try:
                self._update_all_patients()
                time.sleep(5)  # Update every 5 seconds
            except Exception as e:
                print(f"Simulator error: {e}")

    def _update_all_patients(self):
        """Update all admitted patients"""
        for patient_id, patient in self.store.patients.items():
            if patient["status"] == "admitted":
                self._update_patient_vitals(patient_id, patient)

    def _update_patient_vitals(self, patient_id, patient):
        """Update individual patient vitals with realistic variation"""
        vitals = patient.get("vitals", {})
        labs = patient.get("labs", {})
        trend = patient.get("trend", {})

        # Initialize progression tracking
        if patient_id not in self.sepsis_progression:
            self.sepsis_progression[patient_id] = {"phase": "stable", "duration": 0}

        phase = self.sepsis_progression[patient_id]

        # ===== PATIENT 1: RAJESH KUMAR (High Risk - Septic Shock) =====
        if patient_id == 1:
            # Deteriorating patient - simulate septic shock
            vitals["HR"] = self._adjust_value(vitals.get("HR", 118), 115, 130, 2)
            vitals["Temp"] = self._adjust_value(vitals.get("Temp", 38.9), 38.5, 40.2, 0.2)
            vitals["MAP"] = self._adjust_value(vitals.get("MAP", 58), 52, 65, 1.5)
            vitals["SBP"] = self._adjust_value(vitals.get("SBP", 88), 80, 100, 2)
            vitals["DBP"] = self._adjust_value(vitals.get("DBP", 52), 48, 60, 1)
            vitals["Resp"] = self._adjust_value(vitals.get("Resp", 26), 24, 32, 0.5)
            vitals["O2Sat"] = self._adjust_value(vitals.get("O2Sat", 91), 88, 94, 0.5)

            labs["Lactate"] = self._adjust_value(labs.get("Lactate", 3.2), 2.8, 4.5, 0.15)
            labs["WBC"] = self._adjust_value(labs.get("WBC", 18.4), 17, 22, 0.3)
            labs["Creatinine"] = self._adjust_value(labs.get("Creatinine", 2.1), 1.8, 3.0, 0.1)
            labs["Platelets"] = self._adjust_value(labs.get("Platelets", 88), 70, 100, 1)

            # Update risk score - fluctuate around high range (85-95%)
            current_risk = patient.get("sepsisRisk", 0.87)
            change = random.uniform(-0.015, 0.015)  # ±1.5% variation
            patient["sepsisRisk"] = max(0.80, min(0.95, current_risk + change))

        # ===== PATIENT 2: MEENA SUBRAMANIAM (Moderate -> High progression) =====
        elif patient_id == 2:
            # This patient will gradually worsen from MODERATE to HIGH
            phase["duration"] += 1

            if phase["duration"] < 15:  # First 75 seconds - moderate
                phase["phase"] = "moderate"
                vitals["HR"] = self._adjust_value(vitals.get("HR", 96), 92, 110, 1.5)
                vitals["Temp"] = self._adjust_value(vitals.get("Temp", 37.8), 37.5, 39.0, 0.15)
                vitals["MAP"] = self._adjust_value(vitals.get("MAP", 76), 72, 82, 1)
                vitals["Resp"] = self._adjust_value(vitals.get("Resp", 19), 18, 24, 0.3)
                vitals["O2Sat"] = self._adjust_value(vitals.get("O2Sat", 96), 94, 98, 0.2)

                labs["WBC"] = self._adjust_value(labs.get("WBC", 12.1), 11, 15, 0.2)
                labs["Lactate"] = self._adjust_value(labs.get("Lactate", 1.6), 1.4, 2.2, 0.1)

                # Risk increases slowly
                patient["sepsisRisk"] = min(
                    0.75, patient.get("sepsisRisk", 0.41) + random.uniform(0.005, 0.015)
                )

            else:  # After 75 seconds - rapid deterioration to HIGH RISK
                phase["phase"] = "deteriorating"
                vitals["HR"] = self._adjust_value(vitals.get("HR", 110), 105, 135, 2.5)
                vitals["Temp"] = self._adjust_value(vitals.get("Temp", 39.0), 38.8, 40.5, 0.3)
                vitals["MAP"] = self._adjust_value(vitals.get("MAP", 72), 60, 75, 2)
                vitals["SBP"] = self._adjust_value(vitals.get("SBP", 112), 95, 120, 2)
                vitals["DBP"] = self._adjust_value(vitals.get("DBP", 68), 55, 70, 1.5)
                vitals["Resp"] = self._adjust_value(vitals.get("Resp", 22), 20, 32, 0.8)
                vitals["O2Sat"] = self._adjust_value(vitals.get("O2Sat", 95), 91, 97, 0.5)

                labs["WBC"] = self._adjust_value(labs.get("WBC", 15), 14, 20, 0.4)
                labs["Lactate"] = self._adjust_value(labs.get("Lactate", 2.2), 2.0, 3.5, 0.2)
                labs["Creatinine"] = self._adjust_value(labs.get("Creatinine", 1.1), 1.1, 1.8, 0.05)

                # Risk jumps to HIGH
                patient["sepsisRisk"] = min(
                    0.88, patient.get("sepsisRisk", 0.75) + random.uniform(0.01, 0.03)
                )

        # ===== PATIENT 3: ANBU SELVAM (Stable Low Risk) =====
        elif patient_id == 3:
            # Stable patient - normal variation
            vitals["HR"] = self._adjust_value(vitals.get("HR", 78), 74, 85, 0.5)
            vitals["Temp"] = self._adjust_value(vitals.get("Temp", 37.0), 36.8, 37.3, 0.1)
            vitals["MAP"] = self._adjust_value(vitals.get("MAP", 90), 87, 95, 0.5)
            vitals["Resp"] = self._adjust_value(vitals.get("Resp", 16), 15, 19, 0.2)
            vitals["O2Sat"] = self._adjust_value(vitals.get("O2Sat", 98), 97, 99, 0.1)

            labs["WBC"] = self._adjust_value(labs.get("WBC", 8.4), 8.0, 9.0, 0.1)
            labs["Lactate"] = self._adjust_value(labs.get("Lactate", 0.9), 0.8, 1.2, 0.05)

            # Risk stays LOW
            patient["sepsisRisk"] = max(0.08, patient.get("sepsisRisk", 0.12) - random.uniform(0, 0.01))

        # ===== Any new admitted patients - stable =====
        else:
            # New patients - stable normal vitals
            vitals["HR"] = self._adjust_value(vitals.get("HR", 76), 72, 82, 0.5)
            vitals["Temp"] = self._adjust_value(vitals.get("Temp", 37.0), 36.8, 37.4, 0.1)
            vitals["MAP"] = self._adjust_value(vitals.get("MAP", 82), 79, 88, 0.5)
            vitals["Resp"] = self._adjust_value(vitals.get("Resp", 16), 15, 19, 0.2)
            vitals["O2Sat"] = self._adjust_value(vitals.get("O2Sat", 98), 97, 99, 0.1)

            patient["sepsisRisk"] = max(0.08, patient.get("sepsisRisk", 0.10) - random.uniform(0, 0.005))

        # Update trends (keep last 8 data points)
        if "trend" not in patient:
            patient["trend"] = {"HR": [], "Temp": [], "MAP": [], "WBC": []}

        for key in ["HR", "Temp", "MAP"]:
            if key not in patient["trend"]:
                patient["trend"][key] = []
            patient["trend"][key].append(vitals.get(key, 0))
            if len(patient["trend"][key]) > 8:
                patient["trend"][key] = patient["trend"][key][-8:]

        # Update WBC trend from labs
        if "WBC" not in patient["trend"]:
            patient["trend"]["WBC"] = []
        patient["trend"]["WBC"].append(labs.get("WBC", 8))
        if len(patient["trend"]["WBC"]) > 8:
            patient["trend"]["WBC"] = patient["trend"]["WBC"][-8:]

        # Update risk level classification
        if patient["sepsisRisk"] >= 0.75:
            patient["riskLevel"] = "High"
        elif patient["sepsisRisk"] >= 0.4:
            patient["riskLevel"] = "Moderate"
        else:
            patient["riskLevel"] = "Low"

        # Store updated vitals and labs
        patient["vitals"] = vitals
        patient["labs"] = labs

        # Check for alerts
        self._check_and_send_alerts(patient_id, patient)

    def _adjust_value(self, current, min_val, max_val, variation):
        """Adjust a vital value with realistic variation"""
        if current is None:
            current = (min_val + max_val) / 2

        # Random walk with bounds
        change = random.uniform(-variation, variation)
        new_val = current + change

        # Keep within bounds
        return max(min_val, min(max_val, round(new_val, 2)))

    def _check_and_send_alerts(self, patient_id, patient):
        """Check risk thresholds and send alerts"""
        risk = patient.get("sepsisRisk", 0)
        now = datetime.now()

        # Initialize alert tracking
        if patient_id not in self.last_alert_time:
            self.last_alert_time[patient_id] = None

        # HIGH RISK - send alert every 5 minutes
        if risk >= 0.75:
            last_alert = self.last_alert_time.get(patient_id)
            should_alert = (
                last_alert is None or (now - last_alert).total_seconds() >= 300
            )  # 5 minutes

            if should_alert:
                self._send_alert(patient_id, patient, risk)
                self.last_alert_time[patient_id] = now

                # Add alert to patient record
                alert_time = now.strftime("%H:%M")
                alert_msg = f"Sepsis probability {risk*100:.0f}% - IMMEDIATE ACTION REQUIRED"

                if "alerts" not in patient:
                    patient["alerts"] = []

                patient["alerts"].insert(0, {"time": alert_time, "msg": alert_msg})
                if len(patient["alerts"]) > 5:
                    patient["alerts"] = patient["alerts"][:5]

    def _send_alert(self, patient_id, patient, risk):
        """Send SMS alert to doctor"""
        from app.services.sepsis_engine import SepsisEngine

        engine = SepsisEngine()
        doctor_name = patient.get("doctor", "Doctor")
        doctor_phone = patient.get("doctorPhone", "")
        patient_name = patient.get("name", "Patient")

        message = f"🚨 SEPSIS ALERT: {patient_name} - Risk {risk*100:.0f}%\nWard: {patient.get('ward', 'ICU')}\nImmediate assessment required. - ICU Monitor"

        try:
            result = engine.send_alert(patient)
            print(f"[OK] Alert sent to {doctor_name}: {result}")
        except Exception as e:
            print(f"[WARNING] Alert failed: {e}")
