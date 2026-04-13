# ⚡ Real-Time Monitoring System - Complete Guide

## Overview

Your ICU Sepsis Early Warning System now includes **realistic real-time patient monitoring** with:
- ✅ Live vital sign updates (every 5 seconds)
- ✅ Automatic sepsis progression scenarios
- ✅ Audio alerts (double-beep buzzer)
- ✅ SMS alerts to doctors
- ✅ Real-time risk score updates

---

## How It Works

### **Backend Architecture**

```
Flask App (run.py)
    ↓
PatientSimulator (background thread)
    ├─ Updates vitals every 5 seconds
    ├─ Simulates clinical scenarios
    ├─ Manages patient progression phases
    └─ Triggers alerts when risk ≥ 75%
    ↓
PatientStore (in-memory database)
    └─ Stores patient data
    ↓
REST API Endpoints (/api/patient/<id>)
    ↓
Frontend Dashboard (polling)
    ├─ Fetches data every 3 seconds
    ├─ Updates vitals display
    ├─ Re-renders charts
    └─ Triggers audio alerts
```

### **Data Flow**
1. **Simulator** updates patient vitals in background (every 5 sec)
2. **Dashboard** polls for latest data (every 3 sec)
3. **Charts** automatically refresh with new data points
4. **Alerts** triggered when risk crosses thresholds
5. **User** sees live updates in real-time

---

## Patient Scenarios

### **Patient 1: Rajesh Kumar** 🔴 HIGH RISK

**Status**: Septic Shock (Stable High-Risk)

**Vitals** (continuously varying):
- **HR**: 115-130 bpm (severe tachycardia)
- **Temperature**: 38.5-40.2°C (high fever)
- **MAP**: 52-65 mmHg ← **CRITICALLY LOW** (septic shock)
- **SBP/DBP**: 80-100 / 48-60 mmHg
- **Resp**: 24-32 /min (tachypnea)
- **O₂ Sat**: 88-94% (hypoxia)

**Labs** (continuously varying):
- **Lactate**: 2.8-4.5 mmol/L (HIGH - metabolic compromise)
- **WBC**: 17-22 k/μL (HIGH - infection)
- **Creatinine**: 1.8-3.0 mg/dL (HIGH - kidney injury)
- **Platelets**: 70-100 k/μL (LOW - DIC risk)
- **pH**: 7.28 (acidosis)

**Risk Score**: 87-90% (always HIGH)

**Alerts**:
- ⚠️ Automatic SMS alert every 5 minutes
- 🔊 Audio buzzer active
- 🟥 Risk badge shows "High"

**Clinical Interpretation**:
- Multiple organ dysfunction
- Septic shock with hemodynamic instability
- Requires aggressive intervention
- Very poor prognosis without treatment

---

### **Patient 2: Meena Subramaniam** 🟡→🔴 MODERATE → HIGH

**Status**: Progressive Sepsis (Key Real-Time Demo!)

**This patient DETERIORATES in real-time. Watch the progression:**

#### **PHASE 1: Moderate Risk** (First ~75 seconds)

**Vitals**:
- **HR**: 92-110 bpm (elevated but stable)
- **Temperature**: 37.5-39.0°C (rising slowly)
- **MAP**: 72-82 mmHg (acceptable)
- **Resp**: 18-24 /min (normal to slightly elevated)
- **O₂ Sat**: 94-98% (adequate)

**Labs**:
- **WBC**: 11-15 k/μL (elevated but not critical)
- **Lactate**: 1.4-2.2 mmol/L (borderline)
- **Creatinine**: ~0.9-1.1 mg/dL (normal)
- **Platelets**: 180-220 k/μL (normal)

**Risk Score**: ~41-70% (MODERATE)

**Appearance**: Patient looks sick but stable. Could be sepsis developing.

#### **PHASE 2: Critical Deterioration** (After ~75 seconds)

⚠️ **SUDDEN DETERIORATION** - Patient status changes dramatically

**Vitals** (rapidly worsening):
- **HR**: 105-135 bpm (severe tachycardia) ↑
- **Temperature**: 38.8-40.5°C (high fever) ↑
- **MAP**: 60-75 mmHg (CRITICAL DROP) ↓↓
- **SBP**: 95-120 mmHg (variable)
- **DBP**: 55-70 mmHg (low)
- **Resp**: 20-32 /min (tachypnea) ↑
- **O₂ Sat**: 91-97% (variable, sometimes low) ↓

**Labs** (worsening):
- **Lactate**: 2.0-3.5 mmol/L (elevated - tissue hypoxia) ↑
- **WBC**: 14-20 k/μL (increasing) ↑
- **Creatinine**: 1.1-1.8 mg/dL (rising - kidney dysfunction) ↑
- **Platelets**: Trending down ↓

**Risk Score**: 75-88% (HIGH - ALERT)

**What Changed**: Multiple organ system deterioration
- MAP drop = septic shock (vasodilation + reduced contractility)
- Lactate rise = tissue hypoxia (inadequate oxygen delivery)
- Creatinine rise = kidney injury
- Lactate + low MAP + organ dysfunction = **SEPTIC SHOCK**

**Alert Triggers**:
- 🔊 **Audio buzzer activates** when risk hits 75%
- 📱 **SMS alert sent** to doctor
- 🟥 Risk badge changes to "High"
- ⏰ Alerts repeat every 5 minutes

**Clinical Decision Point**:
- This is when ICU team should intervene
- Start pressors for MAP support
- Obtain blood cultures
- Start antibiotics STAT
- Fluid resuscitation

---

### **Patient 3: Anbu Selvam** 🟢 LOW RISK (Stable)

**Status**: Stable, Normal ICU Patient

**Vitals** (minimal variation):
- **HR**: 74-85 bpm (normal)
- **Temperature**: 36.8-37.3°C (normal)
- **MAP**: 87-95 mmHg (normal)
- **Resp**: 15-19 /min (normal)
- **O₂ Sat**: 97-99% (excellent)

**Labs** (all normal):
- **WBC**: 8.0-9.0 k/μL (normal)
- **Lactate**: 0.8-1.2 mmol/L (normal)
- **Creatinine**: <1.0 mg/dL (normal)
- **All other labs**: Within normal limits

**Risk Score**: 8-12% (LOW)

**Status**: ✅ No alerts, routine monitoring
- Patient is stable
- No concerning trends
- Continue standard care

---

## Real-Time Features Demonstrated

### **1. Live Vital Sign Updates** 📊

**What's Happening**:
- Every 5 seconds, the simulator updates each patient's vitals
- Changes are small and clinically realistic (random walk variation)
- Reflects real ICU vital sign fluctuations

**Where to See It**:
- Top right panel: "Vitals" section
- Numbers update every 5 seconds
- Historical data accumulates in trends

**Example**: Meena's HR over 2 minutes
```
21:11:41 - HR: 96 bpm
21:11:46 - HR: 98 bpm
21:11:51 - HR: 102 bpm
21:11:56 - HR: 105 bpm
21:12:01 - HR: 108 bpm
...continues rising...
21:12:56 - HR: 125 bpm (post-75 sec jump)
```

### **2. Real-Time Risk Score Updates** 📈

**What's Happening**:
- Risk score recalculated based on current vitals
- Changes gradually for stable patients
- Jumps rapidly for deteriorating patients (Meena)

**Where to See It**:
- Left panel: "Sepsis Risk Score"
- Large percentage display
- Color changes: 🟢 Green → 🟡 Yellow → 🔴 Red

**Example**: Meena's risk progression
```
Time        Risk    Trend           Phase
00:00       41%     → slowly rising  MODERATE
00:15       48%     → slowly rising  MODERATE
00:30       56%     → rising         MODERATE
00:45       65%     → rising         MODERATE
01:00       73%     → rising         MODERATE
01:15       75%     → CRITICAL!      HIGH ← Alert!
01:20       82%     → stabilizing    HIGH
01:25       88%     → stable high    HIGH
```

### **3. Automatic Chart Updates** 📉

**What's Happening**:
- Each vital sign has its own trend chart
- Shows last 8 data points (rolling window)
- Charts update automatically every 3 seconds

**Where to See It**:
- "8-hr Trends" section
- Tab buttons: HR, Temp, MAP, WBC
- Red warning line for abnormal ranges

**Benefits**:
- See visual trends at a glance
- Identify rapid vs. slow changes
- Pattern recognition for clinical events

### **4. Audio Alerts** 🔊

**What's Happening**:
- Double-beep buzzer plays when risk ≥ 75%
- Uses Web Audio API (browser-based)
- 800Hz sine wave, 0.5 seconds each beep

**Where to Hear It**:
- When Meena's risk hits HIGH threshold
- Only the first time risk crosses 75%
- Resets if risk drops below 75% and rises again

**Technical Details**:
- Browser-based audio (no server sound)
- May be blocked by browser autoplay policy
- Click dashboard first if sound not working
- Can be integrated with hospital buzzer system

### **5. SMS Alerts** 📱

**What's Happening**:
- When risk ≥ 75%, SMS queued to doctor's phone
- Sends every 5 minutes (to avoid spam)
- Includes patient name, risk %, ward, doctor name

**Where to See It**:
- "Active Alerts" section shows timestamp
- "SMS Alert Log" shows all sent alerts
- SMS format:
  ```
  🚨 SEPSIS ALERT: Rajesh Kumar - Risk 87%
  Ward: ICU-A2
  Doctor: Dr. Priya Nair
  Immediate assessment required
  ```

**Status**:
- ✅ Alert logging works (visible in dashboard)
- ⚠️ Real SMS requires Twilio setup (see SMS_SETUP.md)
- Without Twilio: Alerts logged but not sent to phone

---

## Watching the System in Action

### **Best Demonstration Sequence**

**Time: ~3 minutes total**

```
[0:00-0:30]
1. Open dashboard: http://localhost:5000/dashboard
2. Click "Meena Subramaniam"
3. Note her initial status: Moderate Risk (~41%), normal vitals
4. Watch vitals change every 5 seconds
5. Observe risk slowly increasing (41% → 50% → 60%)

[0:30-1:15]
6. Risk continues rising (60% → 70% → 75%)
7. Vitals visibly worsening: HR↑, Temp↑, MAP↓
8. Watch when risk crosses 75% (HIGH threshold)
9. 🔊 Audio alert plays (if enabled)
10. Alert appears in "Active Alerts" section

[1:15-1:30]
11. Risk stabilizes in HIGH range (75-88%)
12. View trend graphs showing deterioration
13. Check "Top Features" to understand why high risk
14. Observe SMS alert logged with timestamp

[1:30-2:00]
15. Check other patients:
    - Rajesh: Always HIGH, always alerting
    - Anbu: Always LOW, stable
16. Compare vital ranges between patients
17. Note different risk stratifications

[2:00-3:00]
18. Let system continue monitoring
19. Watch alerts repeat every 5 minutes for high-risk
20. Observe new vitals added to trend graphs
```

---

## Configuration Options

### **Simulator Update Interval**

**Current**: Vitals update every 5 seconds

**To Change** (in `app/services/simulator.py`):
```python
time.sleep(5)  # Line 34 - change number (seconds)
```

### **Dashboard Poll Interval**

**Current**: Dashboard fetches data every 3 seconds

**To Change** (in `app/templates/dashboard.html`):
```javascript
}, 3000);  // Line 194 - change number (milliseconds)
```

### **Alert SMS Frequency**

**Current**: Every 5 minutes (300 seconds)

**To Change** (in `app/services/simulator.py`):
```python
(now - last_alert).total_seconds() >= 300  # Line 199
```

### **Risk Thresholds**

**Current**:
- High Risk: ≥ 75%
- Moderate Risk: 40-75%
- Low Risk: < 40%

**To Change** (in `app/services/simulator.py`):
```python
if patient["sepsisRisk"] >= 0.75:  # Line 160
    patient["riskLevel"] = "High"
elif patient["sepsisRisk"] >= 0.4:  # Line 162
    patient["riskLevel"] = "Moderate"
```

---

## Troubleshooting

### **Vitals Not Updating**

**Problem**: Dashboard shows same values, no changes

**Solutions**:
1. Refresh page (F5)
2. Check browser console (F12) for errors
3. Verify Flask server is running
4. Check API endpoint: `http://localhost:5000/api/patient/2`
5. Ensure patient is "admitted" (status='admitted')

### **Audio Alert Not Playing**

**Problem**: No buzzer sound when risk hits HIGH

**Solutions**:
1. **Browser autoplay policy**: Click dashboard first to enable audio
2. **Browser settings**: Check audio is not muted
3. **Check console**: Open F12 developer tools
4. **Test audio**: Open browser audio context
5. **Try different browser**: Chrome/Firefox/Edge

### **Charts Not Updating**

**Problem**: Trend charts show old data

**Solutions**:
1. Click "HR" or other chart tab to refresh
2. Hard refresh page (Ctrl+Shift+R)
3. Check if `trend` data exists in patient object
4. Verify `initCharts()` is being called after update
5. Clear browser cache

### **Alerts Not Showing**

**Problem**: Risk is HIGH but no alerts appearing

**Solutions**:
1. Check patient's doctor phone number is set
2. Verify risk score is actually ≥ 75%
3. Check last alert time (5-minute throttle in place)
4. Look in console for alert errors
5. Test with manual "Send Alert" button

### **Meena Not Deteriorating**

**Problem**: Patient stays in Moderate phase

**Solutions**:
1. Wait longer (75+ seconds needed)
2. Verify duration counter is incrementing
3. Check simulator is running (should see in Flask logs)
4. Hard refresh dashboard
5. Check patient ID is correct (should be 2)

---

## API Endpoints for Real-Time Data

### **Get Live Patient Data**

```bash
GET /api/patient/<id>
```

**Response**:
```json
{
  "id": 2,
  "name": "Meena Subramaniam",
  "sepsisRisk": 0.88,
  "riskLevel": "High",
  "vitals": {
    "HR": 115,
    "Temp": 39.5,
    "MAP": 70,
    "O2Sat": 95
  },
  "labs": {
    "WBC": 18,
    "Lactate": 3.0,
    "Creatinine": 1.5
  }
}
```

**Update Frequency**: Data updates every 5 seconds from simulator

### **Get All Admitted Patients**

```bash
GET /api/patients
```

**Returns**: Array of all admitted patients with live vitals

---

## How to Extend the System

### **Add New Patient Scenarios**

Edit `app/services/simulator.py`:

```python
elif patient_id == 4:  # New patient ID
    # Your custom vital variation logic
    vitals["HR"] = self._adjust_value(vitals.get("HR", 80), 75, 95, 1)
    # ... more vitals ...
    patient["sepsisRisk"] = custom_risk_calculation()
```

### **Add Alert Methods**

In `app/services/sepsis_engine.py`:

```python
def send_alert(self, patient):
    # Current: Log or SMS
    # Add: Email, Pager, Hospital PA system, etc.
```

### **Integrate with Hospital Devices**

Connect to bedside monitors:
```python
# In simulator, replace vital updates with real device data
vital_data = read_from_bedside_monitor(patient_id)
patient["vitals"] = vital_data
```

---

## Medical Context

### **Sepsis Recognition Criteria (qSOFA)**

Meena demonstrates progression through:

1. **Early signs** (Phase 1):
   - Fever (Temp ↑)
   - Tachycardia (HR ↑)
   - Elevated WBC
   - → Still compensating (Moderate Risk)

2. **Septic Shock Signs** (Phase 2):
   - **Hypotension** (MAP ↓ - CRITICAL)
   - Lactate elevation (tissue hypoxia)
   - Altered perfusion
   - → Organ dysfunction (HIGH Risk)

### **Why MAP Drop is Critical**

Mean Arterial Pressure (MAP) < 65 mmHg indicates:
- **Septic shock** (distributive shock)
- Inadequate organ perfusion
- Risk of multi-organ failure
- **Requires immediate intervention**
  - Vasopressors (norepinephrine)
  - Fluids
  - Antibiotics

### **Why Lactate Matters**

Lactate > 2 mmol/L indicates:
- Anaerobic metabolism (tissue hypoxia)
- Metabolic acidosis
- Poor prognosis if persistent
- Used for severity scoring (SOFA, qSOFA)

---

## Next Steps

1. ✅ **Understand the demo**: Run through demonstration sequence above
2. ✅ **Monitor Meena**: Watch real-time deterioration
3. ✅ **Review Rajesh**: See what septic shock looks like
4. ✅ **Set up SMS**: Follow SMS_SETUP.md for real alerts
5. ✅ **Customize**: Modify patient scenarios for your use case
6. ✅ **Deploy**: Follow DEPLOYMENT_SUMMARY.md for production

---

## Contact & Support

For questions about real-time features:
- Check logs: `flask --app app run --debug`
- Review simulator code: `app/services/simulator.py`
- Check dashboard: `app/templates/dashboard.html`
- API docs: `app/routes.py`

Good luck with your sepsis monitoring system! 🏥⚡
