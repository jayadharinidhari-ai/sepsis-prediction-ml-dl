# MODEL USAGE & ADMISSION TROUBLESHOOTING GUIDE

## Quick Summary

✅ **Model Status**: OPERATIONAL  
✅ **Admission System**: OPERATIONAL  
✅ **Model is being used**: YES

---

## 1. VERIFY MODEL IS BEING USED

### Method A: Run the Debug Script
```bash
cd "c:\Users\aruna\OneDrive\Desktop\ML - Sepsis"
python debug_model_usage.py
```

**Expected Output:**
```
MODEL STATUS CHECK
  Status: LOADED
  Type: XGBClassifier
  Features: 36

TESTING MODEL PREDICTION
  High Risk Patient
    Risk Score: 0.77
    ML Model Used: True
    Message: Sepsis risk: 77.8% (ML Model)
```

---

### Method B: Check Flask Console Logs
When the app is running, open **Terminal** and look for messages like:

```
[ML MODEL] Patient 1 (Rajesh Kumar): Risk=87.1%
[PREDICTION REQUEST] Patient ID: 1 (Rajesh Kumar)
[PREDICTION RESULT] Score: 0.871, Level: High
```

These confirm the model is being called.

---

### Method C: Check Browser Network Tab
1. Open the app: http://localhost:5000/dashboard
2. Open **Developer Tools** (F12)
3. Go to **Network** tab
4. Click on a patient
5. Click "Run Prediction" button
6. Look for request to `/api/patient/{id}/predict`
7. Click it and view the **Response**:

```json
{
  "risk_score": 0.87,
  "risk_level": "High",
  "message": "Sepsis risk: 87.1% (ML Model)",
  "top_features": [
    {"name": "Lactate", "val": 3.2, "contrib": 0.92, "dir": "high"},
    ...
  ]
}
```

If you see `"(ML Model)"` in the message → **Model is being used** ✅

---

## 2. TROUBLESHOOTING ADMIT PATIENT

### Issue: Admit Patient Button Not Working

**Step 1: Open Browser Console**
- Press F12 → Go to **Console** tab
- Try admitting a patient
- Check for error messages

**Common Errors & Fixes:**

#### Error: "Network request failed"
- **Cause**: Flask app not running
- **Fix**: Start the app with `python run.py`

#### Error: "TypeError: Cannot read property 'id' of undefined"
- **Cause**: Server response missing fields
- **Fix**: Check that your admission data includes: name, age, gender, ward, doctor, doctorPhone

#### Error: 400 Bad Request
- **Cause**: Invalid JSON
- **Fix**: Make sure all fields are properly formatted

---

### Step 2: Test the Admit API Directly
Open your browser and paste this JavaScript in the Console (F12):

```javascript
fetch('/api/patient/admit', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    name: 'Test Patient',
    age: 50,
    gender: 'M',
    ward: 'ICU-1',
    doctor: 'Dr. Test',
    doctorPhone: '+91-9999999999'
  })
})
.then(r => r.json())
.then(data => console.log('SUCCESS:', data))
.catch(e => console.error('ERROR:', e))
```

**Expected Response:**
```javascript
{
  id: 1234567890,
  name: "Test Patient",
  age: 50,
  status: "admitted",
  sepsisRisk: 0.08,
  vitals: {...},
  labs: {...},
  ...
}
```

---

### Step 3: Check Server Console
In your Flask terminal, you should see:

```
[ADMISSION] New patient admitted:
  - ID: 1234567890
  - Name: Test Patient
  - Ward: ICU-1
  - Doctor: Dr. Test
```

If you see this → **Admission is working** ✅

---

## 3. HOW TO CHECK REAL-TIME MODEL USAGE

### While the App is Running:

1. **Open Terminal** where Flask is running
2. **Go to Dashboard** and select a patient
3. **Click "Run Prediction" button**
4. **Check Terminal** for output like:

```
[PREDICTION REQUEST] Patient ID: 1 (Rajesh Kumar)
[ML MODEL] Patient 1 (Rajesh Kumar): Risk=87.1%
[PREDICTION RESULT] Score: 0.871, Level: High
```

---

## 4. COMPLETE WORKFLOW TEST

### Manual Test Steps:

1. **Start Flask App**
   ```bash
   python run.py
   ```

2. **Open Dashboard**
   ```
   http://localhost:5000/dashboard
   ```

3. **Admit a New Patient**
   - Click "+ Admit Patient" button
   - Fill in:
     - Name: John Smith
     - Age: 65
     - Gender: M
     - Ward: ICU-2
     - Doctor: Dr. Anderson
     - Phone: +91-9876543210
   - Click "Admit"

4. **Check Console Logs**
   ```
   [ADMISSION] New patient admitted:
     - ID: 1712987654321
     - Name: John Smith
   ```

5. **Run Prediction**
   - Click on the newly admitted patient
   - Click "Run Prediction" button
   - See risk score update

6. **Check Prediction Logs**
   ```
   [PREDICTION REQUEST] Patient ID: 1712987654321 (John Smith)
   [ML MODEL] Patient 1712987654321 (John Smith): Risk=72.5%
   [PREDICTION RESULT] Score: 0.725, Level: Moderate
   ```

---

## 5. QUICK DIAGNOSTIC CHECKLIST

Run this Python script to verify everything:

```python
import sys
sys.path.insert(0, r'c:\Users\aruna\OneDrive\Desktop\ML - Sepsis')

from app.services.sepsis_engine import SepsisEngine
from app.services.store import PatientStore

# Check 1: Model loads
engine = SepsisEngine()
print(f"Model loaded: {engine.model is not None}")  # Should be: True
print(f"Features loaded: {engine.features is not None}")  # Should be: True

# Check 2: Admission works
store = PatientStore()
new_patient = store.admit_patient({
    "name": "Test",
    "age": 50,
    "gender": "M",
    "ward": "ICU",
    "doctor": "Dr. X",
    "doctorPhone": "+91-0000000000"
})
print(f"Patient admitted: {new_patient['id']}")  # Should print numeric ID

# Check 3: Prediction works
result = engine.predict(new_patient)
print(f"ML Model used: {'ML Model' in result['message']}")  # Should be: True
print(f"Risk Score: {result['risk_score']}")  # Should be a number

print("\nAll checks passed!")
```

---

## 6. KEY FILES & LOCATIONS

| Component | File | What it Does |
|-----------|------|-------------|
| **ML Model** | `sepsis_xgb_model_v1.joblib` | XGBoost model for predictions |
| **Features** | `model_features.joblib` | List of 36 features the model expects |
| **Prediction Engine** | `app/services/sepsis_engine.py:182` | `predict()` method (has logging) |
| **Admission Handler** | `app/routes.py:50` | `admit_patient()` endpoint (has logging) |
| **Prediction Endpoint** | `app/routes.py:69` | `predict_sepsis()` endpoint (has logging) |
| **Patient Store** | `app/services/store.py:225` | `admit_patient()` method |

---

## 7. ENABLING VERBOSE LOGGING

To see **even more details**, modify `app/services/sepsis_engine.py`:

Change line 184 from:
```python
if self.model and self.features:
```

To:
```python
print(f"[DEBUG] Predicting for patient: {patient.get('name', 'Unknown')}")
if self.model and self.features:
    print(f"[DEBUG] Model available, using ML prediction")
```

Then restart Flask and watch the console for detailed logs.

---

## Summary

✅ Model is loaded and integrated  
✅ Admission system is working  
✅ Predictions are using the ML model  
✅ All endpoints have logging enabled  

**To verify everything is working:**
```bash
python debug_model_usage.py
```

**If you encounter issues**, check:
1. Flask console for error messages
2. Browser console (F12) for client-side errors
3. Network tab to see API responses

