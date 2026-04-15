# API Reference - ML-Sepsis HITL System

## Base URL
```
http://localhost:5000
```

## Authentication
Currently uses localStorage-based session (demo/development only).
Production should use JWT tokens or session cookies.

---

## HITL Feedback Endpoints

### 1. Submit HITL Feedback Form
**Submit clinical feedback for a patient to train the model**

```
POST /api/hitl/submit
Content-Type: application/json
```

**Request Body:**
```json
{
  "clinician_id": "DR_SMITH",
  "patient_id": 1,
  "vitals_assessment": "stable|concerning|critical",
  "labs_assessment": "normal|abnormal|critical",
  "clinical_impression": "yes|no",
  "risk_assessment": "correct|incorrect|partial",
  "notes": "Additional clinical observations..."
}
```

**Response (201 Created):**
```json
{
  "success": true,
  "feedback": {
    "id": 1,
    "clinician_id": "DR_SMITH",
    "patient_id": 1,
    "timestamp": "2026-04-15T10:30:00",
    "vitals_assessment": "stable",
    "labs_assessment": "normal",
    "clinical_impression": "yes",
    "risk_assessment": "correct",
    "notes": "Additional observations",
    "status": "pending"
  },
  "can_retrain": false,
  "feedback_count": 1
}
```

**Status Code**: 201 Created
**Error**: 400 Bad Request if missing fields

---

### 2. Get HITL Feedback List
**Retrieve all HITL feedback forms (optionally filtered by clinician)**

```
GET /api/hitl/list?clinician_id=DR_SMITH
```

**Query Parameters:**
- `clinician_id` (optional): Filter by specific clinician ID

**Response (200 OK):**
```json
{
  "success": true,
  "feedback": [
    {
      "id": 1,
      "clinician_id": "DR_SMITH",
      "patient_id": 1,
      "timestamp": "2026-04-15T10:30:00",
      "vitals_assessment": "stable",
      "labs_assessment": "normal",
      "clinical_impression": "yes",
      "risk_assessment": "correct",
      "notes": "Patient stable",
      "status": "pending"
    },
    {
      "id": 2,
      "clinician_id": "DR_SMITH",
      "patient_id": 2,
      "timestamp": "2026-04-15T11:00:00",
      "vitals_assessment": "concerning",
      "labs_assessment": "abnormal",
      "clinical_impression": "yes",
      "risk_assessment": "incorrect",
      "notes": "Model underestimated risk",
      "status": "pending"
    }
  ],
  "count": 2,
  "can_retrain": false
}
```

---

### 3. Check Retraining Status
**Check if model retraining is available (10+ forms collected)**

```
GET /api/hitl/status
```

**Response (200 OK):**
```json
{
  "feedback_count": 8,
  "can_retrain": false,
  "forms_needed": 2
}
```

When `can_retrain` is `true` and `forms_needed` is 0:
- The retraining button in the UI becomes enabled
- Clinician can click to initiate model retraining

**Response when ready:**
```json
{
  "feedback_count": 10,
  "can_retrain": true,
  "forms_needed": 0
}
```

---

## Patient Endpoints

### 4. Get Patient Summary
**Download complete patient summary with all clinical feedback**

```
GET /api/patient/<patient_id>/summary
```

**Path Parameters:**
- `patient_id`: Integer ID of the patient

**Response (200 OK):**
```json
{
  "patient_info": {
    "id": 1,
    "name": "Rajesh Kumar",
    "age": 67,
    "gender": "M",
    "ward": "ICU-A2",
    "admitted": "2026-04-07"
  },
  "current_risk": {
    "risk_score": 0.87,
    "risk_level": "High",
    "updating_time": "2026-04-15T10:45:00"
  },
  "current_vitals": {
    "HR": 118,
    "Temp": 38.9,
    "SBP": 88,
    "MAP": 58,
    "DBP": 52,
    "Resp": 26,
    "O2Sat": 91,
    "EtCO2": 32
  },
  "current_labs": {
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
    "Glucose": 162
  },
  "clinical_feedback": [
    {
      "id": 1,
      "clinician_id": "DR_SMITH",
      "patient_id": 1,
      "timestamp": "2026-04-15T10:30:00",
      "vitals_assessment": "concerning",
      "labs_assessment": "abnormal",
      "clinical_impression": "yes",
      "risk_assessment": "correct",
      "notes": "Patient shows clear sepsis signs",
      "status": "pending"
    }
  ],
  "feedback_count": 1
}
```

**Error Response (404 Not Found):**
```json
{
  "error": "Patient not found"
}
```

---

## Existing Patient Endpoints (Still Available)

### Get All Patients
```
GET /api/patients
```

Returns list of all admitted patients.

### Get Specific Patient
```
GET /api/patient/<patient_id>
```

Returns patient details.

### Run Sepsis Prediction
```
POST /api/patient/<patient_id>/predict
```

Runs AI model prediction and returns risk score.

### Get Alerts
```
GET /api/alerts
```

Returns all sent alerts.

---

## Data Model Reference

### HITL Feedback Object
```javascript
{
  "id": number,                    // Unique feedback ID
  "clinician_id": string,          // Clinician who submitted
  "patient_id": number,            // Patient ID
  "timestamp": ISO8601 string,     // When feedback was submitted
  "vitals_assessment": string,     // stable|concerning|critical
  "labs_assessment": string,       // normal|abnormal|critical
  "clinical_impression": string,   // yes|no (does patient have sepsis?)
  "risk_assessment": string,       // correct|incorrect|partial
  "notes": string,                 // Optional notes
  "status": string                 // pending|processed (default: pending)
}
```

### Patient Summary Object
```javascript
{
  "patient_info": {
    "id": number,
    "name": string,
    "age": number,
    "gender": string,
    "ward": string,
    "admitted": date string
  },
  "current_risk": {
    "risk_score": number (0-1),    // Sepsis probability
    "risk_level": string,          // High|Moderate|Low
    "updating_time": ISO8601 string
  },
  "current_vitals": {},            // Current vital signs
  "current_labs": {},              // Current lab values
  "clinical_feedback": [],         // Array of HITL feedback objects
  "feedback_count": number         // Total feedback forms
}
```

---

## Usage Examples

### JavaScript/Fetch

**Submit HITL Feedback:**
```javascript
const feedback = {
  clinician_id: "DR_SMITH",
  patient_id: 1,
  vitals_assessment: "concerning",
  labs_assessment: "abnormal",
  clinical_impression: "yes",
  risk_assessment: "correct",
  notes: "Patient deteriorating"
};

fetch('/api/hitl/submit', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(feedback)
})
.then(r => r.json())
.then(data => {
  if (data.success) {
    console.log('Feedback submitted!', data.feedback);
    if (data.can_retrain) {
      console.log('Ready to retrain model!');
    }
  }
});
```

**Get Patient Summary:**
```javascript
fetch('/api/patient/1/summary')
  .then(r => r.json())
  .then(data => {
    console.log('Patient:', data.patient_info.name);
    console.log('Risk:', data.current_risk.risk_level);
    console.log('Feedback forms:', data.feedback_count);
  });
```

### cURL Examples

**Submit Feedback:**
```bash
curl -X POST http://localhost:5000/api/hitl/submit \
  -H 'Content-Type: application/json' \
  -d '{
    "clinician_id": "DR_SMITH",
    "patient_id": 1,
    "vitals_assessment": "stable",
    "labs_assessment": "normal",
    "clinical_impression": "no",
    "risk_assessment": "correct",
    "notes": "Patient improving"
  }'
```

**Check Status:**
```bash
curl http://localhost:5000/api/hitl/status
```

**Get Summary:**
```bash
curl http://localhost:5000/api/patient/1/summary | jq .
```

---

## Error Handling

### Common Error Responses

**400 Bad Request** - Missing required fields
```json
{
  "error": "Missing required field: patient_id"
}
```

**404 Not Found** - Patient doesn't exist
```json
{
  "error": "Patient not found"
}
```

**500 Internal Server Error** - Server-side issue
```json
{
  "success": false,
  "error": "Error message"
}
```

---

## Rate Limiting
Currently not implemented. For production, add:
- Per-clinician feedback submission limits
- API request throttling
- DDoS protection

---

## Security Considerations

### Current (Development)
⚠️ Uses localStorage - NOT SECURE for production

### Recommended (Production)
1. Implement JWT token-based authentication
2. Use HTTP-only cookies for sessions
3. Hash passwords with bcrypt
4. Enforce HTTPS only
5. Implement CORS properly
6. Add request validation and sanitization
7. Rate limit API endpoints
8. Log all HITL submissions for audit
9. Encrypt sensitive patient data
10. Implement role-based access control on backend

---

## Versioning
Current implementation: v1.0 (Development)
No API versioning yet. For production, consider:
- URL versioning: `/api/v1/hitl/submit`
- Header versioning: `API-Version: 1.0`
- Deprecation warnings
