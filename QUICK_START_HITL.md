# Quick Start Guide - Multi-Role HITL System

## 🚀 Quick Testing Steps

### Step 1: Start the Application
```bash
cd /Users/jayadhariniradhakrishnan/ML-Sepsis
python3 run.py
```
Then visit: `http://localhost:5000`

### Step 2: Test Clinician Login
1. Click "👨‍⚕️ Clinician" tab
2. Enter:
   - ID: `DR_SMITH`
   - Name: `Dr. John Smith`
   - Department: `ICU`
3. Click "Login as Clinician"

### Step 3: Explore Patient Tab (First Tab)
- See list of 3 admitted patients with risk badges
- Each patient has:
  - Risk level (High/Moderate/Low)
  - Demographics
  - 📋 **Add HITL Form** button
  - ⬇️ **Download Summary** button

### Step 4: Submit HITL Form
1. Click "Add HITL Form" on any patient
2. Modal opens with patient name
3. Fill the form:
   - Vitals Assessment: Select any option (Stable, Concerning, Critical)
   - Labs Assessment: Select any option
   - Clinical Impression: YES or NO
   - Risk Assessment: Select accuracy assessment
   - Notes: Add optional comments
4. Click "Submit Training Form"
5. Success message appears → Modal closes

### Step 5: Check HITL Tab (Second Tab)
1. Click "📋 HITL Training Forms" tab
2. See statistics:
   - Forms submitted: Should show 1 (or more if you submitted multiple)
   - Forms needed to retrain: Shows remaining (9, 8, 7... when you reach 10 → 0)
   - Progress: Shows percentage (10%, 20%, ... 100%)
3. Submitted forms listed below with all details

### Step 6: Download Patient Summary
1. Go back to Patients tab
2. Click "⬇️ Download Summary" on any patient
3. A `.txt` file downloads with:
   - Patient info
   - Current vitals & labs
   - Risk assessment
   - All HITL feedback forms
   - Timestamps

### Step 7: Reach 10 Forms & Retrain
1. Submit HITL forms from different patients (9 total to reach 10)
2. After 10th form submission:
   - Go to HITL tab
   - "Retraining Progress" shows 100%
   - Button text changes to "🔄 Retrain Model (Ready!)"
   - Button becomes enabled (green, clickable)
3. Click button to initiate retraining

### Step 8: Test Other Roles
**Nurse Login:**
- Tab: "👩‍⚕️ Nurse"
- ID: `NURSE_JOHN`
- Name: `John Nurse`
- Ward: `ICU-A2`
- Result: Access monitoring dashboard only (no HITL)

**Patient Login:**
- Tab: "🧑 Patient"
- ID: `PAT_001`
- Name: `Patient Name`
- DOB: `01/01/1950`
- Result: Access patient-specific info only

## 🔄 API Endpoints (For Testing with Curl)

### Submit HITL Feedback
```bash
curl -X POST http://localhost:5000/api/hitl/submit \
  -H "Content-Type: application/json" \
  -d '{
    "clinician_id": "DR_SMITH",
    "patient_id": 1,
    "vitals_assessment": "stable",
    "labs_assessment": "normal",
    "clinical_impression": "yes",
    "risk_assessment": "correct",
    "notes": "Patient is stable"
  }'
```

### Get HITL Feedback List
```bash
curl http://localhost:5000/api/hitl/list?clinician_id=DR_SMITH
```

### Get Patient Summary
```bash
curl http://localhost:5000/api/patient/1/summary
```

### Check Retraining Status
```bash
curl http://localhost:5000/api/hitl/status
```

## 📊 What Each Tab Shows

### Patient Tab
- Patient name, ID, age, gender, ward
- Risk score and color-coded risk level
- Risk badge (High/Moderate/Low)
- Two action buttons

### HITL Tab
- Big statistics box at top
- Retraining button (disabled until 10 forms)
- List of all submitted forms
- Form details: Vitals, Labs, Clinical Impression, Risk Assessment, Notes

## 🎨 UI Features

| Feature | Location |
|---------|----------|
| Role selector | Top of login |
| Patient list | Patients tab |
| HITL form modal | Opens on "Add HITL Form" |
| Statistics panel | Top of HITL tab |
| Retraining button | Below statistics |
| Feedback list | Below retraining button |
| Download option | Patient card (Patients tab) |

## 💡 Tips

- localStorage stores your login → Refresh page stays logged in
- Click logout button (top-right) to clear session
- Forms are stored in memory (resets if app restarts)
- Each clinician sees only their own HITL forms
- Patient summaries include ALL clinician feedback

## ❌ Common Issues

| Issue | Solution |
|-------|----------|
| Modal won't close | Click the × button or outside modal |
| Forms not showing | Make sure you're logged in as clinician |
| Can't retrain | Need exactly 10 forms (check HITL tab status) |
| Download doesn't work | Check browser download settings |
| Logout not working | Clear browser localStorage manually |

## 🔐 Security Notes (Production)

Current implementation uses localStorage (NOT SECURE for production):
- ✅ Fine for: Development, testing, demos
- ❌ Not for: Production, real patient data

### Production Recommendations:
1. Use HTTP-only cookies for sessions
2. Implement JWT token-based auth
3. Hash passwords before storing
4. Use HTTPS for all traffic
5. Implement role-based access control on backend
6. Add database for persistent storage
7. Add audit logs for HITL submissions
