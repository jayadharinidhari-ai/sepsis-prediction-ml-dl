# Unified Dashboard - Quick Start & Deployment

## 🚀 Quick Start (30 seconds)

```bash
# 1. Navigate to project
cd /Users/jayadhariniradhakrishnan/ML-Sepsis

# 2. Start the Flask app
python run.py

# 3. Open in browser
http://localhost:8000/login

# 4. Login and explore
ID: DR_SMITH
Role: Physician
```

---

## 📋 What Changed

### **New Files Created** ✨
```
app/templates/login.html              (800+ lines)  Login page
app/templates/unified_dashboard.html  (1000+ lines) Main interface
INTEGRATED_DASHBOARD_GUIDE.md         (600+ lines)  Complete guide
CLINICIAN_INPUT_SPECIFICATION.md      (500+ lines)  Input details
```

### **Updated Files** 🔄
```
app/routes.py          Added /login and /unified-dashboard routes
app/__init__.py        Enabled session management
```

### **Still Available** (for backwards compatibility)
```
/dashboard        (original monitoring)
/review           (original review queue)
/api/*            (all APIs unchanged)
```

---

## 🔐 Login System Details

### **Authentication Method**
```
Type: Browser-based localStorage + sessionStorage
Security: For development (upgrade for production)
Session Duration: Browser session (logout to clear)
Data Stored: Clinician ID, Name, Role, Department
```

### **Demo Credentials**
```
Clinician ID: DR_SMITH
Full Name: Dr. John Smith
Role: Physician
Department: ICU
```

### **Create New User**
```
Simply login with ANY credentials:
ID: DR_JOHNSON (follows pattern: DR_LASTNAME)
Name: Dr. Jane Johnson
Role: Specialist
Department: Emergency
→ System auto-creates session
```

---

## 📊 Dashboard Tabs

### **Tab 1: Patient Monitoring**
```
Displays:
  - Real-time patient list
  - Vitals (HR, Temp, SBP, O2Sat)
  - AI risk scores
  - Risk level badges (CRITICAL/HIGH/...)
  - Statistics (total patients, critical count, avg score)

Access: All roles (Physician, Specialist, Resident, Nurse)
Auto-refresh: Every 5-10 seconds
```

### **Tab 2: Review Queue**
```
Displays:
  - Pending predictions (HIGH/CRITICAL priority)
  - Patient ID and risk level
  - AI scores (LSTM, XGBoost, Ensemble)
  - Action buttons:
    ✓ Correct - Mark model correct
    ✗ Incorrect - Mark model wrong
    - Dismiss - Skip this prediction

  - Statistics:
    Pending Reviews count
    Total Reviewed count
    Model Accuracy %

Access: Physician, Specialist, Resident (NOT Nurse)
Actions: Stored in database with clinician_id
```

### **Tab 3: Model Retraining**
```
Displays:
  - System requirements (4 items to check)
  - Pre-retraining checklist (5 items)
  - System status metrics
  - Retraining instructions
  - "Start Model Retraining" button

Access: Physician and Specialist ONLY
Button enabled when:
  ✓ 10+ reviews collected
  ✓ 2+ clinicians reviewed
  ✓ User is Physician or Specialist
  ✓ Database is intact
```

---

## 🔗 URL Routes

```
/login                      → Login page
/unified-dashboard          → Main dashboard (3 tabs)
/dashboard                  → Legacy monitoring only
/review                     → Legacy review queue only
/api/patients              → Get all patients
/api/human-loop/review-queue      → Get pending reviews
/api/human-loop/approve    → Mark prediction correct/incorrect
/api/human-loop/statistics → Get accuracy metrics
```

---

## 📱 Role-Based Access Control

```
╔═══════════╦═══════════╦════════════╦═════════╦═══════╗
║ Feature   ║ Physician ║ Specialist ║ Resident║ Nurse ║
╠═══════════╬═══════════╬════════════╬═════════╬═══════╣
║ Monitor   ║ ✅        ║ ✅         ║ ✅      ║ ✅    ║
║ Review    ║ ✅        ║ ✅         ║ ✅      ║ ❌    ║
║ Retrain   ║ ✅        ║ ✅         ║ ❌      ║ ❌    ║
║ Stats     ║ ✅        ║ ✅         ║ ✅      ║ ❌    ║
╚═══════════╩═══════════╩════════════╩═════════╩═══════╝
```

---

## 🤖 Retraining Workflow

### **Step 1: Review Predictions** (Tab 2)
```
Click on prediction → Review clinically → Click ✓ or ✗
System logs: {prediction_id, clinician_id, is_correct}
Need: 10+ reviews, 2+ clinicians
```

### **Step 2: Check Retraining Tab** (Tab 3)
```
Requirements:
  ✓ 10+ reviewed predictions
  ✓ 2+ clinicians reviewed
  ✓ You're logged in as Physician
  ✓ Database is consistent

Button status:
  ENABLED ✅ → Ready to retrain
  DISABLED ❌ → See message for what's needed
```

### **Step 3: Trigger Retraining**
```
Click "🤖 Start Model Retraining" button
→ See command: python retrain_from_feedback.py
→ Copy command
```

### **Step 4: Run Retraining**
```bash
# In terminal
cd /Users/jayadhariniradhakrishnan/ML-Sepsis
python retrain_from_feedback.py --min-reviews 10 --epochs 20

# Output:
# 📊 Training on 25 approved + 5 rejected predictions
# 🚀 Training LSTM model...
# [progress bars...]
# ✅ RETRAINING COMPLETE
```

### **Step 5: Verify Improvement**
```
Model automatically reloaded
Next predictions use improved model
Check Tab 2: Accuracy should increase
```

---

## 🔧 Configuration

### **Session Configuration** (app/__init__.py)
```python
app.config["SECRET_KEY"] = "sepsis-demo-secret-key-change-in-production"
app.config["SESSION_TYPE"] = "filesystem"
app.config["SESSION_PERMANENT"] = False
```

### **Retraining Parameters** (retrain_from_feedback.py)
```python
--min-reviews 10      # Minimum reviews to trigger
--epochs 20          # Training iterations
--test-split 0.2     # 80% train, 20% test
--verbose            # Show detailed progress
```

### **Database** (app/services/human_loop_manager.py)
```python
DB_PATH = "human_feedback.db"  # Created automatically
Tables:
  - predictions (stores reviews and feedback)
  - review_queue (tracks pending reviews)
  - feedback_summary (aggregated stats)
```

---

## 🧪 Testing the System

### **Test Login**
```
URL: http://localhost:8000/login
Test credentials (try any):

1. Physician:
   ID: DR_SMITH
   Name: Dr. John Smith
   Role: Physician
   Dept: ICU
   
2. Specialist:
   ID: DR_JOHNSON  
   Name: Dr. Jane Johnson
   Role: Specialist
   Dept: Emergency
   
3. Resident:
   ID: DR_BROWN
   Name: Dr. Bob Brown
   Role: Resident
   Dept: Internal Med
```

### **Test Review Workflow**
```
1. Login as Physician
2. Go to Tab 2 (Review Queue)
3. Click ✓ on 15 predictions
4. Go to Tab 3 (Retraining)
5. Button should now be ENABLED
6. Click "Start Model Retraining"
7. Run command shown
```

### **Test Role Access**
```
1. Login as Resident
2. Tab 1: ✅ Can see patients
3. Tab 2: ✅ Can review
4. Tab 3: ❌ Button disabled - "Retraining requires Physician access"
```

---

## 🐛 Troubleshooting

### **Problem: Login Page Showing But Can't Submit**
```
Solution:
1. Check browser console (F12)
2. Verify Clinician ID format: DR_LASTNAME
3. Check all fields are filled
4. Refresh page and try again
```

### **Problem: Can't Review Predictions in Tab 2**
```
Solution:
1. Make sure you logged in as Physician/Specialist
2. Check that patients exist (Tab 1 should show some)
3. Wait 5 seconds for data to load
4. Check browser console for errors
```

### **Problem: Retraining Button Disabled**
```
Possible causes:
❌ Not logged in as Physician - Try Physician role
❌ Less than 10 reviews - Review more predictions
❌ Only 1 clinician reviewed - Have another clinician review
❌ Database error - Check human_feedback.db integrity

Solution: Check Tab 3 for specific status message
```

### **Problem: Retraining Command Not Working**
```
Solution:
1. Verify directory: cd /Users/jayadhariniradhakrishnan/ML-Sepsis
2. Check Python path: which python
3. Run with full path if needed:
   /usr/local/bin/python3 retrain_from_feedback.py
4. Verify requirements installed: pip install -r requirements.txt
```

### **Problem: No Patients Showing in Tab 1**
```
Solution:
1. Wait 10-15 seconds for simulator to generate patients
2. Refresh page (F5)
3. Check app logs for simulator errors
4. Verify PatientSimulator is running
```

---

## 📦 Deployment Checklist

### **Before Production**
```
☐ Change app/__init__.py SECRET_KEY to secure value
☐ Enable HTTPS (use Flask-HTTPS or reverse proxy)
☐ Implement proper user database (not localStorage)
☐ Add password hashing (bcrypt/werkzeug.security)
☐ Enable CSRF protection (Flask-SeaSurf)
☐ Set up database backups for human_feedback.db
☐ Log all clinician actions for audit trail
☐ Implement role-based authorization at API level
☐ Test with real patient data
☐ Get regulatory approval (HIPAA, FDA if needed)
☐ Set up error logging and monitoring
☐ Document data retention policies
```

### **Production Modifications Needed**
```
File: app/__init__.py
Change: app.config["SECRET_KEY"] = os.environ.get('SECRET_KEY')

File: app/routes.py
Add: Proper middleware authentication checking

Database: Use PostgreSQL instead of SQLite
Backup: Daily backup schedule for human_feedback.db
Logging: Store all clinician actions with timestamps
```

---

## 🔐 Session Management

### **How Sessions Work**
```javascript
// Client side (browser)
localStorage.setItem('clinician', JSON.stringify({
  id: "DR_SMITH",
  name: "Dr. John Smith",
  role: "physician",
  department: "ICU",
  loginTime: "2026-04-15T10:30:00Z"
}));

// Checked on each page load
// Redirects to /login if not present
```

### **Logging Out**
```
Click "🔓 Logout" button
→ Clears localStorage
→ Clears sessionStorage
→ Redirects to /login
```

---

## 📊 Database Schema

### **Table: predictions**
```sql
CREATE TABLE predictions (
  prediction_id INTEGER PRIMARY KEY,
  patient_id VARCHAR,
  clinician_id VARCHAR,        -- Your ID from login
  lstm_score FLOAT,
  xgboost_score FLOAT,
  ensemble_score FLOAT,
  risk_level VARCHAR,
  clinician_correct BOOLEAN,   -- Your ✓/✗ feedback
  clinician_notes TEXT,
  created_at DATETIME,
  reviewed_at DATETIME
);
```

### **Table: review_queue**
```sql
CREATE TABLE review_queue (
  review_id INTEGER PRIMARY KEY,
  prediction_id INTEGER,
  patient_id VARCHAR,
  status VARCHAR DEFAULT 'pending',  -- pending/reviewed/dismissed
  priority INTEGER,
  created_at DATETIME
);
```

---

## 🎓 For Clinicians

### **Three Things You Need to Know**

**1. How to Login**
```
URL: http://localhost:8000/login
Fields: ID (DR_SMITH), Name, Role (Physician), Department (ICU)
```

**2. How to Review**
```
Tab 2: Review Queue
Per prediction: Click ✓ if correct, ✗ if wrong
Need 10+ reviews minimum
```

**3. How to Retrain**
```
Tab 3: Check if requirements met (usually: 10 reviews + Physician)
Click "🤖 Start Model Retraining"
Run: python retrain_from_feedback.py
Wait 2-5 minutes for completion
```

---

## 📚 Documentation Files

```
INTEGRATED_DASHBOARD_GUIDE.md        ← Complete user guide
CLINICIAN_INPUT_SPECIFICATION.md     ← What to input for retraining
HUMAN_IN_THE_LOOP_GUIDE.md          ← Detailed HITL concepts
HUMAN_IN_THE_LOOP_SUMMARY.md        ← Technical implementation
```

---

## 🚀 Deployment Command

```bash
# Start the application
python run.py

# OR with environment variables
FLASK_ENV=production python run.py

# Access
Browser: http://localhost:8000/login
```

---

**Version**: 1.0  
**Status**: ✅ Ready  
**Last Updated**: April 15, 2026
