# Clinician Retraining Input Specification

## 🎯 What Clinicians Must Input for Model Retraining

### **Phase 1: Authentication (Login Tab)**

**Required Inputs:**
```
Field               Format              Example         Purpose
─────────────────────────────────────────────────────────────────
Clinician ID        DR_LASTNAME         DR_SMITH        Audit trail
Full Name           Title + Full Name   Dr. John Smith  Record identification
Role                Dropdown selection  Physician       Access control
Department          Free text           ICU             Context tracking
```

**Role Impacts Retraining:**
```
Physician ✅    → Can trigger retraining
Specialist ✅   → Can trigger retraining
Resident ❌     → Cannot trigger
Nurse ❌        → Cannot trigger
```

---

### **Phase 2: Prediction Review (Tab 2: Review Queue)**

**Input Per Prediction:**
```
Input Type          Required        Example             Used For
────────────────────────────────────────────────────────────────
View Patient ID     (required)      PT_12345            Context
View Vitals         (required)      HR=120, Temp=39.2   Clinical context
Review AI Score     (required)      LSTM: 82%, XGB: 78% Comparison
Validation Button   (required)      Click ✓ or ✗        Core feedback
Optional Notes      (optional)      "High WBC, low BP"  Additional context

Clinical Decision:  ONE of:
  ✓ Correct         → "AI prediction matches my assessment"
  ✗ Incorrect       → "AI got it wrong / false positive"
  - Dismiss         → "Can't assess / ambiguous case"
```

**Data Collected Per Review:**
```
{
  "prediction_id": 12345,
  "patient_id": "PT_001",
  "clinician_id": "DR_SMITH",
  "is_correct": true,         ← Your binary input
  "timestamp": "2026-04-15T14:30:00Z",
  "lstm_score": 0.82,
  "xgboost_score": 0.78,
  "ensemble_score": 0.80,
  "features": {...},
  "notes": "..."              ← Optional notes
}
```

**Minimum Dataset Requirements:**
```
Before retraining allowed:
✓ 10+ predictions marked ✓ (correct) or ✗ (incorrect)
✓ 2+ different clinician IDs involved
✓ Dismissals are NOT counted toward the 10
```

---

### **Phase 3: Retraining Trigger (Tab 3: Model Retraining)**

**Clinician Inputs Before Pressing Button:**
```
Checklist Item                              Input Type      Your Action
─────────────────────────────────────────────────────────────────────────
1. Reviewed ≥10 predictions                VIEW only       Review count shown
2. Multiple clinicians reviewed             VERIFY          System checks automatically
3. You have Physician role                  CHECK           Your role displayed
4. Database is consistent                   VERIFY          System checks automatically
```

**When Button is Pressed:**
```
You input:  → (nothing directly - system uses your Clinician ID)

System looks up:
- Your Clinician ID from session
- Verifies you're Physician
- Collects all marked feedback from database
- Prepares training command
- Displays: python retrain_from_feedback.py
```

---

### **Phase 4: Retraining Execution (Terminal)**

**Command Inputs (Optional):**
```
Basic (no parameters):
  python retrain_from_feedback.py

With options:
  python retrain_from_feedback.py --min-reviews 10 --epochs 20

Parameters:
  --min-reviews N    → Minimum reviews collected (default: 10)
  --epochs N         → Training iterations (default: 20)
  --test-split 0.2   → Validation data % (default: 0.2)
  --verbose          → Show detailed progress
```

**What Gets Passed to Retraining Automatically:**
```
From Database:
  ✓ All marked predictions (is_correct = true/false)
  ✓ Original feature vectors from when predictions were made
  ✓ Patient vitals that were used
  ✓ Outcome labels (your approve/reject = training labels)
  
What clinician reviews DO:
  ✓ Create (1) = "Model was correct"
  ✓ Create (0) = "Model was incorrect"
  
Model learns:
  ✓ When features → Sepsis (1)
  ✓ When features → No sepsis (0)
```

---

## 📋 Complete Data Flow for Retraining

```
PHASE 1: LOGIN
  Input: Clinician ID, Name, Role, Department
  ↓
  System: Stores clinician_id in session
  
PHASE 2: REVIEWS (Review Queue Tab)
  Input (multiple times): ✓ or ✗ for each prediction
  ↓
  System: Logs each review to database:
    {
      prediction_id, patient_id, clinician_id,
      is_correct (your binary choice),
      features (serum glucose, WBC, etc),
      lstm_score, xgboost_score
    }
  ↓
  System: Tracks review count and unique clinicians
  
PHASE 3: RETRAINING TAB
  Display: "10 reviewed, 2 clinicians, Ready ✓"
  Input: Click "Start Model Retraining" button
  ↓
  System: Verifies all 4 requirements
  System: Displays terminal command
  
PHASE 4: TERMINAL EXECUTION
  Input: Paste and run command
  
  python retrain_from_feedback.py
  ↓
  System Actions:
    1. Query database: SELECT all reviews with is_correct mark
    2. Extract features and labels
    3. Split into train/test
    4. Train LSTM on marked feedback
    5. Evaluate accuracy
    6. Save new model weights
    7. Report improvement
```

---

## 🔢 Data Specifications

### **Prediction Features Sent to Retraining**

```
Feature Name           Data Type    Range        Example
────────────────────────────────────────────────────────
Serum Glucose         float        (60-500)     145.7
WBC (White Blood)     float        (0.5-50)     12.3
Lactate               float        (0.5-20)     2.1
Heart Rate            int          (40-200)     118
Temperature           float        (35-42)      39.2
Systolic BP           int          (60-200)     92
...18 total features  ...          ...          ...

Label (Your Input):   int          {0, 1}       1 = Correct
```

### **Training Data Preparation**

```
Raw Feedback Collection:
  Total reviews: 45
  Marked correct: 32
  Marked incorrect: 13
  Dismissed: X (not used)

Dataset for training:
  Features: 32 × 18 (for correct predictions)
           + 13 × 18 (for incorrect predictions)
  Labels:   32 × [1] + 13 × [0]
  
Train/Test Split (80/20):
  Training: 36 samples
  Testing: 9 samples
```

---

## ✅ Checklist: What You Must Know

### **Before First Retraining**

```
☐ I know my Clinician ID format (DR_LASTNAME)
☐ I understand my role determines retraining access
☐ I know where Tab 2 (Review Queue) is
☐ I know how to mark predictions ✓ or ✗
☐ I understand I need 10+ reviews minimum
☐ I know retraining requires 2+ clinicians
☐ I know only Physicians can trigger retraining
☐ I can run Python commands in terminal
☐ I know the command: python retrain_from_feedback.py
☐ I understand retraining takes 2-5 minutes
```

---

## 🔀 Role-Based Input Access

| Action | Physician | Specialist | Resident | Nurse |
|--------|-----------|-----------|----------|-------|
| Login | ✅ | ✅ | ✅ | ✅ |
| View Patients | ✅ | ✅ | ✅ | ✅ |
| Review Predictions | ✅ | ✅ | ✅ | ❌ |
| Mark Correct/Incorrect | ✅ | ✅ | ✅ | ❌ |
| **Input Reviews for Training** | ✅ | ✅ | ✅ | ❌ |
| **Trigger Retraining** | ✅ | ✅ | ❌ | ❌ |
| **Receive Retraining Command** | ✅ | ✅ | ❌ | ❌ |

---

## 📊 Database Table: What Your Inputs Create

### **predictions Table** (Updated with your review)
```
Column Name           Data Type      What You Provided
──────────────────────────────────────────────────────
prediction_id         INT PK         System generated
patient_id           VARCHAR        From AI prediction
clinician_id         VARCHAR        YOU: From login
clinician_correct    BOOLEAN        YOU: ✓ = true, ✗ = false
clinician_notes      TEXT           YOU: Optional comments
review_timestamp     DATETIME       System: When you reviewed
xlstm_score          FLOAT          From model
xgboost_score        FLOAT          From model
ensemble_score       FLOAT          From model
risk_level           VARCHAR        CRITICAL/HIGH/...
features_json        JSON           Full feature vector
```

---

## 🎯 Input Timing & Frequency

### **Recommended Schedule**
```
Day 1-3:   Review 3-5 predictions daily = 10+ total
Day 4:     Check retraining status
Day 5:     Physician triggers retraining
Day 6-30:  Monitor improved model
Day 30+:   Next retraining cycle
```

### **Review Time Per Prediction**
```
Average time: 2-3 minutes per prediction
- 1 min: Review vitals and AI scores
- 1 min: Assess clinically
- 0.5 min: Click ✓ or ✗
- 0.5 min: Optional notes
```

---

## 🚀 Final Summary

**What The System Asks From Clinicians:**

1. **Login Inputs** (1 time)
   ```
   Clinician ID, Name, Role, Department
   ```

2. **Review Inputs** (10+ times)
   ```
   ✓ Correct / ✗ Incorrect / - Dismiss
   + Optional notes
   ```

3. **Retraining Inputs** (1 time per cycle)
   ```
   Click button
   Run terminal command
   ```

**What The System Does With Your Inputs:**
```
Review feedback → Store in database
                → Extract features & labels
                → Train LSTM on clinician decisions
                → Improve model accuracy
                → Updated predictions use new model
```

---

**Version**: 1.0  
**Status**: ✅ Ready for Clinical Use  
**Last Updated**: April 15, 2026
