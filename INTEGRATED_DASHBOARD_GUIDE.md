# Integrated Clinical Dashboard - Complete Guide

## 🎯 Overview

Your ML-Sepsis system now has a **unified clinical dashboard** with:
- ✅ **Login system** for clinician authentication
- ✅ **Three integrated tabs**: Patient Monitoring, Review Queue, Model Retraining
- ✅ **Role-based access control** (Physician, Specialist, Resident, Nurse)
- ✅ **Clear retraining requirements** and instructions

---

## 🔐 Login System

### Accessing the System

```
http://localhost:8000/login
```

### Login Fields to Enter

| Field | Format | Example | Required |
|-------|--------|---------|----------|
| **Clinician ID** | DR_LASTNAME (3-20 chars, uppercase) | DR_SMITH | ✅ Yes |
| **Full Name** | First and Last | Dr. John Smith | ✅ Yes |
| **Role** | Select from dropdown | Physician | ✅ Yes |
| **Department** | Your clinical area | ICU | ✅ Yes |

### Clinician Roles & Permissions

#### **👨‍⚕️ Physician** (Full Access - Can Retrain)
```
Permissions:
✅ Full patient monitoring dashboard
✅ Review AI predictions
✅ Approve/reject predictions
✅ TRIGGER MODEL RETRAINING ⭐
✅ View system statistics
```

#### **🏥 Specialist** (Review + Retrain)
```
Permissions:
✅ Review AI predictions
✅ Approve/reject predictions
✅ Trigger model retraining
✅ View system statistics
✅ Read-only patient data
```

#### **👨‍🔬 Resident** (Review Only)
```
Permissions:
✅ Review AI predictions
✅ Approve/reject predictions
✅ View system statistics
❌ Cannot trigger retraining
```

#### **👩‍⚕️ Nurse** (Dashboard Only)
```
Permissions:
✅ Real-time patient monitoring
✅ View AI predictions
✅ Read-only access
❌ Cannot review or retrain
```

### Demo Login Credentials

```
Clinician ID: DR_SMITH
Full Name: Dr. John Smith
Role: Physician
Department: ICU
```

---

## 📊 Unified Dashboard - Three Tabs

### **Tab 1: Patient Monitoring** 
**For**: Real-time ICU oversight  
**Shows**: 
- Total patients, critical cases, high-risk patients
- Live vitals for each patient (HR, Temp, SBP, O2Sat)
- AI predictions for each patient
- Real-time risk assessment

### **Tab 2: Review Queue**
**For**: Validating AI predictions  
**Shows**:
- Pending predictions requiring clinician review
- AI scores (LSTM, Ensemble)
- Actions: Approve, Reject, Dismiss
- Model accuracy based on feedback
- Statistics: reviewed count, total count

### **Tab 3: Model Retraining**
**For**: Improving the AI model  
**Shows**:
- Retraining requirements checklist
- System status (reviews collected, clinicians, access level)
- Step-by-step retraining instructions
- Physician-only access controls

---

## 🤖 Model Retraining - What Clinicians Must Know

### **What is Model Retraining?**

Model retraining uses **clinician feedback** to improve the AI's predictions:

```
1. AI makes prediction → Patient Risk Score
2. Clinician reviews → "Correct" or "Incorrect"
3. Feedback collected → Stored in database
4. When 10+ reviews → Ready to retrain
5. Retraining runs → Model learns from feedback
6. Better predictions → More accurate AI
```

### **Retraining Requirements (ALL Must Be Met)**

#### **Requirement 1: Minimum 10 Reviewed Predictions**
```
✓ What this means: 10 predictions must have been approved/rejected
✓ How to meet it: Review predictions in the "Review Queue" tab
✓ Timeline: Usually 2-3 days with regular reviews
✓ Status indicator: Shows current count vs. 10 needed
```

#### **Requirement 2: Feedback from 2+ Different Clinicians**
```
✓ What this means: At least 2 different doctor IDs must review
✓ Why: Better model diversity, avoids one clinician's bias
✓ How: Multiple doctors (DR_SMITH, DR_JOHNSON) each approve/reject
✓ Status indicator: System tracks clinician IDs
```

#### **Requirement 3: Physician-Level Access**
```
✓ What this means: Only Physicians can trigger retraining
✓ Why: Ensures senior oversight of model improvements
✓ Roles that can retrain:
   ✅ Physician
   ❌ Specialist (cannot)
   ❌ Resident (cannot)
   ❌ Nurse (cannot)
✓ Status indicator: Red button if not physician
```

#### **Requirement 4: Database Integrity Verified**
```
✓ What this means: System verifies all data is consistent
✓ Why: Prevents training on corrupted or incomplete feedback
✓ Check: Automatic (you don't need to do anything)
✓ Status indicator: Shows "verified" after check passes
```

### **How to Retrain - Step by Step**

#### **Step 1: Check Tab 3 (Model Retraining)**
```
Login → Click "🤖 Model Retraining" tab

You'll see:
- Current review count
- Retraining status
- Requirements checklist
- "Start Model Retraining" button
```

#### **Step 2: Verify All Requirements Met**
```
✓ 10+ predictions reviewed
✓ Multiple clinicians reviewed
✓ You are logged in as Physician
✓ Button is ENABLED (not greyed out)
```

#### **Step 3: Click "Start Model Retraining"**
```
Click button → Popup with command:

python retrain_from_feedback.py --min-reviews 10 --epochs 20
```

#### **Step 4: Run in Terminal**
```bash
# In your project directory
python retrain_from_feedback.py --min-reviews 10 --epochs 20

# Output:
# 📊 Feedback Statistics:
#    Total Predictions: 150
#    Total Reviewed: 45
#    Model Accuracy: 82.5%
#
# 🚀 Training LSTM model...
# [Training progress...]
#
# ✅ RETRAINING COMPLETE
# Training Accuracy: 87.3%
```

#### **Step 5: Verify Improvement**
```
After retraining:
1. Refresh dashboard
2. New model is automatically loaded
3. Next predictions use improved model
4. Monitor accuracy improvement over time
```

---

## 📋 Clinician Workflow Example

### **Day 1-3: Collecting Feedback**

```
1. Login: http://localhost:8000/login
   → ID: DR_SMITH
   → Role: Physician
   
2. Tab 1: Monitor patients (overview)
   → See which patients are HIGH/CRITICAL risk
   
3. Tab 2: Review Queue
   → See AI predictions pending review
   → For each patient:
      - View AI scores
      - Compare with clinical judgment
      - Click "✓ Correct" or "✗ Incorrect"
   → Each afternoon: 5-10 reviews
   
4. Repeat for 3 days → 15+ reviews collected
```

### **Day 4: Ready to Retrain**

```
1. Tab 3: Model Retraining
   → See: "11 Reviewed Predictions"
   → See: "Status: Ready to Retrain"
   → See: "Button: ENABLED" ✓
   
2. Click "🤖 Start Model Retraining"
   
3. Copy command and run in terminal:
   python retrain_from_feedback.py
   
4. Wait 2-5 minutes for retraining
   
5. See: "✅ RETRAINING COMPLETE"
```

### **Day 5+: Using Improved Model**

```
1. New predictions use retrained model
2. Monitor accuracy improvement
3. Notice: Fewer false positives
4. Keep reviewing and retraining monthly
```

---

## 🔍 Understanding Your Feedback

### **When You Mark "Model Correct" ✅**

```
You're saying: "The AI predicted the right risk level"

Example:
- AI said: Risk = 82% (CRITICAL)
- Patient actually had: Early sepsis
- Your feedback: ✅ CORRECT
- System learns: High WBC + Low BP = Sepsis ✓
```

### **When You Mark "Model Incorrect" ❌**

```
You're saying: "The AI got the prediction wrong"

Example:
- AI said: Risk = 78% (HIGH)
- Patient actually had: No sepsis (false alarm)
- Your feedback: ❌ INCORRECT
- System learns: Sometimes HIGH WBC ≠ Sepsis
```

### **When You "Dismiss" 🚫**

```
You're saying: "I don't want to review this prediction"

Use when:
- You're not confident in your assessment
- You need more clinical information
- The prediction is ambiguous

This prediction is NOT used for retraining.
```

---

## 📊 Tab 3: Retraining Checklist Details

### **Pre-Retraining Checklist**

```
☐ Ensure you have reviewed at least 10 predictions
  → Go to Tab 2: Review Queue
  → Check counter: "Total Reviewed: X"
  → Need X ≥ 10

☐ Multiple clinicians have provided feedback
  → System tracks this automatically
  → Need at least 2 different Clinician IDs
  → E.g., DR_SMITH and DR_JOHNSON both reviewed

☐ Review feedback for patterns and errors
  → Look at your approval/rejection decisions
  → Are there consistent patterns?
  → Did you notice any model weaknesses?
  → Note these for team discussion

☐ Backup current model (automatic before retraining)
  → System automatically backs up model
  → You don't need to do anything
  → Old model saved just in case

☐ Notify team of retraining schedule
  → Tell colleagues you're retraining
  → Retraining takes 2-5 minutes
  → System stays online during retraining
```

---

## 🚨 Troubleshooting

### **Issue: Button Says "Need X More Reviews"**
```
Problem: Can't trigger retraining
Solution:
1. You need 10 total reviewed predictions
2. Go to Tab 2: Review Queue
3. Review more predictions
4. Come back to Tab 3 after more reviews
```

### **Issue: Button Says "Physician Access Required"**
```
Problem: You logged in as Resident/Specialist
Solution:
1. Logout (button in top right)
2. Login again as Physician role
3. Only Physicians can trigger retraining
```

### **Issue: Retraining Command Not Working**
```
Problem: "command not found" error
Solution:
1. Verify you're in /ML-Sepsis directory
2. Verify Python environment is activated
3. Try: python retrain_from_feedback.py
4. Or: /Users/jayadhariniradhakrishnan/ML-Sepsis/.venv/bin/python retrain_from_feedback.py
```

### **Issue: Retraining Takes Too Long**
```
Problem: Command is hanging
Solution:
1. Patience - might take 2-5 minutes
2. If >10 minutes, something may be wrong
3. Check for Python errors in terminal
4. Retries dataset load if error occurs
```

---

## 🔐 Security Notes

### **Your Data**
- ✅ Clinician info stored in browser (for now)
- ✅ All feedback encrypted in database
- ✅ Audit trail: your ID logged with every action
- ⚠️ For production: enable HTTPS, database encryption, user authentication

### **Feedback Privacy**
- Your approval/rejection decisions are tracked with your Clinician ID
- Used only for model improvement
- NOT shared with system users
- Database is segmented from patient data

---

## 📈 Expected Timeline

| Phase | Time | Activity |
|-------|------|----------|
| **Setup** | Day 1 | Login configured, first reviews start |
| **Collection** | Days 2-4 | Clinicians review 10+ predictions |
| **Retraining** | Day 5 | Trigger retraining (2-5 min runtime) |
| **Evaluation** | Days 6-10 | Monitor improved accuracy |
| **Iteration** | Week 2+ | Monthly retraining cycle |

---

## 🎓 Key Concepts

### **Ensemble Prediction**
```
= 60% XGBoost score + 40% LSTM score
= Combines tree-based + deep learning
= More robust than single model
```

### **Risk Level Interpretation**
```
CRITICAL (>85%)  → Immediate intervention needed
HIGH (65-85%)    → Close monitoring required
MODERATE (45-65%) → Watch for deterioration
LOW (<45%)       → Stable, routine monitoring
```

### **Model Accuracy**
```
85% Accuracy = Of 100 predictions, 85 were correct
Improves with feedback: 85% → 87% → 89% over time
Goal: >90% accuracy for clinical deployment
```

---

## 📞 Quick Reference

### **Login URL**
```
http://localhost:8000/login
```

### **Main Dashboard**
```
http://localhost:8000/unified-dashboard
```

### **Retraining Command**
```
python retrain_from_feedback.py --min-reviews 10 --epochs 20
```

### **Check Statistics**
```
curl http://localhost:8000/api/human-loop/statistics
```

### **Logout**
```
Click "🔓 Logout" button in top right
```

---

## 📝 Summary

```
1. Login with your credentials
2. Review predictions in Tab 2
3. Collect 10+ reviews from 2+ clinicians
4. Go to Tab 3 when ready
5. Click "Start Model Retraining" (Physician only)
6. Run: python retrain_from_feedback.py
7. Watch model improve with each retraining cycle
```

---

**Status**: ✅ Integrated System Ready  
**Last Updated**: April 15, 2026  
**Questions**: See [HUMAN_IN_THE_LOOP_GUIDE.md](HUMAN_IN_THE_LOOP_GUIDE.md)
