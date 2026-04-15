# Human-in-the-Loop (HITL) Implementation Guide

## Overview

This guide covers the **Human-in-the-Loop (HITL)** system integrated into the ML-Sepsis application. HITL enables clinicians to validate AI predictions, provide feedback, and continuously improve the model through validated real-world data.

## 🎯 Key Features

### 1. **Automated Review Queue**
- All HIGH and CRITICAL risk predictions automatically added to review queue
- Prioritized by risk level (CRITICAL > HIGH)
- Chronologically ordered for FIFO processing

### 2. **Clinician Feedback**
- Approve/Reject predictions (model correct or incorrect)
- Add optional notes for context
- Track clinician identity for audit trail

### 3. **Accuracy Tracking**
- Real-time model accuracy based on clinician validations
- Review completion percentage
- Correct vs. Incorrect breakdown

### 4. **Model Retraining**
- Use clinician feedback to retrain LSTM model
- Automatic data extraction from database
- Improve model performance on real-world data

### 5. **Audit Trail**
- All predictions stored with timestamps
- Clinician identities recorded
- Complete feedback history per patient

---

## 🏗️ Architecture

### Database Schema

```
predictions table:
├── id (PK)
├── patient_id
├── timestamp
├── features (JSON)
├── lstm_score
├── xgb_score
├── ensemble_score
├── model_type
├── risk_level
├── reviewed (0/1)
├── review_timestamp
├── clinician_id
├── clinician_correct (0/1)
└── notes

review_queue table:
├── id (PK)
├── prediction_id (FK)
├── patient_id
├── timestamp
├── risk_score
├── risk_level
├── priority
├── assigned_to
└── status (pending/reviewed/dismissed)
```

### Data Flow

```
Patient Data
    ↓
Sepsis Engine Prediction
    ↓
Human Loop Manager
    ├─> Store in predictions table
    └─> Add HIGH/CRITICAL to review_queue
    ↓
Clinician Review Dashboard
    ├─> Display pending reviews
    ├─> Collect feedback
    └─> Update review_queue status
    ↓
Feedback Analysis
    ├─> Calculate accuracy metrics
    ├─> Export training data
    └─> Trigger retraining
    ↓
Model Retraining
    ├─> Use validated feedback labels
    ├─> Retrain LSTM with new data
    └─> Deploy improved model
```

---

## 🚀 Usage

### Part 1: Automatic Prediction Tracking

When the Flask app runs, all predictions are automatically added to the HITL system:

```python
# In sepsis_engine.py predict() method:
hlm = get_human_loop_manager()
prediction_id = hlm.add_prediction(
    patient_id=patient_id,
    features=features_dict,
    lstm_score=lstm_score,
    xgb_score=xgb_score,
    ensemble_score=ensemble_score,
    model_type=model_info,
    risk_level=response["risk_level"]
)
```

### Part 2: Clinical Review

#### Access the Review Dashboard

```bash
# Start the Flask app
python run.py

# Open browser to review queue
http://localhost:8000/review
```

#### Review Interface Features:

1. **Statistics Panel** - Shows:
   - Total predictions collected
   - Number reviewed
   - Current model accuracy
   - Correct vs. Incorrect counts

2. **Review Queue** - Displays:
   - Patient ID and timestamp
   - Risk level (CRITICAL/HIGH/MODERATE/LOW)
   - LSTM, XGBoost, and Ensemble scores
   - Key vital signs and lab values
   - Model type used for prediction

3. **Action Buttons**:
   - ✓ **Model Correct** - Clinician validates prediction
   - ✗ **Model Incorrect** - Clinician rejects prediction
   - **Dismiss** - Skip without feedback

#### Example Clinician Workflow:

```
1. Clinician logs into http://localhost:8000/review
2. Views pending predictions (sorted by risk)
3. For each prediction:
   a. Reviews AI scores and vital signs
   b. Compares with clinical judgment
   c. Clicks "Model Correct" or "Model Incorrect"
   d. Optionally adds notes
   e. Enters clinician ID for audit trail
4. System updates accuracy metrics
5. Dashboard refreshes automatically
```

### Part 3: Data Export for Analysis

Export collected feedback for external analysis or tool integration:

```bash
# API endpoint to get feedback data
curl http://localhost:8000/api/human-loop/export-feedback

# Returns:
{
  "success": true,
  "sample_count": 45,
  "features": [...],
  "labels": [1, 0, 1, 1, 0, ...]
}
```

### Part 4: Model Retraining

#### Manual Retraining Process

```bash
# Check current feedback statistics
curl http://localhost:8000/api/human-loop/statistics

# Retrain model from feedback
python retrain_from_feedback.py --min-reviews 10 --epochs 20

# Output:
# 📊 Feedback Statistics:
#    Total Predictions: 150
#    Total Reviewed: 45
#    Model Accuracy: 82.5%
# 
# 🚀 Training LSTM model...
# ... [training progress] ...
# 
# ✅ RETRAINING COMPLETE
# Training Accuracy: 87.3%
```

#### Retraining Parameters

```bash
python retrain_from_feedback.py \
  --min-reviews 10      # Minimum reviewed predictions needed
  --epochs 20           # Training epochs
```

---

## 📊 API Endpoints

### Get Review Queue

```http
GET /api/human-loop/review-queue?status=pending&limit=10

Response:
{
  "success": true,
  "count": 5,
  "queue": [
    {
      "id": 1,
      "prediction_id": 123,
      "patient_id": "P001",
      "timestamp": "2026-04-15T10:30:00",
      "risk_score": 0.85,
      "risk_level": "HIGH",
      "lstm_score": 0.82,
      "xgb_score": 0.88,
      "ensemble_score": 0.85,
      "features": {...}
    }
  ]
}
```

### Approve Prediction

```http
POST /api/human-loop/approve

Body:
{
  "prediction_id": 123,
  "clinician_id": "DR_SMITH",
  "is_correct": true,
  "notes": "Correctly identified infection"
}

Response:
{
  "success": true,
  "message": "Prediction approved by DR_SMITH",
  "feedback": "correct",
  "statistics": {
    "total_predictions": 150,
    "total_reviewed": 45,
    "review_percentage": 30.0,
    "accuracy": 82.5,
    "correct": 37,
    "incorrect": 8
  }
}
```

### Dismiss Review

```http
POST /api/human-loop/dismiss

Body:
{
  "prediction_id": 123,
  "clinician_id": "DR_SMITH"
}

Response:
{
  "success": true,
  "message": "Review dismissed"
}
```

### Get Statistics

```http
GET /api/human-loop/statistics

Response:
{
  "success": true,
  "statistics": {
    "total_predictions": 150,
    "total_reviewed": 45,
    "review_percentage": 30.0,
    "accuracy": 82.5,
    "correct": 37,
    "incorrect": 8
  }
}
```

### Get Patient History

```http
GET /api/human-loop/history/<patient_id>?limit=20

Response:
{
  "success": true,
  "patient_id": "P001",
  "predictions": [...]
}
```

### Export Feedback

```http
GET /api/human-loop/export-feedback?reviewed_only=true

Response:
{
  "success": true,
  "sample_count": 45,
  "features": [...],
  "labels": [1, 0, 1, 1, 0, ...]
}
```

---

## 📈 Performance Monitoring

### Key Metrics

1. **Review Coverage**: Percentage of predictions reviewed
2. **Model Accuracy**: Based on clinician validations
3. **Clinician Agreement Rate**: How often clinicians agree with model
4. **Feedback Trends**: Patterns in model errors

### Example Monitoring Query

```python
from app.services.human_loop_manager import HumanLoopManager

hlm = HumanLoopManager()
stats = hlm.get_feedback_statistics()

print(f"Accuracy: {stats['accuracy']:.1f}%")
print(f"Reviews: {stats['total_reviewed']}/{stats['total_predictions']}")
print(f"Correct: {stats['correct']}, Incorrect: {stats['incorrect']}")
```

---

## 🔄 Continuous Improvement Loop

### 1. **Collect Predictions**
Every high-risk prediction automatically enters review queue

### 2. **Clinician Review**
Clinicians validate predictions through dashboard

### 3. **Accumulate Feedback**
Feedback stored in database with clinician annotations

### 4. **Analyze Performance**
Track model accuracy based on validation data

### 5. **Retrain Periodically**
Run retraining when sufficient feedback accumulated (10-50 samples)

### 6. **Deploy Improved Model**
New model weights replace old ones in production

### 7. **Monitor Improvement**
Track accuracy metrics before/after retraining

---

## 🛠️ Customization

### Adjusting Review Queue Priority

```python
# In human_loop_manager.py - modify add_prediction():
if risk_level in ["HIGH", "CRITICAL"]:
    priority = 3 if risk_level == "CRITICAL" else 2  # Adjust values
```

### Changing Risk Level Thresholds

```python
# In sepsis_engine.py - modify predict():
if response["risk_level"] in ["HIGH", "CRITICAL"]:
    response["review_required"] = True
```

### Custom Feedback Fields

Extend the predictions table schema:

```python
# In human_loop_manager.py - modify _init_database():
cursor.execute("""
ALTER TABLE predictions ADD COLUMN custom_field TEXT
""")
```

---

## 🔒 Security Considerations

1. **Audit Trail**: All clinician actions logged with timestamp and ID
2. **Data Privacy**: Patient data encrypted at rest (configure for production)
3. **Access Control**: Implement authentication for /review endpoint
4. **Clinician Verification**: Require clinician ID for feedback submission

### Production Security Checklist

- [ ] Enable database encryption
- [ ] Add Flask authentication to /review endpoint
- [ ] Implement clinician role-based access control
- [ ] Enable HTTPS for all connections
- [ ] Set up backup and recovery procedures
- [ ] Configure audit logging to external system
- [ ] Implement rate limiting on API endpoints

---

## ❓ Troubleshooting

### Issue: Review Queue Shows "All Caught Up"

**Cause**: High-risk predictions in current session not triggering review queue

**Solution**: 
- Risk level calculation might be too lenient
- Adjust threshold in sepsis_engine.py
- Check that predictions are marked as HIGH/CRITICAL

### Issue: Retraining Fails with "Insufficient Reviews"

**Cause**: Not enough clinician feedback collected

**Solution**:
```bash
# Check current feedback count
curl http://localhost:8000/api/human-loop/statistics

# Collect more feedback before retraining
python retrain_from_feedback.py --min-reviews 10
```

### Issue: Database File Not Found

**Cause**: First time initialization

**Solution**:
```python
# Manually initialize database
from app.services.human_loop_manager import HumanLoopManager
hlm = HumanLoopManager()
# Creates human_feedback.db automatically
```

---

## 📚 Related Documentation

- [Main README](README.md) - System overview
- [LSTM Integration Guide](LSTM_INTEGRATION_GUIDE.md) - Deep learning details
- [Quick Start](QUICK_START.md) - Getting started
- [API Reference](REAL_TIME_GUIDE.md) - API endpoints

---

## 📝 Example: End-to-End Workflow

```bash
# 1. Start the application
python run.py
# Output: Flask running on http://localhost:8000

# 2. Predictions automatically made (check terminal)
# Output showing LSTM and ensemble scores

# 3. Access review dashboard
# Open http://localhost:8000/review in browser

# 4. Review predictions
# Click "Model Correct" or "Model Incorrect"

# 5. Check statistics
curl http://localhost:8000/api/human-loop/statistics

# 6. When enough feedback collected, retrain
python retrain_from_feedback.py

# 7. Verify improved accuracy
# Dashboard will show updated metrics
```

---

**Last Updated**: April 15, 2026  
**System**: ML-Sepsis with Human-in-the-Loop  
**Status**: Production Ready
