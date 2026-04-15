# Human-in-the-Loop Quick Reference

## 🚀 Quick Start (2 minutes)

### 1. **Launch Application**
```bash
python run.py
# Flask starts on http://localhost:8000
```

### 2. **Access Review Dashboard**
```
http://localhost:8000/review
```

### 3. **Review and Validate Predictions**
- See pending HIGH/CRITICAL risk predictions
- Click ✓ Model Correct or ✗ Model Incorrect
- Enter your clinician ID
- System updates accuracy

### 4. **Check Performance**
Dashboard automatically shows:
- Total predictions collected
- Predictions reviewed
- Current model accuracy %
- Correct vs. Incorrect counts

---

## 📱 Dashboard Commands

| Action | Description |
|--------|-------------|
| **✓ Model Correct** | Clinician agrees with AI prediction |
| **✗ Model Incorrect** | Clinician disagrees with AI |
| **Dismiss** | Skip review without feedback |
| **Refresh** | Auto-refreshes every 30 seconds |

---

## 🔧 Retraining (When Ready)

### Check Feedback Status
```bash
curl http://localhost:8000/api/human-loop/statistics
```

### Retrain Model
```bash
# Requires minimum 10 reviewed predictions
python retrain_from_feedback.py

# With custom parameters
python retrain_from_feedback.py --min-reviews 10 --epochs 20
```

### Monitor During Retraining
```
📊 Feedback Statistics:
   Total Predictions: 150
   Total Reviewed: 45
   Model Accuracy: 82.5%

🚀 Training complete
Training Accuracy: 87.3%
```

---

## 📊 Key Metrics

```
Review Coverage = (Reviewed / Total) × 100%
Model Accuracy = (Correct / Reviewed) × 100%
```

**Example**:
- Total Predictions: 150
- Reviewed: 45 (30% coverage)
- Correct: 37 (82.2% accuracy)

---

## 🔗 API Cheat Sheet

### Get Review Queue
```bash
curl http://localhost:8000/api/human-loop/review-queue
```

### Submit Review
```bash
curl -X POST http://localhost:8000/api/human-loop/approve \
  -H "Content-Type: application/json" \
  -d '{
    "prediction_id": 123,
    "clinician_id": "DR_SMITH",
    "is_correct": true,
    "notes": "Clinical notes here"
  }'
```

### Get Statistics
```bash
curl http://localhost:8000/api/human-loop/statistics
```

### Export Feedback Data
```bash
curl http://localhost:8000/api/human-loop/export-feedback
```

---

## 🎯 Workflow at a Glance

```
1. Predictions Made
   ↓
2. HIGH/CRITICAL → Review Queue
   ↓
3. Clinician Reviews
   ↓
4. Feedback Submitted
   ↓
5. Accuracy Updated
   ↓
6. Enough Feedback? → Retrain
   ↓
7. Model Improved
```

---

## ⚠️ Common Issues

| Issue | Solution |
|-------|----------|
| "All Caught Up" | No pending reviews - wait for new high-risk predictions |
| Retraining fails | Need 10+ reviewed predictions: `python retrain_from_feedback.py` |
| Database not found | System creates it automatically on first run |
| Dashboard not loading | Ensure Flask running: `python run.py` |

---

## 📂 File Locations

```
/human_feedback.db          ← Review database (auto-created)
/app/services/human_loop_manager.py    ← Core HITL logic
/app/templates/review_queue.html       ← Dashboard UI
/app/routes_human_loop.py   ← API endpoints
/retrain_from_feedback.py   ← Retraining script
```

---

## 🔐 Audit Trail

Every action recorded:
- **Clinician ID**: Who reviewed
- **Timestamp**: When reviewed
- **Decision**: Correct/Incorrect
- **Notes**: Optional context
- **Prediction Details**: AI scores and features

---

## 📈 Timeline to Model Improvement

```
Day 1-2: Collect 10-15 predictions
         Review and validate
         
Day 3:   Run retraining
         New model deployed
         
Day 4+:  Monitor improved accuracy
         Collect more feedback
         Plan next retraining
```

---

## 🚀 Production Readiness

### Before Deployment

- [ ] Database backup strategy defined
- [ ] Clinician authentication enabled
- [ ] HTTPS configured
- [ ] Audit logging to external system
- [ ] Performance monitoring active
- [ ] Error handling tested

### During Operation

- [ ] Monitor dashboard regularly
- [ ] Accumulate feedback periodically
- [ ] Review accuracy trends
- [ ] Retrain when metrics plateau
- [ ] Document changes

---

## 💡 Pro Tips

1. **Quick Reviews**: Average clinician reviews 5-10 predictions per minute
2. **Batch Retraining**: Retrain when accuracy drops below 85%
3. **Feedback Patterns**: Look for systematic model errors in notes
4. **Clinician IDs**: Use format "DR_LASTNAME" for consistency
5. **Backup**: Keep database backups before retraining

---

## 📞 Support

For detailed documentation:
- [Full HITL Guide](HUMAN_IN_THE_LOOP_GUIDE.md)
- [System Architecture](IMPLEMENTATION_SUMMARY.md)
- [API Reference](REAL_TIME_GUIDE.md)

---

**Version**: 1.0  
**Last Updated**: April 15, 2026  
**System**: ML-Sepsis with LSTM + XGBoost Ensemble
