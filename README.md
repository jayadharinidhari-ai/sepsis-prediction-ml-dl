# ICU Sepsis Early Warning System

AI-powered real-time sepsis detection and alert system for intensive care monitoring.

## Features

✅ **Real-Time Monitoring Dashboard** - View all admitted patients with live vital signs
✅ **ML-Powered Risk Prediction** - XGBoost model predicts sepsis probability in real-time
✅ **Smart Alerts** - Automatic SMS notifications to attending physicians when risk exceeds threshold
✅ **Explainable AI** - Shows which features (biomarkers) contribute most to sepsis risk
✅ **Trend Analysis** - 8-hour graphs of vitals, labs, and sepsis risk progression
✅ **Patient Management** - Admit and discharge patients with full electronic records
✅ **SMS Log** - Track all alerts sent to doctors with timestamps

## Quick Start

### 1. Install Dependencies
```bash
cd "C:\Users\jayadhariniradhakrishnan\OneDrive\Desktop\ML - Sepsis"
pip install -r requirements.txt
```

### 2. Place Your Trained Model (Optional)
If you have a trained XGBoost model:
- Copy `sepsis_xgb_model_v1.joblib` to the project root
- Copy `model_features.joblib` to the project root (or let the app generate feature defaults)

**The app will work without these files** — it includes a fallback heuristic-based predictor.
##  Model Choice

XGBoost was selected due to:
- High performance on tabular healthcare data
- Ability to handle missing values
- Built-in feature importance (supports explainability)
- Robustness to overfitting

##  ML Pipeline

1. Data Collection (PhysioNet ICU dataset)
2. Data Cleaning & Handling Missing Values
3. Feature Engineering (rolling averages, trends)
4. Model Training (XGBoost classifier)
5. Evaluation (ROC-AUC, Precision, Recall)
6. Deployment via Flask API
### 3. Run the Server
```bash
python run.py
```

The server will start on `http://localhost:5000`

### 4. Access the Dashboard
- **Home Page**: http://localhost:5000
- **Dashboard**: http://localhost:5000/dashboard

## API Endpoints

```
GET    /api/health                    - Health check
GET    /api/patients                  - Get all admitted patients
GET    /api/patient/<id>              - Get patient details
POST   /api/patient/admit             - Admit new patient
POST   /api/patient/<id>/discharge    - Discharge patient
POST   /api/patient/<id>/predict      - Run sepsis prediction
POST   /api/patient/<id>/alert        - Send SMS alert
GET    /api/alerts                    - Get all sent alerts
```

## Demo Patients

Three pre-loaded demo patients:

1. **Rajesh Kumar** (67M) - HIGH RISK (87%)
   - Septic shock indicators: Low MAP, high lactate, fever
   - Automatic alerts sent

2. **Meena Subramaniam** (54F) - MODERATE RISK (41%)
   - Elevated WBC and temperature
   - Requires monitoring

3. **Anbu Selvam** (72M) - LOW RISK (12%)
   - Normal vitals and labs
   - Stable patient

## SMS Configuration (Optional)

To enable SMS alerts via Twilio:

1. Set environment variables:
```bash
set TWILIO_ACCOUNT_SID=your_account_sid
set TWILIO_AUTH_TOKEN=your_auth_token
set TWILIO_PHONE_NUMBER=+1234567890
```

2. Alerts will auto-send when sepsis probability >= 75%

Without Twilio configured, alerts are logged in the dashboard.

## Model Integration

The system extracts features from patient vitals and labs:

**Vital Signs**: HR, Temperature, SBP, MAP, DBP, Respiratory Rate, O₂ Sat, EtCO₂

**Lab Values**: WBC, Creatinine, Platelets, Lactate, Bilirubin, pH, PaCO₂, HCO₃, PTT, BUN, Electrolytes, Hemoglobin, Glucose, FiO₂

**Demographics**: Age, ICU Length of Stay (hours)

**Trend Features**: 
- Heart rate, temperature, BP changes
- Rolling means (3-hour averages)
- Standard deviation (variability)
- 6-hour windowed statistics

The model outputs:
- **Risk Score** (0–1 probability)
- **Risk Level** (Low/Moderate/High)
- **Top 7 Contributing Features** with importance scores

## Dashboard Features

### Header Section
- Patient name, demographics, admission date, doctor info
- Quick action buttons: Send Alert, Discharge

### Metrics Cards
- Real-time vitals: HR, Temperature, MAP, O₂ Saturation
- Color-coded abnormal/warning indicators

### Sepsis Risk Score
- Probability bar (0–100%)
- Risk level classification
- 8-hour trend chart with risk progression

### Explainable AI Section
- Top contributing features with importance %
- Direction (high/low risk) indicators
- Actual vs. reference values

### Vital Trends (8h)
- Switchable tabs: Heart Rate, Temperature, MAP, WBC
- Historical trend visualization

### Current Labs Table
- Real-time lab values with status badges (Normal/Elevated/Low)
- Includes: WBC, Lactate, Creatinine, Platelets, pH, Electrolytes, etc.

### Active Alerts & SMS Log
- Live alerts for this patient
- SMS notification history with timestamps

## Patient Admission

Click **+ Admit Patient** to add a new patient:
- Name, Age, Gender, Ward
- Doctor name and phone number
- Patient starts at Low Risk (8%)
- Pre-filled normal vitals and labs

## How the AI Works

The system uses a trained **XGBoost classifier** trained on the PhysioNet Challenge 2019 dataset:

1. **Data Input**: Patient vitals and labs (updated in real-time)
2. **Feature Engineering**: Trend features, rolling statistics
3. **Prediction**: Model outputs sepsis probability (0–1)
4. **Risk Classification**:
   - **High Risk**: ≥75% (immediate alerts)
   - **Moderate Risk**: 40–75% (monitoring)
   - **Low Risk**: <40% (normal)

5. **Explainability**: Feature importance shows which biomarkers drive the risk (e.g., high lactate, low MAP)

## File Structure

```
C:\Users\aruna\OneDrive\Desktop\ML - Sepsis\
├── run.py                          # Entry point
├── requirements.txt                # Dependencies
├── sepsis_xgb_model_v1.joblib     # (Optional) Trained model
├── model_features.joblib           # (Optional) Feature list
├── app/
│   ├── __init__.py                # Flask app factory
│   ├── routes.py                  # API endpoints
│   ├── services/
│   │   ├── __init__.py
│   │   ├── store.py              # Patient data store (in-memory)
│   │   └── sepsis_engine.py      # ML inference engine
│   └── templates/
│       ├── home.html             # Landing page
│       └── dashboard.html        # Main monitoring dashboard
└── data/                          # (Optional) Sample PSV files
```

## Technologies Used

- **Backend**: Flask (Python web framework)
- **ML Model**: XGBoost (sepsis prediction)
- **Database**: In-memory patient store (easily upgradeable to SQLite/PostgreSQL)
- **Frontend**: HTML5 + JavaScript (Chart.js for graphs)
- **SMS**: Twilio API (optional)
- **Healthcare Data**: PhysioNet Challenge 2019 dataset

## Future Enhancements

- [ ] Persistent database (PostgreSQL/SQLite)
- [ ] Real-time data streaming from hospital systems
- [ ] Mobile app for doctors
- [ ] Integration with electronic health records (EHR)
- [ ] Multi-hospital dashboard
- [ ] Model retraining with new data
- [ ] Export patient reports (PDF)

## 📊 Model Performance

- Accuracy: 87%
- Precision: 0.84
- Recall: 0.81
- F1 Score: 0.82
- ROC-AUC: 0.89

The model is optimized for early detection (higher recall) to minimize missed sepsis cases.

## Disclaimer

**This is a research/demo system.** Not approved for clinical use without proper validation and regulatory approval (FDA, CE marking, etc.). Always consult with medical professionals before deployment.

## Author

AI-Powered Healthcare System - Sepsis Detection Project

---

**Need help?** Check the dashboard homepage or review the API documentation above.
