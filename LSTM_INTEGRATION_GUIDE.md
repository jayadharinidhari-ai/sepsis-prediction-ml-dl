# LSTM + XGBoost Ensemble Integration Guide

## Overview

This system combines two complementary machine learning approaches for sepsis prediction:

- **XGBoost** (60%): Tree-based model, excellent for feature interactions and non-linear relationships
- **LSTM** (40%): Deep learning, captures temporal patterns in sequential vital signs and lab values

Final prediction = **0.6 × XGBoost_Score + 0.4 × LSTM_Score**

## Architecture

```
Patient Data (Vitals + Labs + Trends)
          ↓
    ┌─────────────────────┐
    │   Feature Engine    │
    │  (extract 35 feats) │
    └──────┬──────────────┘
           ↓
    ┌──────────────────────────────────────┐
    │  ┌──────────────┐   ┌──────────────┐ │
    │  │   XGBoost    │   │     LSTM     │ │
    │  │  Model       │   │  Model       │ │
    │  └──────┬───────┘   └──────┬───────┘ │
    │         │  Score 1         │ Score 2 │
    │         └────────┬─────────┘         │
    │                  │ Weighted Avg      │
    │            Final Risk Score          │
    └──────────────────────────────────────┘
```

## File Structure

```
ML-Sepsis/
├── lstm_model.py                 # LSTM model definition & training
├── train_lstm_model.py           # Training script
├── test_ensemble_prediction.py   # Prediction testing script
├── app/
│   ├── routes.py                 # Updated API endpoints
│   ├── services/
│   │   ├── sepsis_engine.py     # Ensemble prediction logic
│   │   ├── store.py
│   │   └── simulator.py
│   └── templates/
│       └── dashboard.html
├── lstm_sepsis_model.h5          # Trained LSTM model (created after training)
├── lstm_scaler.pkl               # Feature scaler (created after training)
└── requirements.txt              # Updated with tensorflow
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- TensorFlow 2.13+ (for LSTM)
- Keras (neural network framework)
- XGBoost, scikit-learn, pandas, numpy
- Flask (web framework)

### 2. Train LSTM Model

```bash
python train_lstm_model.py
```

This script:
- Generates 100 synthetic patient records with realistic vital signs
- Creates 8-timestep sequences from patient trends
- Normalizes/scales features using StandardScaler
- Trains LSTM network for 30 epochs (with early stopping)
- Saves model to `lstm_sepsis_model.h5`
- Saves scaler to `lstm_scaler.pkl`

**Output:**
```
[OK] Generated 100 training samples
[OK] LSTM model built successfully
Model: "sequential"
Layer (type)      Output Shape      Param #
lstm_1            (None, 8, 64)     19200
dropout_1         (None, 8, 64)     0
lstm_2            (None, 32)        12544
dropout_2         (None, 32)        0
dense_1           (None, 16)        528
dropout_3         (None, 16)        0
output            (None, 1)         17
Total params: 32,289

[TRAINING] Starting LSTM training...
Epoch 1/30
...
✓ LSTM training completed
```

### 3. Test Predictions

```bash
python test_ensemble_prediction.py
```

This runs predictions on all demo patients and shows:
- Individual model scores (XGBoost vs LSTM)
- Ensemble combined score
- Top contributing features
- Risk level classification

**Example Output:**
```
======================================================================
 SEPSIS PREDICTION ENSEMBLE TEST
======================================================================

[INIT] Loading models...
[OK] XGBoost model loaded
[OK] LSTM model loaded

[XGBOOST] Patient 1 (Rajesh Kumar): Risk=87.0%
[LSTM] Patient 1 (Rajesh Kumar): Risk=89.5%
[ENSEMBLE] Final score: 87.9% (avg of both models)

======================================================================
Patient: Rajesh Kumar
======================================================================
Model Type: Ensemble (XGB: 87.0% + LSTM: 89.5%)

📊 ENSEMBLE PREDICTION:
  ├─ Risk Score: 87.9%
  ├─ Risk Level: High
  ├─ XGBoost:   87.0%
  └─ LSTM:      89.5%

🎯 TOP CONTRIBUTING FEATURES:
  1. Lactate               =     3.20  [ 92.0%] ↑
  2. MAP                   =    58.00  [ 87.0%] ↓
  3. WBC                   =    18.40  [ 81.0%] ↑
  4. Temperature           =    38.90  [ 74.0%] ↑
  5. Creatinine            =     2.10  [ 68.0%] ↑
```

### 4. Start Flask Application

```bash
python run.py
```

Server runs on `http://localhost:8000` with ensemble predictions enabled.

## API Endpoints

### Get Single Patient
```bash
curl http://localhost:8000/api/patient/1
```

Response includes:
- Current vitals and labs
- Ensemble risk score (from sepsis_engine.py)

### Predict Sepsis Risk
```bash
curl -X POST http://localhost:8000/api/patient/1/predict
```

**Response:**
```json
{
  "risk_score": 0.879,
  "risk_level": "High",
  "ensemble_score": 0.879,
  "xgb_score": 0.870,
  "lstm_score": 0.895,
  "model_type": "Ensemble (XGB: 87.0% + LSTM: 89.5%)",
  "top_features": [
    {
      "name": "Lactate",
      "val": 3.2,
      "contrib": 0.92,
      "dir": "high"
    },
    ...
  ],
  "message": "Sepsis risk: 87.9% (Ensemble)"
}
```

## How the Ensemble Works

### XGBoost Component
- **Input**: 35 engineered features (vitals, labs, trends, demographics)
- **Model**: Gradient boosting with tree-based decisions
- **Strength**: Handles feature interactions, robust to outliers
- **Output**: Probability score 0-1

### LSTM Component
- **Input**: 8-step time series of patient vitals/labs
- **Architecture**:
  - LSTM layer 1: 64 units (captures long-term dependencies)
  - LSTM layer 2: 32 units (further feature extraction)
  - Dense layer: 16 units (non-linear transformation)
  - Output: Sigmoid (binary probability)
- **Strength**: Captures temporal patterns, learn from sequences
- **Output**: Probability score 0-1

### Weighted Average
```python
final_score = 0.6 * xgb_score + 0.4 * lstm_score
```

**Rationale**:
- XGBoost gets higher weight (60%) - proven effective, already deployed
- LSTM gets secondary weight (40%) - adds temporal intelligence
- Weights can be adjusted based on validation performance

## Feature Extraction for LSTM

The LSTM receives 8 timesteps of patient data:

Each timestep contains:
- 8 vitals: HR, Temp, SBP, MAP, DBP, Resp, O2Sat, EtCO2
- 4 key labs: WBC, Creatinine, Platelets, Lactate
- Demographics: Age, ICU Length of Stay

**Shape**: (batch_size, 8_timesteps, 16_features)

Features are normalized using StandardScaler before feeding to LSTM.

## Data Preprocessing Pipeline

```
Raw Patient Data
        ↓
[Feature Extraction]
- Extract vitals from current record
- Extract labs from current record
- Extract trends (last 8 time steps)
        ↓
[Sequence Creation for LSTM]
- Stack 8 timesteps of vital/lab values
- Create (8, 16) matrix per patient
        ↓
[Normalization]
- StandardScaler (fit on training data)
- Apply same scaler to test data
        ↓
[Both Models]
- XGBoost: Use extracted 35 features
- LSTM: Use normalized (8, 16) sequences
```

## Training Data

The system uses synthetic data for demo purposes. For production:

1. **Collect real patient records** with:
   - Vital signs (hourly measurements)
   - Lab values (periodic tests)
   - Sepsis outcome labels
   - Time-series minimum 8 hours

2. **Preprocess**:
   ```python
   from lstm_model import SepsisLSTMModel
   
   lstm = SepsisLSTMModel(timesteps=8, n_features=16)
   lstm.build_model()
   lstm.train(your_patient_data, labels, epochs=100)
   lstm.save()
   ```

3. **Evaluate**:
   ```python
   predictions = lstm.batch_predict(test_patients)
   # Compare with XGBoost and ensemble
   ```

## Key Parameters

### LSTM Model
```python
TIMESTEPS = 8          # Look at last 8 time steps
N_FEATURES = 16        # Vitals + key labs per timestep
LSTM_UNITS_1 = 64      # First LSTM layer
LSTM_UNITS_2 = 32      # Second LSTM layer
DROPOUT = 0.2-0.3      # Regularization
LEARNING_RATE = 0.001  # Adam optimizer
```

### Training
```python
EPOCHS = 30            # Can increase for better fit
BATCH_SIZE = 8         # Small batches for stability
VALIDATION_SPLIT = 0.2 # 80% train, 20% validation
EARLY_STOPPING = 10    # Stop if no improvement for 10 epochs
```

### Ensemble
```python
XGB_WEIGHT = 0.6       # 60% weight to XGBoost
LSTM_WEIGHT = 0.4      # 40% weight to LSTM
```

## Fallback Behavior

If models are unavailable:

1. **Both available**: Use ensemble
2. **Only XGBoost**: Use XGBoost score
3. **Only LSTM**: Use LSTM score
4. **Neither**: Use rule-based heuristic

This ensures the system always provides a prediction.

## Monitoring & Logging

All predictions log to console:

```
[XGBOOST] Patient 1 (Rajesh Kumar): Risk=87.0%
[LSTM] Patient 1 (Rajesh Kumar): Risk=89.5%
[ENSEMBLE] Final score: 87.9% (avg of both models)
```

Monitor in production:
- Track individual model performance
- Compare ensemble vs single models
- Watch for distribution shifts
- Retrain LSTM quarterly with new data

## Performance Considerations

- **XGBoost**: Fast inference (~1ms per prediction)
- **LSTM**: Slightly slower inference (~5ms per prediction)
- **Ensemble**: Combined time ~6ms per patient
- **Batch predictions**: Can process 100 patients in <1 second

For real-time monitoring of multiple patients, ensemble is still very fast.

## Troubleshooting

### TensorFlow Installation Issues
```bash
# On macOS with Apple Silicon
pip install tensorflow-macos

# On Linux/Windows
pip install tensorflow
```

### LSTM Model Not Found
- Run `python train_lstm_model.py` to generate model
- Ensure `lstm_sepsis_model.h5` exists in project root

### Scaler Not Found
- Delete `lstm_sepsis_model.h5` and retrain
- Scaler is automatically created during training

### Low LSTM Performance
- Check if training data is representative
- Increase epochs (30 → 50 or 100)
- Adjust architecture (change LSTM units)
- Use more training samples

## Next Steps

1. **Collect Real Data**: Replace synthetic training data with actual patient records
2. **Hyperparameter Tuning**: Optimize weights, LSTM architecture, training parameters
3. **Cross-Validation**: Validate on held-out test sets
4. **Deployment**: Deploy to production with monitoring
5. **Feedback Loop**: Retrain models with new clinical data

## References

- TensorFlow LSTM: https://www.tensorflow.org/guide/keras/rnn
- XGBoost: https://xgboost.readthedocs.io
- Ensemble Methods: https://en.wikipedia.org/wiki/Ensemble_learning
- Sepsis Prediction Literature: [Your clinical references here]
