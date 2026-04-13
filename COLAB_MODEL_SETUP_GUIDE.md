# How to Get Your Trained Model from Colab to Web App

## OPTION A: Fastest Way (Recommended)

### Step 1: Open Your Colab Notebook
- Go to: https://colab.research.google.com/drive/1O5fPIPn92PSzEB6tI6TDkvv3Ne4SO11l
- Or open your saved notebook

### Step 2: Run All Cells
- Click "Runtime" → "Run all"
- Wait for training to complete (~5-10 minutes)

### Step 3: Execute This Cell (at the END of notebook)
Copy-paste and run this cell:

```python
import joblib

# Save your trained model
joblib.dump(model_ada, 'sepsis_xgb_model_v1.joblib')
joblib.dump(X_train_advanced.columns.tolist(), 'model_features.joblib')

print("[OK] Models saved!")

# Download to your computer
from google.colab import files
files.download('sepsis_xgb_model_v1.joblib')
files.download('model_features.joblib')
print("[OK] Check your Downloads folder!")
```

### Step 4: Move Downloaded Files
```
Downloads/
├── sepsis_xgb_model_v1.joblib
└── model_features.joblib

↓ Move to ↓

c:/Users/aruna/OneDrive/Desktop/ML - Sepsis/
├── sepsis_xgb_model_v1.joblib
└── model_features.joblib
```

### Step 5: Restart Web App
```bash
cd "c:/Users/aruna/OneDrive/Desktop/ML - Sepsis"
python run.py
```

✅ Your model is now live in the web app!

---

## OPTION B: Using Pre-trained Model (If Already Trained)

If you already ran the Colab notebook before:

### Step 1: Check Colab Files
- Open Colab
- Click "Files" icon (left sidebar)
- Look for `sepsis_xgb_model_v1.joblib` and `model_features.joblib`

### Step 2: Download
- Right-click each file → Download
- Save to your computer

### Step 3: Move to Project Directory
```
c:/Users/aruna/OneDrive/Desktop/ML - Sepsis/
```

### Step 4: Verify
```bash
cd "c:/Users/aruna/OneDrive/Desktop/ML - Sepsis"
python verify_model.py
```

Should show:
```
[OK] SUCCESS - Your trained ML model is ACTIVE
```

---

## What Each File Is For

| File | Size | Purpose |
|------|------|---------|
| `sepsis_xgb_model_v1.joblib` | ~108 KB | Your trained XGBoost model |
| `model_features.joblib` | ~346 bytes | List of 36 feature names |

Both files needed for the app to work correctly!

---

## Troubleshooting

**If the web app doesn't load your model:**

1. Check app startup log:
   ```bash
   tail /tmp/flask_app.log | grep "Model loaded"
   ```

2. Should see:
   ```
   [OK] Model loaded from C:\Users\...\sepsis_xgb_model_v1.joblib
   [OK] Features loaded from C:\Users\...\model_features.joblib
   ```

3. If not showing, verify files exist:
   ```bash
   ls -lh "c:/Users/aruna/OneDrive/Desktop/ML - Sepsis/"*.joblib
   ```

---

## Quick Summary

|  | Location |
|---|----------|
| Your Colab Notebook | https://colab.research.google.com (run & download files) |
| Downloaded Files | Your computer's Downloads folder |
| Final Location | `c:/Users/aruna/OneDrive/Desktop/ML - Sepsis/` |
| Web App | http://localhost:5000 |

---

## Already Have Files Downloaded?

If you already have the .joblib files on your computer:

```bash
# Copy both files to project directory
cp "path/to/downloads/sepsis_xgb_model_v1.joblib" "c:/Users/aruna/OneDrive/Desktop/ML - Sepsis/"
cp "path/to/downloads/model_features.joblib" "c:/Users/aruna/OneDrive/Desktop/ML - Sepsis/"

# Restart app
cd "c:/Users/aruna/OneDrive/Desktop/ML - Sepsis"
python run.py

# Verify
python verify_model.py
```

Done! Your model is now active. ✅
