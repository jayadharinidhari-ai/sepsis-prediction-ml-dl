# COMPLETE GUIDE: From Colab Model to Web App

## WHAT YOU HAVE
Your Colab notebook with a trained XGBoost model called `model_ada`

## WHAT YOU NEED TO DO
Convert it to 2 joblib files and place them in your web app directory

## PROCESS (Takes 10 minutes)

### PART 1: Get Files from Colab (5 minutes)

1. **Open your Colab notebook:**
   - Go to: https://colab.research.google.com/drive/1O5fPIPn92PSzEB6tI6TDkvv3Ne4SO11l
   - Or go to your Google Drive and open the notebook

2. **Go to the bottom of the notebook (after all training code)**

3. **Add a new code cell and paste this:**

```python
import joblib
import os

print("="*70)
print("EXPORTING TRAINED MODEL")
print("="*70)

model_filename = 'sepsis_xgb_model_v1.joblib'
features_filename = 'model_features.joblib'

# Save model
joblib.dump(model_ada, model_filename)
print(f"[OK] Model saved: {model_filename}")

# Save features
joblib.dump(X_train_advanced.columns.tolist(), features_filename)
print(f"[OK] Features saved: {features_filename}")

# Download
from google.colab import files
print("\n[Downloading...]")
files.download(model_filename)
files.download(features_filename)
print("[OK] Check Downloads folder!")
```

4. **Click "Run Cell" (or press Ctrl+Enter)**

5. **Wait for downloads**
   You'll see popup asking to download 2 files:
   - `sepsis_xgb_model_v1.joblib`
   - `model_features.joblib`
   
   Accept both downloads!

---

### PART 2: Move Files to Project (2 minutes)

1. **Open your Downloads folder**
   ```
   C:\Users\aruna\Downloads\
   ```

2. **Find both .joblib files:**
   - `sepsis_xgb_model_v1.joblib` (108 KB)
   - `model_features.joblib` (346 bytes)

3. **Move them to your project:**
   ```
   C:\Users\aruna\OneDrive\Desktop\ML - Sepsis\
   ```

   (Or copy-paste them there)

---

### PART 3: Activate Model in Web App (3 minutes)

1. **Restart the Flask app:**
   ```bash
   cd "C:/Users/aruna/OneDrive/Desktop/ML - Sepsis"
   python run.py
   ```

2. **Check the startup log**
   You should see:
   ```
   [OK] Model loaded from C:\...\sepsis_xgb_model_v1.joblib
   [OK] Features loaded from C:\...\model_features.joblib
   ```

3. **Verify it's working:**
   ```bash
   python verify_model.py
   ```

   Should show:
   ```
   [OK] SUCCESS - Your trained ML model is ACTIVE
   ```

---

## VISUAL FILE STRUCTURE

```
BEFORE (without your model):
c:/Users/aruna/OneDrive/Desktop/ML - Sepsis/
├── run.py
├── requirements.txt
├── verify_model.py
├── app/
└── (no .joblib files - using demo model)

AFTER (with your model):
c:/Users/aruna/OneDrive/Desktop/ML - Sepsis/
├── run.py
├── requirements.txt
├── verify_model.py
├── sepsis_xgb_model_v1.joblib         ← YOUR MODEL
├── model_features.joblib              ← YOUR FEATURES
├── app/
└── (app automatically loads these files!)
```

---

## VERIFICATION CHECKLIST

- [ ] Colab notebook runs successfully
- [ ] You downloaded 2 .joblib files from Colab
- [ ] Files are in `C:\Users\aruna\OneDrive\Desktop\ML - Sepsis\`
- [ ] Flask app started with `[OK] Model loaded...` message
- [ ] `python verify_model.py` shows success
- [ ] API predictions show `(ML Model)` in the message
- [ ] Web app at http://localhost:5000 is responsive

---

## TROUBLESHOOTING

**Problem: "Model not found" error**
- Check if .joblib files exist:
  ```bash
  ls "c:/Users/aruna/OneDrive/Desktop/ML - Sepsis/"*.joblib
  ```

**Problem: Predictions still say "fallback heuristic"**
- Restart the app:
  ```bash
  python run.py
  ```

**Problem: Colab download didn't work**
- Manually download from Colab Files panel:
  1. Click "Files" icon (left sidebar)
  2. Right-click each file → Download
  3. Move to project directory

**Problem: "Module not found" in Colab**
- Make sure you ran this cell first:
  ```python
  !pip install joblib xgboost pandas numpy scikit-learn
  ```

---

## EXPECTED RESULTS

When using YOUR trained model, predictions should:
- ✓ Show different feature importance than demo model
- ✓ Change based on actual patient vitals
- ✓ Say "(ML Model)" in the message
- ✓ Use all 36 features you trained with

Example:
```json
{
  "message": "Sepsis risk: 23.4% (ML Model)",
  "top_features": [
    {"name": "Resp", "importance": 0.08},
    {"name": "Lactate", "importance": 0.07},
    ...
  ]
}
```

---

## NEED HELP?

Run this script to diagnose issues:
```bash
python verify_model.py
```

It will tell you exactly what's loaded and working!

---

That's it! You're done. Your trained model will power your web app! 🎉
