#!/usr/bin/env python3
"""
LSTM Model Training Script for Sepsis Prediction
=====================================================

This script trains the LSTM deep learning model on synthetic patient data.
Run this before using the integrated LSTM+XGBoost ensemble system.

Usage:
    python train_lstm_model.py

Output:
    - lstm_sepsis_model.h5 (trained model)
    - lstm_scaler.pkl (feature scaler)
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    try:
        from lstm_model import train_and_save_lstm
        
        print("\n" + "=" * 70)
        print(" LSTM MODEL TRAINING FOR SEPSIS PREDICTION")
        print("=" * 70)
        print("\nThis script will:")
        print("  1. Generate 100 synthetic patient records")
        print("  2. Create time-series sequences from patient vitals/labs")
        print("  3. Normalize/scale features")
        print("  4. Train LSTM neural network (epochs=30)")
        print("  5. Save model to lstm_sepsis_model.h5")
        print("  6. Save scaler to lstm_scaler.pkl")
        
        response = input("\nProceed with training? (y/n): ").strip().lower()
        
        if response not in ['y', 'yes']:
            print("[CANCELLED] Training aborted by user")
            return
        
        # Train and save
        lstm_model = train_and_save_lstm(epochs=30)
        
        print("\n" + "=" * 70)
        print(" ✓ TRAINING COMPLETE")
        print("=" * 70)
        print("\nYou can now:")
        print("  1. Use the LSTM model in Flask app (automatic integration)")
        print("  2. Test predictions: python test_lstm.py")
        print("  3. View ensemble results in /api/patient/predict endpoint")
        
    except ImportError as e:
        print(f"\n[ERROR] Missing required package: {e}")
        print("\nInstall dependencies with:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
