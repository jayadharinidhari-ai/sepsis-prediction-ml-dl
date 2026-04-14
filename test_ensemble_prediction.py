#!/usr/bin/env python3
"""
Test Script for LSTM + XGBoost Ensemble Predictions
====================================================

Tests the integrated sepsis prediction system with both LSTM and XGBoost models.
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.services.sepsis_engine import SepsisEngine
from app.services.store import PatientStore

def print_prediction_result(result, patient_name):
    """Pretty print prediction results"""
    print(f"\n{'='*70}")
    print(f"Patient: {patient_name}")
    print(f"{'='*70}")
    
    print(f"Model Type: {result.get('model_type', 'Unknown')}")
    print(f"\n📊 ENSEMBLE PREDICTION:")
    print(f"  ├─ Risk Score: {result['ensemble_score']*100:.1f}%")
    print(f"  ├─ Risk Level: {result['risk_level']}")
    
    if result.get('xgb_score') is not None:
        print(f"  ├─ XGBoost:   {result['xgb_score']*100:.1f}%")
    if result.get('lstm_score') is not None:
        print(f"  ├─ LSTM:      {result['lstm_score']*100:.1f}%")
    
    print(f"\n🎯 TOP CONTRIBUTING FEATURES:")
    for i, feat in enumerate(result.get('top_features', [])[:5], 1):
        direction = "↑" if feat['dir'] == 'high' else "↓"
        print(f"  {i}. {feat['name']:20s} = {feat['val']:8.2f}  [{feat['contrib']*100:5.1f}%] {direction}")
    
    print(f"\n📝 {result['message']}")
    print()

def main():
    print("\n" + "="*70)
    print(" SEPSIS PREDICTION ENSEMBLE TEST")
    print("="*70)
    
    try:
        # Initialize engine and store
        print("\n[INIT] Loading models...")
        engine = SepsisEngine()
        store = PatientStore()
        
        # Get demo patients
        patients = store.get_admitted_patients()
        
        print(f"[OK] Models loaded. Testing on {len(patients)} demo patients...\n")
        
        # Test predictions on each patient
        for patient in patients:
            result = engine.predict(patient)
            print_prediction_result(result, patient['name'])
            
            # Update patient risk in store
            store.update_patient_risk(
                patient['id'],
                result['ensemble_score'],
                result['top_features']
            )
        
        # Summary table
        print("\n" + "="*70)
        print(" SUMMARY TABLE")
        print("="*70)
        print(f"{'Patient':<25} {'Risk %':<10} {'Level':<12} {'Model':<20}")
        print("-"*70)
        
        for patient in patients:
            result = engine.predict(patient)
            print(f"{patient['name']:<25} {result['ensemble_score']*100:>7.1f}%  {result['risk_level']:<12} {result['model_type']:<20}")
        
        print("\n✓ Testing complete!")
        print("\nNote: Predictions use whichever models are available:")
        print("  • Both XGBoost + LSTM: Uses weighted ensemble (60% XGB + 40% LSTM)")
        print("  • Only XGBoost: Uses XGBoost score")
        print("  • Only LSTM: Uses LSTM score")
        print("  • Neither: Uses rule-based heuristic fallback")
        
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
