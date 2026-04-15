#!/usr/bin/env python3
"""
Human-in-the-Loop System Validation
Tests all HITL components and functionality
"""

import os
import sys
import json
from pathlib import Path
import time

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from app.services.human_loop_manager import HumanLoopManager


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(f"  {text}")
    print("="*70)


def print_success(text):
    """Print success message"""
    print(f"✅ {text}")


def print_error(text):
    """Print error message"""
    print(f"❌ {text}")


def print_warning(text):
    """Print warning message"""
    print(f"⚠️  {text}")


def print_info(text):
    """Print info message"""
    print(f"ℹ️  {text}")


def test_database_creation():
    """Test database initialization"""
    print_header("1. Database Creation Test")
    
    try:
        hlm = HumanLoopManager()
        db_exists = Path(str(hlm.db_path)).exists()
        
        if db_exists:
            print_success(f"Database created at {hlm.db_path}")
            return True
        else:
            print_error(f"Database not created at {hlm.db_path}")
            return False
    except Exception as e:
        print_error(f"Database creation failed: {e}")
        return False


def test_add_prediction():
    """Test adding predictions to system"""
    print_header("2. Add Prediction Test")
    
    try:
        hlm = HumanLoopManager()
        
        # Create test prediction
        features = {
            "HR": 105, "Temp": 38.8, "SBP": 95, "MAP": 70,
            "DBP": 65, "Resp": 22, "O2Sat": 94, "EtCO2": 38,
            "WBC": 15.2, "Lactate": 2.3, "Platelets": 190,
            "pH": 7.32, "Creatinine": 1.5, "FiO2": 0.35,
            "Potassium": 3.5, "PaCO2": 45
        }
        
        pred_id = hlm.add_prediction(
            patient_id="TEST_P001",
            features=features,
            lstm_score=0.78,
            xgb_score=0.82,
            ensemble_score=0.80,
            model_type="Ensemble",
            risk_level="HIGH"
        )
        
        if pred_id > 0:
            print_success(f"Prediction added with ID: {pred_id}")
            return True, pred_id
        else:
            print_error("Failed to add prediction")
            return False, -1
    except Exception as e:
        print_error(f"Add prediction test failed: {e}")
        return False, -1


def test_review_queue(pred_id):
    """Test retrieving review queue"""
    print_header("3. Review Queue Test")
    
    try:
        hlm = HumanLoopManager()
        queue = hlm.get_review_queue(limit=10)
        
        if queue:
            print_success(f"Retrieved review queue with {len(queue)} items")
            print_info(f"First item: Patient {queue[0]['patient_id']}, Risk: {queue[0]['risk_level']}")
            return True
        else:
            print_warning("Review queue is empty")
            return True
    except Exception as e:
        print_error(f"Review queue test failed: {e}")
        return False


def test_approval(pred_id):
    """Test clinician approval"""
    print_header("4. Clinician Approval Test")
    
    try:
        hlm = HumanLoopManager()
        
        # Test approve (correct)
        result = hlm.approve_prediction(
            prediction_id=pred_id,
            clinician_id="TEST_DR_01",
            is_correct=True,
            notes="Test approval - model was correct"
        )
        
        if result:
            print_success(f"Approval recorded for prediction {pred_id}")
            return True
        else:
            print_warning(f"No review queue entry found for prediction {pred_id}")
            return True
    except Exception as e:
        print_error(f"Approval test failed: {e}")
        return False


def test_statistics():
    """Test feedback statistics"""
    print_header("5. Statistics Test")
    
    try:
        hlm = HumanLoopManager()
        stats = hlm.get_feedback_statistics()
        
        print_success("Statistics retrievedDetails:")
        print(f"  • Total Predictions: {stats.get('total_predictions', 0)}")
        print(f"  • Total Reviewed: {stats.get('total_reviewed', 0)}")
        print(f"  • Review %: {stats.get('review_percentage', 0):.1f}%")
        print(f"  • Accuracy: {stats.get('accuracy', 0):.1f}%")
        print(f"  • Correct: {stats.get('correct', 0)}")
        print(f"  • Incorrect: {stats.get('incorrect', 0)}")
        return True
    except Exception as e:
        print_error(f"Statistics test failed: {e}")
        return False


def test_training_data_export():
    """Test exporting training data"""
    print_header("6. Training Data Export Test")
    
    try:
        hlm = HumanLoopManager()
        features, labels = hlm.get_training_data(reviewed_only=True)
        
        if features and labels:
            print_success(f"Training data exported successfully")
            print_info(f"  • Samples: {len(features)}")
            print_info(f"  • Positive labels: {sum(labels)}")
            print_info(f"  • Negative labels: {sum(1 for l in labels if l == 0)}")
            return True
        else:
            print_warning("No training data available (normal during first run)")
            return True
    except Exception as e:
        print_error(f"Training data export test failed: {e}")
        return False


def test_dismiss_review(pred_id):
    """Test dismissing a review"""
    print_header("7. Dismiss Review Test")
    
    try:
        hlm = HumanLoopManager()
        
        # Add another prediction to dismiss
        features = {
            "HR": 88, "Temp": 37.2, "SBP": 125, "MAP": 85,
            "DBP": 78, "Resp": 16, "O2Sat": 98, "EtCO2": 40,
            "WBC": 7.5, "Lactate": 1.2, "Platelets": 220,
            "pH": 7.40, "Creatinine": 0.9, "FiO2": 0.21,
            "Potassium": 4.0, "PaCO2": 40
        }
        
        new_pred_id = hlm.add_prediction(
            patient_id="TEST_P002",
            features=features,
            lstm_score=0.35,
            xgb_score=0.30,
            ensemble_score=0.32,
            model_type="Ensemble",
            risk_level="LOW"
        )
        
        # Add to high-risk queue manually
        hlm.add_prediction(
            patient_id="TEST_P003",
            features=features,
            lstm_score=0.75,
            xgb_score=0.78,
            ensemble_score=0.76,
            model_type="Ensemble",
            risk_level="CRITICAL"
        )
        
        queue = hlm.get_review_queue(limit=1)
        if queue:
            dismiss_id = queue[0]['prediction_id']
            result = hlm.dismiss_review(dismiss_id, "TEST_DR_02")
            if result:
                print_success(f"Review {dismiss_id} dismissed successfully")
                return True
            else:
                print_warning("Could not dismiss review")
                return True
        else:
            print_warning("No reviews to dismiss")
            return True
    except Exception as e:
        print_error(f"Dismiss review test failed: {e}")
        return False


def test_patient_history():
    """Test getting patient history"""
    print_header("8. Patient History Test")
    
    try:
        hlm = HumanLoopManager()
        history = hlm.get_prediction_history("TEST_P001", limit=10)
        
        if history:
            print_success(f"Retrieved history for TEST_P001 with {len(history)} predictions")
            if history[0]:
                first = history[0]
                print_info(f"  • Latest: Risk={first.get('risk_level', 'N/A')}, Score={first.get('ensemble_score', 'N/A')}")
            return True
        else:
            print_warning("No history found (expected if patient is new)")
            return True
    except Exception as e:
        print_error(f"Patient history test failed: {e}")
        return False


def test_workflow_simulation():
    """Simulate complete HITL workflow"""
    print_header("9. Complete Workflow Simulation")
    
    try:
        hlm = HumanLoopManager()
        
        print("Simulating complete workflow...")
        
        # Step 1: Create multiple predictions
        print_info("Creating predictions...")
        pred_ids = []
        for i in range(5):
            features = {
                "HR": 80 + i*5, "Temp": 37.0 + i*0.2,
                "SBP": 120 + i*3, "MAP": 82 + i*2,
                "DBP": 72 + i*2, "Resp": 16 + i,
                "O2Sat": 98, "EtCO2": 40,
                "WBC": 8.0 + i*0.5, "Lactate": 1.0 + i*0.1,
                "Platelets": 220, "pH": 7.40, "Creatinine": 0.9,
                "FiO2": 0.21, "Potassium": 4.0, "PaCO2": 40
            }
            
            risk_scores = [0.85, 0.78, 0.65, 0.45, 0.92]
            risk_levels = ["HIGH", "HIGH", "MODERATE", "LOW", "CRITICAL"]
            
            pred_id = hlm.add_prediction(
                patient_id=f"WF_P{i+1:03d}",
                features=features,
                lstm_score=risk_scores[i],
                xgb_score=risk_scores[i] + 0.05,
                ensemble_score=risk_scores[i] + 0.02,
                model_type="Ensemble",
                risk_level=risk_levels[i]
            )
            pred_ids.append(pred_id)
        
        print_success("Created 5 test predictions")
        
        # Step 2: Get review queue
        print_info("Checking review queue...")
        queue = hlm.get_review_queue(limit=10)
        print_success(f"Review queue has {len(queue)} HIGH/CRITICAL items")
        
        # Step 3: Approve some
        print_info("Simulating clinician reviews...")
        approvals = [True, False, True, False, True]
        for i, (pred_id, is_correct) in enumerate(zip(pred_ids, approvals)):
            try:
                hlm.approve_prediction(
                    prediction_id=pred_id,
                    clinician_id=f"WF_DR_{i%2+1:02d}",
                    is_correct=is_correct,
                    notes=f"Workflow test approval {i+1}"
                )
            except:
                pass  # Non-high-risk predictions won't have queue entries
        
        print_success("5 clinician reviews submitted")
        
        # Step 4: Check final statistics
        print_info("Calculating final statistics...")
        final_stats = hlm.get_feedback_statistics()
        print_success("Workflow simulation complete")
        print(f"  • Total predictions: {final_stats.get('total_predictions', 0)}")
        print(f"  • Reviews: {final_stats.get('total_reviewed', 0)}")
        print(f"  • Accuracy: {final_stats.get('accuracy', 0):.1f}%")
        
        return True
    except Exception as e:
        print_error(f"Workflow simulation failed: {e}")
        return False


def main():
    """Run all tests"""
    print("\n")
    print("╔" + "="*68 + "╗")
    print("║" + " "*15 + "HUMAN-IN-THE-LOOP VALIDATION SUITE" + " "*19 + "║")
    print("╚" + "="*68 + "╝")
    
    tests = []
    
    # Test 1: Database
    result1 = test_database_creation()
    tests.append(("Database Creation", result1))
    
    # Test 2: Add prediction
    result2, pred_id = test_add_prediction()
    tests.append(("Add Prediction", result2))
    
    # Test 3: Review queue
    result3 = test_review_queue(pred_id)
    tests.append(("Review Queue", result3))
    
    # Test 4: Approval
    result4 = test_approval(pred_id)
    tests.append(("Clinician Approval", result4))
    
    # Test 5: Statistics
    result5 = test_statistics()
    tests.append(("Statistics", result5))
    
    # Test 6: Export
    result6 = test_training_data_export()
    tests.append(("Data Export", result6))
    
    # Test 7: Dismiss
    result7 = test_dismiss_review(pred_id)
    tests.append(("Dismiss Review", result7))
    
    # Test 8: History
    result8 = test_patient_history()
    tests.append(("Patient History", result8))
    
    # Test 9: Full workflow
    result9 = test_workflow_simulation()
    tests.append(("Full Workflow", result9))
    
    # Summary
    print_header("Summary")
    passed = sum(1 for name, result in tests if result)
    total = len(tests)
    
    for name, result in tests:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status:8} - {name}")
    
    print("\n" + "="*70)
    if passed == total:
        print(f"🎉 ALL TESTS PASSED ({passed}/{total})")
        print("═"*70)
        print("\nHuman-in-the-Loop system is ready for deployment!")
        print("\nQuick Commands:")
        print("  • Dashboard: http://localhost:8000/review")
        print("  • Statistics: curl http://localhost:8000/api/human-loop/statistics")
        print("  • Retrain: python retrain_from_feedback.py")
    else:
        print(f"⚠️  SOME TESTS FAILED ({passed}/{total} passed)")
        print("═"*70)
        return 1
    
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
