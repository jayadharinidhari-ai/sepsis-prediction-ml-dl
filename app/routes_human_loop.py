"""
Human-in-the-Loop Routes
API endpoints for clinician review, approval, and feedback
"""

from flask import Blueprint, request, jsonify
from app.services.human_loop_manager import get_human_loop_manager
import logging

logger = logging.getLogger(__name__)

human_loop_bp = Blueprint('human_loop', __name__, url_prefix='/api/human-loop')

hlm = get_human_loop_manager()


@human_loop_bp.route('/review-queue', methods=['GET'])
def get_review_queue():
    """
    Get pending predictions for clinician review
    Query params: status (pending, reviewed, dismissed), limit (default: 10)
    """
    try:
        status = request.args.get('status', 'pending')
        limit = int(request.args.get('limit', 10))
        
        queue = hlm.get_review_queue(status=status, limit=limit)
        
        # Parse JSON features
        for item in queue:
            if isinstance(item['features'], str):
                import json
                item['features'] = json.loads(item['features'])
        
        return jsonify({
            "success": True,
            "count": len(queue),
            "queue": queue
        }), 200
    except Exception as e:
        logger.error(f"[ERROR] Failed to fetch review queue: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@human_loop_bp.route('/approve', methods=['POST'])
def approve_prediction():
    """
    Clinician approves/rejects a prediction
    Body: {
        "prediction_id": int,
        "clinician_id": str,
        "is_correct": bool,
        "notes": str (optional)
    }
    """
    try:
        data = request.json
        prediction_id = data.get('prediction_id')
        clinician_id = data.get('clinician_id')
        is_correct = data.get('is_correct', True)
        notes = data.get('notes', '')
        
        if not prediction_id or not clinician_id:
            return jsonify({"success": False, "error": "Missing prediction_id or clinician_id"}), 400
        
        result = hlm.approve_prediction(prediction_id, clinician_id, is_correct, notes)
        
        if result:
            # Get updated statistics
            stats = hlm.get_feedback_statistics()
            return jsonify({
                "success": True,
                "message": f"Prediction approved by {clinician_id}",
                "feedback": "correct" if is_correct else "incorrect",
                "statistics": stats
            }), 200
        else:
            return jsonify({"success": False, "error": "Failed to approve prediction"}), 500
    except Exception as e:
        logger.error(f"[ERROR] Failed to approve prediction: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@human_loop_bp.route('/dismiss', methods=['POST'])
def dismiss_review():
    """
    Clinician dismisses a review without feedback
    Body: {
        "prediction_id": int,
        "clinician_id": str
    }
    """
    try:
        data = request.json
        prediction_id = data.get('prediction_id')
        clinician_id = data.get('clinician_id')
        
        if not prediction_id or not clinician_id:
            return jsonify({"success": False, "error": "Missing fields"}), 400
        
        result = hlm.dismiss_review(prediction_id, clinician_id)
        
        if result:
            return jsonify({
                "success": True,
                "message": "Review dismissed"
            }), 200
        else:
            return jsonify({"success": False, "error": "Failed to dismiss review"}), 500
    except Exception as e:
        logger.error(f"[ERROR] Failed to dismiss review: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@human_loop_bp.route('/statistics', methods=['GET'])
def get_statistics():
    """Get feedback statistics and model accuracy"""
    try:
        stats = hlm.get_feedback_statistics()
        return jsonify({
            "success": True,
            "statistics": stats
        }), 200
    except Exception as e:
        logger.error(f"[ERROR] Failed to fetch statistics: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@human_loop_bp.route('/history/<patient_id>', methods=['GET'])
def get_patient_history(patient_id):
    """Get prediction history for a specific patient"""
    try:
        limit = int(request.args.get('limit', 20))
        history = hlm.get_prediction_history(patient_id, limit=limit)
        
        return jsonify({
            "success": True,
            "patient_id": patient_id,
            "predictions": history
        }), 200
    except Exception as e:
        logger.error(f"[ERROR] Failed to fetch history: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@human_loop_bp.route('/export-feedback', methods=['GET'])
def export_feedback():
    """Export reviewed feedback for analysis or retraining"""
    try:
        reviewed_only = request.args.get('reviewed_only', 'true').lower() == 'true'
        features_list, labels_list = hlm.get_training_data(reviewed_only=reviewed_only)
        
        return jsonify({
            "success": True,
            "sample_count": len(features_list),
            "features": features_list,
            "labels": labels_list
        }), 200
    except Exception as e:
        logger.error(f"[ERROR] Failed to export feedback: {e}")
        return jsonify({"success": False, "error": str(e)}), 500
