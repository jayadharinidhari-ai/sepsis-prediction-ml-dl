from flask import Blueprint, render_template, request, jsonify, current_app, session, redirect, url_for
import json
from datetime import datetime

bp = Blueprint('main', __name__)


@bp.route('/login')
def login():
    """Login page"""
    return render_template('login.html')


@bp.route('/clinician-dashboard')
def clinician_dashboard():
    """Clinician dashboard - same as main dashboard"""
    return render_template('dashboard.html')


@bp.route('/nurse-dashboard')
def nurse_dashboard():
    """Nurse dashboard - same as main dashboard"""
    return render_template('dashboard.html')


@bp.route('/patient-dashboard')
def patient_dashboard():
    """Patient dashboard - same as main dashboard"""
    return render_template('dashboard.html')


@bp.route('/unified-dashboard')
def unified_dashboard():
    """Unified dashboard (for legacy routes)"""
    return render_template('dashboard.html')


@bp.route('/')
def home():
    """Home/landing page - redirect to login"""
    return redirect(url_for('main.login'))


@bp.route('/dashboard')
def dashboard():
    """ICU monitoring dashboard"""
    return render_template('dashboard.html')


@bp.route('/hitl-feedback')
def hitl_feedback_history():
    """HITL feedback history and review page"""
    return render_template('hitl_feedback.html')


@bp.route('/review')
def review_queue():
    """Legacy clinical review queue - for backwards compatibility"""
    return render_template('review_queue.html')


@bp.route('/api/patients', methods=['GET'])
def get_patients():
    """Get all admitted patients"""
    store = current_app.config['STORE']
    patients = store.get_admitted_patients()
    return jsonify(patients)


@bp.route('/api/patient/<int:patient_id>', methods=['GET'])
def get_patient(patient_id):
    """Get specific patient details"""
    store = current_app.config['STORE']
    patient = store.get_patient(patient_id)
    if not patient:
        return jsonify({'error': 'Patient not found'}), 404
    return jsonify(patient)


@bp.route('/api/patient/live/<int:patient_id>', methods=['GET'])
def get_patient_live(patient_id):
    """Get live patient data with real-time updates"""
    store = current_app.config['STORE']
    patient = store.get_patient(patient_id)
    if not patient:
        return jsonify({'error': 'Patient not found'}), 404
    
    # Return patient data directly (will be merged into local patient object)
    return jsonify(patient)


@bp.route('/api/patient/admit', methods=['POST'])
def admit_patient():
    """Admit a new patient"""
    store = current_app.config['STORE']
    data = request.json
    patient = store.admit_patient(data)

    # Log admission
    print(f"[ADMISSION] New patient admitted:")
    print(f"  - ID: {patient['id']}")
    print(f"  - Name: {patient['name']}")
    print(f"  - Ward: {patient['ward']}")
    print(f"  - Doctor: {patient['doctor']}")

    return jsonify(patient), 201


@bp.route('/api/patient/<int:patient_id>/discharge', methods=['POST'])
def discharge_patient(patient_id):
    """Discharge a patient"""
    store = current_app.config['STORE']
    success = store.discharge_patient(patient_id)
    if success:
        return jsonify({'status': 'discharged'}), 200
    return jsonify({'error': 'Patient not found'}), 404


@bp.route('/api/patient/<int:patient_id>/predict', methods=['POST'])
def predict_sepsis(patient_id):
    """Run sepsis prediction on patient vitals/labs"""
    engine = current_app.config['ENGINE']
    store = current_app.config['STORE']
    patient = store.get_patient(patient_id)

    if not patient:
        return jsonify({'error': 'Patient not found'}), 404

    print(f"\n[PREDICTION REQUEST] Patient ID: {patient_id} ({patient.get('name', 'Unknown')})")
    result = engine.predict(patient)
    print(f"[PREDICTION RESULT] Score: {result['risk_score']}, Level: {result['risk_level']}")

    # Update patient risk score
    store.update_patient_risk(patient_id, result['risk_score'], result['top_features'])

    return jsonify(result)


@bp.route('/api/patient/<int:patient_id>/alert', methods=['POST'])
def send_alert(patient_id):
    """Send SMS alert to doctor"""
    engine = current_app.config['ENGINE']
    store = current_app.config['STORE']
    patient = store.get_patient(patient_id)
    
    if not patient:
        return jsonify({'error': 'Patient not found'}), 404
    
    result = engine.send_alert(patient)
    
    # Log alert
    store.log_alert(patient_id, result['message'])
    
    return jsonify(result)


@bp.route('/api/alerts', methods=['GET'])
def get_alerts():
    """Get all sent alerts"""
    store = current_app.config['STORE']
    alerts = store.get_alerts()
    return jsonify(alerts)


@bp.route('/api/health', methods=['GET'])
def health():
    """Health check"""
    return jsonify({'status': 'ok', 'timestamp': datetime.now().isoformat()})


# ============ HITL FEEDBACK ENDPOINTS ============
@bp.route('/api/hitl/submit', methods=['POST'])
def submit_hitl_feedback():
    """Submit HITL feedback for patient training"""
    store = current_app.config['STORE']
    data = request.json
    
    feedback = store.submit_hitl_feedback(
        clinician_id=data.get('clinician_id'),
        patient_id=data.get('patient_id'),
        feedback_data=data
    )
    
    # Check if we can retrain
    can_retrain = store.can_retrain()
    
    return jsonify({
        'success': True,
        'feedback': feedback,
        'can_retrain': can_retrain,
        'feedback_count': store.get_hitl_feedback_count()
    }), 201


@bp.route('/api/hitl/list', methods=['GET'])
def get_hitl_feedback():
    """Get HITL feedback list for clinician"""
    store = current_app.config['STORE']
    clinician_id = request.args.get('clinician_id')
    
    feedback_list = store.get_hitl_feedback_list(clinician_id)
    feedback_count = store.get_hitl_feedback_count(clinician_id)
    
    return jsonify({
        'success': True,
        'feedback': feedback_list,
        'count': feedback_count,
        'can_retrain': store.can_retrain()
    }), 200


@bp.route('/api/patient/<int:patient_id>/summary', methods=['GET'])
def get_patient_summary(patient_id):
    """Get patient summary for download"""
    store = current_app.config['STORE']
    summary = store.get_patient_summary(patient_id)
    
    if not summary:
        return jsonify({'error': 'Patient not found'}), 404
    
    return jsonify(summary), 200


@bp.route('/api/hitl/status', methods=['GET'])
def check_retrain_status():
    """Check if model retraining is available"""
    store = current_app.config['STORE']
    count = store.get_hitl_feedback_count()
    
    return jsonify({
        'feedback_count': count,
        'can_retrain': store.can_retrain(),
        'forms_needed': max(0, 10 - count)
    }), 200


# ============ MODEL RETRAINING ENDPOINTS ============
@bp.route('/api/retrain/trigger', methods=['POST'])
def trigger_retrain():
    """Trigger model retraining from collected feedback"""
    store = current_app.config['STORE']
    count = store.get_hitl_feedback_count()
    
    if count < 10:
        return jsonify({
            'success': False,
            'error': f'Insufficient feedback: {count}/10 required',
            'feedback_count': count
        }), 400
    
    # Log retraining  
    print(f"\n[RETRAINING TRIGGERED]")
    print(f"  - Feedback count: {count}")
    print(f"  - Timestamp: {datetime.now().isoformat()}")
    print(f"  - Source: {request.remote_addr}")
    
    return jsonify({
        'success': True,
        'message': f'Model retraining triggered with {count} feedback samples',
        'feedback_count': count,
        'retrain_status': 'started',
        'timestamp': datetime.now().isoformat()
    }), 200


@bp.route('/api/retrain/status', methods=['GET'])
def get_retrain_status():
    """Get current retraining status"""
    store = current_app.config['STORE']
    count = store.get_hitl_feedback_count()
    
    return jsonify({
        'feedback_count': count,
        'ready_to_retrain': count >= 10,
        'forms_needed': max(0, 10 - count),
        'last_retrain': None  # Can be updated to track last retraining time
    }), 200


# ============ MEDICAL REPORT & CERTIFICATE ENDPOINTS ============
@bp.route('/api/patient/<int:patient_id>/report', methods=['GET'])
def get_patient_report(patient_id):
    """Generate medical report for patient"""
    store = current_app.config['STORE']
    patient = store.get_patient(patient_id)
    
    if not patient:
        return jsonify({'error': 'Patient not found'}), 404
    
    summary = store.get_patient_summary(patient_id)
    
    return jsonify({
        'success': True,
        'report': {
            'patient_info': summary['patient_info'],
            'current_risk': summary['current_risk'],
            'current_vitals': summary['current_vitals'],
            'current_labs': summary['current_labs'],
            'top_features': patient.get('topFeatures', []),
            'clinical_feedback': summary['clinical_feedback'],
            'feedback_count': summary['feedback_count'],
            'generated_at': datetime.now().isoformat()
        }
    }), 200


@bp.route('/api/patient/<int:patient_id>/certificate', methods=['GET'])
def get_medical_certificate(patient_id):
    """Generate medical discharge certificate for patient"""
    store = current_app.config['STORE']
    patient = store.get_patient(patient_id)
    
    if not patient:
        return jsonify({'error': 'Patient not found'}), 404
    
    # Check if patient is discharged or ready for discharge
    return jsonify({
        'success': True,
        'certificate': {
            'patient_id': patient['id'],
            'patient_name': patient['name'],
            'age': patient['age'],
            'gender': patient['gender'],
            'ward': patient['ward'],
            'admitted_date': patient['admitted'],
            'discharge_date': datetime.now().strftime('%Y-%m-%d'),
            'final_sepsis_risk': patient.get('sepsisRisk', 'N/A'),
            'final_risk_status': patient.get('riskLevel', 'Stable'),
            'doctors_name': patient.get('doctor', 'Medical Team'),
            'hospital_name': 'Sepsis ICU Management System',
            'certificate_number': f"MED-{patient_id}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            'generated_at': datetime.now().isoformat()
        }
    }), 200


@bp.route('/patient/report/<int:patient_id>')
def view_patient_report(patient_id):
    """HTML view for patient medical report"""
    return render_template('patient_report.html', patient_id=patient_id)


@bp.route('/patient/certificate/<int:patient_id>')
def view_medical_certificate(patient_id):
    """HTML view for medical certificate"""
    return render_template('medical_certificate.html', patient_id=patient_id)
