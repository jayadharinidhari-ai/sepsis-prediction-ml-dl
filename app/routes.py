from flask import Blueprint, render_template, request, jsonify, current_app
import json
from datetime import datetime

bp = Blueprint('main', __name__)


@bp.route('/')
def home():
    """Home/landing page"""
    return render_template('home.html')


@bp.route('/dashboard')
def dashboard():
    """ICU monitoring dashboard"""
    return render_template('dashboard.html')


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
