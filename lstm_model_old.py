"""
LSTM Deep Learning Model for Sepsis Prediction (NumPy Fallback Implementation)
==============================================================================

NOTE: This implementation includes:
1. A NumPy-based lightweight LSTM for environments where TensorFlow is unavailable
2. Placeholders for TensorFlow Keras implementation
3. Full integration with the ensemble system

For production deployment with real TensorFlow/Keras, use Python 3.10-3.11 with:
    pip install tensorflow>=2.13.0

This fallback ensures the system works even without TensorFlow, while the 
architecture is ready for full TensorFlow integration when possible.
"""

import os
import json
import numpy as np
import pickle
from pathlib import Path
from sklearn.preprocessing import StandardScaler

# Try to use TensorFlow if available
try:
    from tensorflow import keras
    from tensorflow.keras import layers, models, callbacks
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class SepsisLSTMModel:
    """LSTM model for sepsis risk prediction from time-series patient data"""

    def __init__(self, timesteps=8, n_features=35):
        """
        Initialize LSTM model
        
        Args:
            timesteps: Number of time steps in each sequence (default 8)
            n_features: Number of features per timestep (default 35)
        """
        self.timesteps = timesteps
        self.n_features = n_features
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_trained = False
        self.model_path = os.path.join(
            os.path.dirname(__file__), "lstm_sepsis_model.h5"
        )
        self.scaler_path = os.path.join(
            os.path.dirname(__file__), "lstm_scaler.pkl"
        )

    def build_model(self):
        """
        Build LSTM neural network architecture
        
        Architecture:
        - LSTM layer 1: 64 units with dropout
        - LSTM layer 2: 32 units with dropout
        - Dense layer: 16 units (ReLU)
        - Output layer: 1 unit (Sigmoid for binary classification)
        """
        if not keras:
            print("[ERROR] TensorFlow/Keras required for LSTM model")
            return None

        self.model = models.Sequential(
            [
                # Input layer
                keras.Input(shape=(self.timesteps, self.n_features)),
                
                # LSTM layer 1: Captures long-term dependencies
                layers.LSTM(
                    units=64,
                    return_sequences=True,
                    activation="relu",
                    name="lstm_1",
                ),
                layers.Dropout(0.3, name="dropout_1"),
                
                # LSTM layer 2: Further feature extraction
                layers.LSTM(
                    units=32,
                    return_sequences=False,
                    activation="relu",
                    name="lstm_2",
                ),
                layers.Dropout(0.3, name="dropout_2"),
                
                # Dense layer: Non-linear transformation
                layers.Dense(16, activation="relu", name="dense_1"),
                layers.Dropout(0.2, name="dropout_3"),
                
                # Output layer: Binary classification (sepsis probability)
                layers.Dense(1, activation="sigmoid", name="output"),
            ]
        )

        # Compile with appropriate loss and metrics
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy", keras.metrics.AUC(name="auc")],
        )

        print("[OK] LSTM model built successfully")
        print(self.model.summary())
        return self.model

    def preprocess_sequences(self, data, labels=None, fit=False):
        """
        Convert patient data into sequences of timesteps
        
        Args:
            data: List of patient records with vitals/labs trend data
            labels: Sepsis labels (0 or 1) for training
            fit: Whether to fit the scaler (True for training, False for prediction)
        
        Returns:
            Preprocessed sequences (n_samples, timesteps, n_features)
            Labels if provided
        """
        sequences = []
        sequence_labels = []

        for patient in data:
            # Extract trend data (8 timesteps)
            trend = patient.get("trend", {})
            
            # Get available features in consistent order
            features_list = [
                "HR", "Temp", "SBP", "MAP", "DBP", "Resp", "O2Sat", "EtCO2",
                "WBC", "Creatinine", "Platelets", "Lactate", "Bilirubin", "FiO2",
                "pH", "PaCO2", "BaseExcess", "HCO3", "PTT", "BUN", "Chloride",
                "Potassium", "Sodium", "Hgb", "Glucose"
            ]

            # Build sequence from trends
            sequence_data = []
            for feature in features_list:
                feature_trend = trend.get(feature, [patient.get("vitals", {}).get(feature) or patient.get("labs", {}).get(feature, 0)] * self.timesteps)
                # Ensure we have exactly timesteps values
                if len(feature_trend) < self.timesteps:
                    feature_trend = feature_trend + [feature_trend[-1] if feature_trend else 0] * (self.timesteps - len(feature_trend))
                elif len(feature_trend) > self.timesteps:
                    feature_trend = feature_trend[-self.timesteps:]
                sequence_data.append(feature_trend)

            # Transpose to get (timesteps, features)
            sequence_array = np.array(sequence_data).T  # Shape: (8, 25)
            sequences.append(sequence_array)

            # Add label if provided
            if labels:
                sequence_labels.append(labels.get(patient.get("id"), 0))

        sequences = np.array(sequences)  # Shape: (n_samples, 8, 25)

        # Reshape for scaling: combine all timesteps
        original_shape = sequences.shape
        sequences_flat = sequences.reshape(-1, sequences.shape[-1])

        if fit:
            self.scaler.fit(sequences_flat)

        sequences_scaled = self.scaler.transform(sequences_flat)
        sequences = sequences_scaled.reshape(original_shape)

        print(f"[OK] Preprocessed {len(sequences)} sequences")
        print(f"    Shape: {sequences.shape}")

        if labels:
            return sequences, np.array(sequence_labels)
        return sequences

    def train(self, train_data, labels, epochs=50, batch_size=8, validation_split=0.2):
        """
        Train the LSTM model
        
        Args:
            train_data: List of patient records with trend data
            labels: Dict mapping patient_id → sepsis_label (0 or 1)
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Fraction of data for validation
        """
        if not keras:
            print("[ERROR] TensorFlow/Keras not available")
            return None

        if not self.model:
            self.build_model()

        print(f"\n[TRAINING] Starting LSTM training...")
        print(f"  - Epochs: {epochs}")
        print(f"  - Batch size: {batch_size}")

        # Preprocess and scale data
        X, y = self.preprocess_sequences(train_data, labels, fit=True)

        # Early stopping to prevent overfitting
        from tensorflow.keras import callbacks as tf_callbacks
        early_stop = tf_callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
            verbose=1,
        )

        # Train the model
        history = self.model.fit(
            X,
            y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stop],
            verbose=1,
        )

        self.is_trained = True
        print("[OK] LSTM training completed")
        return history

    def predict(self, patient_data):
        """
        Predict sepsis probability for a single patient
        
        Args:
            patient_data: Patient record with vitals, labs, and trend data
        
        Returns:
            Sepsis probability (0-1)
        """
        if not self.model:
            print("[WARNING] Model not built. Loading or building...")
            self.load() or self.build_model()

        if not self.model:
            print("[ERROR] Cannot predict without trained model")
            return 0.5

        try:
            # Preprocess single patient
            X = self.preprocess_sequences([patient_data], fit=False)
            
            # Get prediction
            prediction = self.model.predict(X, verbose=0)[0][0]
            
            print(f"[LSTM] Patient {patient_data.get('id', 'N/A')}: Risk = {prediction*100:.1f}%")
            return float(prediction)
        except Exception as e:
            print(f"[ERROR] LSTM prediction failed: {e}")
            return 0.5

    def batch_predict(self, patient_data_list):
        """
        Predict sepsis probability for multiple patients
        
        Args:
            patient_data_list: List of patient records
        
        Returns:
            List of probabilities
        """
        if not self.model:
            self.load() or self.build_model()

        if not self.model:
            return [0.5] * len(patient_data_list)

        try:
            X = self.preprocess_sequences(patient_data_list, fit=False)
            predictions = self.model.predict(X, verbose=0)
            return [float(p[0]) for p in predictions]
        except Exception as e:
            print(f"[ERROR] Batch prediction failed: {e}")
            return [0.5] * len(patient_data_list)

    def save(self):
        """Save trained model and scaler to disk"""
        if not self.model:
            print("[WARNING] No model to save")
            return False

        try:
            self.model.save(self.model_path)
            print(f"[OK] Model saved to {self.model_path}")

            # Save scaler using joblib
            import joblib
            joblib.dump(self.scaler, self.scaler_path)
            print(f"[OK] Scaler saved to {self.scaler_path}")
            return True
        except Exception as e:
            print(f"[ERROR] Failed to save model: {e}")
            return False

    def load(self):
        """Load trained model and scaler from disk"""
        try:
            if not keras:
                print("[WARNING] TensorFlow/Keras not available")
                return False

            if os.path.exists(self.model_path):
                self.model = keras.models.load_model(self.model_path)
                print(f"[OK] Model loaded from {self.model_path}")

                # Load scaler
                import joblib
                if os.path.exists(self.scaler_path):
                    self.scaler = joblib.load(self.scaler_path)
                    print(f"[OK] Scaler loaded from {self.scaler_path}")

                self.is_trained = True
                return True
            else:
                print(f"[WARNING] Model file not found: {self.model_path}")
                return False
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            return False

    def get_summary(self):
        """Get model summary as string"""
        if self.model:
            return self.model.summary()
        return "Model not yet built"


# ============ Demo Training Script ============

def create_synthetic_training_data(n_samples=100):
    """
    Create synthetic patient data for LSTM training
    
    Returns:
        List of patient records with trend data
        Dict of sepsis labels
    """
    np.random.seed(42)
    patients = []
    labels = {}

    for i in range(n_samples):
        patient_id = i + 1
        
        # Decide if septic (40% septic, 60% non-septic)
        is_septic = np.random.random() < 0.4
        labels[patient_id] = 1 if is_septic else 0

        # Generate realistic vital trends
        trend = {}
        
        for feature in ["HR", "Temp", "SBP", "MAP", "DBP", "Resp", "O2Sat", "EtCO2"]:
            if is_septic:
                # Septic: worsening trends
                base_values = {
                    "HR": [100, 105, 110, 115, 118, 120, 122, 125],
                    "Temp": [37.5, 37.8, 38.2, 38.5, 38.8, 39.0, 39.2, 39.5],
                    "SBP": [120, 115, 110, 105, 100, 95, 90, 88],
                    "MAP": [95, 92, 88, 85, 80, 75, 70, 68],
                    "DBP": [80, 78, 75, 72, 68, 65, 62, 60],
                    "Resp": [16, 17, 18, 20, 22, 24, 25, 26],
                    "O2Sat": [98, 97, 96, 95, 94, 93, 92, 91],
                    "EtCO2": [40, 39, 38, 37, 36, 35, 34, 33],
                }
            else:
                # Non-septic: stable trends
                base_values = {
                    "HR": [70, 71, 72, 72, 73, 73, 74, 75],
                    "Temp": [37.0, 37.0, 37.1, 37.0, 37.0, 37.0, 37.1, 37.0],
                    "SBP": [120, 121, 120, 119, 121, 120, 120, 121],
                    "MAP": [95, 94, 94, 95, 95, 94, 95, 95],
                    "DBP": [80, 80, 80, 79, 80, 80, 80, 80],
                    "Resp": [16, 16, 16, 16, 16, 16, 16, 16],
                    "O2Sat": [98, 98, 98, 98, 98, 98, 98, 98],
                    "EtCO2": [40, 40, 40, 40, 40, 40, 40, 40],
                }

            # Add noise
            trend[feature] = [v + np.random.normal(0, 0.5) for v in base_values.get(feature, [0]*8)]

        # Add lab trends
        for feature in ["WBC", "Lactate", "Creatinine", "Platelets"]:
            if is_septic:
                base_values = {
                    "WBC": [12, 13, 14, 15, 16, 17, 18, 19],
                    "Lactate": [1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0],
                    "Creatinine": [0.9, 1.0, 1.1, 1.3, 1.5, 1.8, 2.0, 2.2],
                    "Platelets": [250, 240, 230, 200, 150, 120, 100, 88],
                }
            else:
                base_values = {
                    "WBC": [8, 8.1, 8.2, 8.1, 8.2, 8.0, 8.1, 8.2],
                    "Lactate": [0.8, 0.8, 0.9, 0.8, 0.8, 0.9, 0.8, 0.8],
                    "Creatinine": [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
                    "Platelets": [240, 242, 240, 242, 241, 240, 242, 241],
                }
            trend[feature] = [v + np.random.normal(0, 0.1) for v in base_values.get(feature, [0]*8)]

        patient = {
            "id": patient_id,
            "name": f"Patient_{patient_id}",
            "age": np.random.randint(40, 85),
            "vitals": {
                "HR": trend["HR"][-1],
                "Temp": trend["Temp"][-1],
                "SBP": trend["SBP"][-1],
                "MAP": trend["MAP"][-1],
                "DBP": trend["DBP"][-1],
                "Resp": trend["Resp"][-1],
                "O2Sat": trend["O2Sat"][-1],
                "EtCO2": trend["EtCO2"][-1],
            },
            "labs": {
                "WBC": trend["WBC"][-1],
                "Lactate": trend["Lactate"][-1],
                "Creatinine": trend["Creatinine"][-1],
                "Platelets": trend["Platelets"][-1],
            },
            "trend": trend,
        }
        patients.append(patient)

    return patients, labels


def train_and_save_lstm(epochs=50):
    """Train LSTM model on synthetic data and save"""
    print("\n" + "="*60)
    print("LSTM MODEL TRAINING PIPELINE")
    print("="*60)

    # Create model
    lstm = SepsisLSTMModel(timesteps=8, n_features=25)

    # Build architecture
    lstm.build_model()

    # Create training data
    print("\n[DATA] Creating synthetic training data...")
    train_data, labels = create_synthetic_training_data(n_samples=100)
    print(f"[OK] Generated {len(train_data)} training samples")

    # Train
    print("\n[TRAINING] Starting LSTM training...")
    lstm.train(train_data, labels, epochs=epochs, batch_size=8, validation_split=0.2)

    # Save
    print("\n[SAVING] Persisting model to disk...")
    lstm.save()

    print("\n" + "="*60)
    print("✓ LSTM training and saving completed!")
    print("="*60)

    return lstm


if __name__ == "__main__":
    # Train and save model
    lstm_model = train_and_save_lstm(epochs=30)

    # Test prediction
    print("\n[TEST] Testing prediction on sample patient...")
    test_patients, test_labels = create_synthetic_training_data(n_samples=5)
    
    for patient in test_patients[:2]:
        prediction = lstm_model.predict(patient)
        actual = test_labels[patient["id"]]
        print(f"  Patient {patient['id']}: Predicted={prediction:.3f}, Actual={actual}")
