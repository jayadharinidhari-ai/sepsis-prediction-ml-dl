"""
Human-in-the-Loop Management System
Handles clinician review, feedback collection, and model retraining
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent.parent / "human_feedback.db"


class HumanLoopManager:
    """Manages human review queue and feedback collection"""
    
    def __init__(self, db_path: str = str(DB_PATH)):
        """Initialize database connection"""
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Create database tables if they don't exist"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Predictions table - stores all predictions
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                features JSON NOT NULL,
                lstm_score REAL NOT NULL,
                xgb_score REAL,
                ensemble_score REAL NOT NULL,
                model_type TEXT NOT NULL,
                risk_level TEXT NOT NULL,
                reviewed INTEGER DEFAULT 0,
                review_timestamp DATETIME,
                clinician_id TEXT,
                clinician_feedback TEXT,
                clinician_correct INTEGER,
                notes TEXT
            )
            """)
            
            # Review queue table - pending reviews
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS review_queue (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prediction_id INTEGER NOT NULL UNIQUE,
                patient_id TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                risk_score REAL NOT NULL,
                risk_level TEXT NOT NULL,
                priority INTEGER DEFAULT 0,
                assigned_to TEXT,
                status TEXT DEFAULT 'pending',
                FOREIGN KEY(prediction_id) REFERENCES predictions(id)
            )
            """)
            
            # Feedback summary table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS feedback_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                total_predictions INTEGER DEFAULT 0,
                total_reviewed INTEGER DEFAULT 0,
                accuracy REAL DEFAULT 0.0,
                last_updated DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            conn.commit()
            conn.close()
            logger.info("[OK] Human feedback database initialized")
        except Exception as e:
            logger.error(f"[ERROR] Database initialization failed: {e}")
    
    def add_prediction(self, patient_id: str, features: Dict, 
                      lstm_score: float, xgb_score: Optional[float],
                      ensemble_score: float, model_type: str, 
                      risk_level: str) -> int:
        """
        Add a new prediction to the system and add to review queue
        Returns prediction_id
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert prediction
            cursor.execute("""
            INSERT INTO predictions 
            (patient_id, features, lstm_score, xgb_score, ensemble_score, model_type, risk_level)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (patient_id, json.dumps(features), lstm_score, xgb_score, 
                  ensemble_score, model_type, risk_level))
            
            prediction_id = cursor.lastrowid
            
            # Add to review queue if risk level is HIGH or CRITICAL
            if risk_level in ["HIGH", "CRITICAL"]:
                priority = 2 if risk_level == "CRITICAL" else 1
                cursor.execute("""
                INSERT INTO review_queue 
                (prediction_id, patient_id, risk_score, risk_level, priority)
                VALUES (?, ?, ?, ?, ?)
                """, (prediction_id, patient_id, ensemble_score, risk_level, priority))
            
            conn.commit()
            conn.close()
            return prediction_id
        except Exception as e:
            logger.error(f"[ERROR] Failed to add prediction: {e}")
            return -1
    
    def get_review_queue(self, status: str = "pending", limit: int = 10) -> List[Dict]:
        """Get pending reviews, sorted by priority (highest first)"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
            SELECT rq.id, rq.prediction_id, rq.patient_id, rq.timestamp,
                   rq.risk_score, rq.risk_level, rq.priority, rq.assigned_to,
                   p.lstm_score, p.xgb_score, p.ensemble_score, p.model_type,
                   p.features
            FROM review_queue rq
            JOIN predictions p ON rq.prediction_id = p.id
            WHERE rq.status = ?
            ORDER BY rq.priority DESC, rq.timestamp ASC
            LIMIT ?
            """, (status, limit))
            
            rows = cursor.fetchall()
            conn.close()
            
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"[ERROR] Failed to get review queue: {e}")
            return []
    
    def approve_prediction(self, prediction_id: int, clinician_id: str, 
                          is_correct: bool, notes: str = "") -> bool:
        """
        Clinician approves/rejects a prediction
        is_correct: True if clinician agrees, False if they disagree
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Find review_queue entry
            cursor.execute("""
            SELECT id FROM review_queue WHERE prediction_id = ?
            """, (prediction_id,))
            
            result = cursor.fetchone()
            if not result:
                logger.warning(f"[WARN] No review queue entry for prediction {prediction_id}")
                conn.close()
                return False
            
            review_id = result[0]
            
            # Update prediction with feedback
            cursor.execute("""
            UPDATE predictions
            SET reviewed = 1, review_timestamp = ?, clinician_id = ?,
                clinician_correct = ?, notes = ?
            WHERE id = ?
            """, (datetime.now(), clinician_id, 1 if is_correct else 0, notes, prediction_id))
            
            # Update review queue status
            cursor.execute("""
            UPDATE review_queue
            SET status = 'reviewed', assigned_to = ?
            WHERE id = ?
            """, (clinician_id, review_id))
            
            conn.commit()
            conn.close()
            
            feedback_str = "✅ CORRECT" if is_correct else "❌ INCORRECT"
            logger.info(f"[OK] Prediction {prediction_id} marked as {feedback_str} by {clinician_id}")
            return True
        except Exception as e:
            logger.error(f"[ERROR] Failed to approve prediction: {e}")
            return False
    
    def get_feedback_statistics(self) -> Dict:
        """Get accuracy statistics from reviewed predictions"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Total predictions
            cursor.execute("SELECT COUNT(*) FROM predictions")
            total_predictions = cursor.fetchone()[0]
            
            # Reviewed predictions
            cursor.execute("SELECT COUNT(*) FROM predictions WHERE reviewed = 1")
            total_reviewed = cursor.fetchone()[0]
            
            # Accuracy (where clinician marked correct)
            if total_reviewed > 0:
                cursor.execute("""
                SELECT COUNT(*) FROM predictions 
                WHERE reviewed = 1 AND clinician_correct = 1
                """)
                correct_count = cursor.fetchone()[0]
                accuracy = (correct_count / total_reviewed) * 100
            else:
                accuracy = 0.0
            
            # Recent feedback
            cursor.execute("""
            SELECT clinician_correct, COUNT(*) as count
            FROM predictions WHERE reviewed = 1
            GROUP BY clinician_correct
            ORDER BY clinician_correct DESC
            """)
            
            feedback_breakdown = {row[0]: row[1] for row in cursor.fetchall()}
            
            conn.close()
            
            return {
                "total_predictions": total_predictions,
                "total_reviewed": total_reviewed,
                "review_percentage": (total_reviewed / total_predictions * 100) if total_predictions > 0 else 0,
                "accuracy": round(accuracy, 2),
                "correct": feedback_breakdown.get(1, 0),
                "incorrect": feedback_breakdown.get(0, 0)
            }
        except Exception as e:
            logger.error(f"[ERROR] Failed to get statistics: {e}")
            return {}
    
    def get_training_data(self, reviewed_only: bool = True) -> Tuple[List[Dict], List[int]]:
        """
        Get data for retraining models based on human feedback
        Returns (features_list, labels_list) where labels are clinician feedback
        """
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            if reviewed_only:
                cursor.execute("""
                SELECT features, clinician_correct
                FROM predictions WHERE reviewed = 1
                ORDER BY review_timestamp DESC
                """)
            else:
                cursor.execute("""
                SELECT features, clinician_correct
                FROM predictions WHERE clinician_correct IS NOT NULL
                """)
            
            rows = cursor.fetchall()
            conn.close()
            
            features_list = [json.loads(row[0]) for row in rows]
            labels_list = [row[1] for row in rows]
            
            return features_list, labels_list
        except Exception as e:
            logger.error(f"[ERROR] Failed to get training data: {e}")
            return [], []
    
    def get_prediction_history(self, patient_id: str, limit: int = 20) -> List[Dict]:
        """Get prediction history for a specific patient"""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute("""
            SELECT * FROM predictions
            WHERE patient_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
            """, (patient_id, limit))
            
            rows = cursor.fetchall()
            conn.close()
            
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"[ERROR] Failed to get prediction history: {e}")
            return []
    
    def dismiss_review(self, prediction_id: int, clinician_id: str) -> bool:
        """Dismiss a review without approving/rejecting (mark as reviewed but no feedback)"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
            SELECT id FROM review_queue WHERE prediction_id = ?
            """, (prediction_id,))
            
            result = cursor.fetchone()
            if not result:
                conn.close()
                return False
            
            review_id = result[0]
            
            cursor.execute("""
            UPDATE review_queue
            SET status = 'dismissed', assigned_to = ?
            WHERE id = ?
            """, (clinician_id, review_id))
            
            conn.commit()
            conn.close()
            
            logger.info(f"[OK] Review {prediction_id} dismissed by {clinician_id}")
            return True
        except Exception as e:
            logger.error(f"[ERROR] Failed to dismiss review: {e}")
            return False


# Global instance
_human_loop_manager = None


def get_human_loop_manager() -> HumanLoopManager:
    """Get or create global HumanLoopManager instance"""
    global _human_loop_manager
    if _human_loop_manager is None:
        _human_loop_manager = HumanLoopManager()
    return _human_loop_manager
