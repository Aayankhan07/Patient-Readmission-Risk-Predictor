import os
import sqlite3
import hashlib
from datetime import datetime

AUDIT_DB_PATH = os.environ.get("AUDIT_DB_PATH", "data/audit_logs.db")


def init_audit_db():
    """Initializes the SQLite audit database and creates the logs table if it doesn't exist."""
    # Ensure parent directory exists
    os.makedirs(os.path.dirname(AUDIT_DB_PATH), exist_ok=True)

    conn = sqlite3.connect(AUDIT_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            patient_id_hash TEXT NOT NULL,
            risk_score REAL NOT NULL,
            risk_tier TEXT NOT NULL,
            model_version TEXT NOT NULL,
            inference_ms REAL NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def log_prediction_audit(
    patient_id: str,
    risk_score: float,
    risk_tier: str,
    model_version: str,
    inference_ms: float,
):
    """Securely logs a hashed prediction transaction for compliance audit trails."""
    try:
        init_audit_db()

        # Hash patient ID to prevent PII exposure in the database
        patient_hash = hashlib.sha256(patient_id.encode()).hexdigest()
        timestamp = datetime.utcnow().isoformat()

        conn = sqlite3.connect(AUDIT_DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO audit_logs (timestamp, patient_id_hash, risk_score, risk_tier, model_version, inference_ms)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                timestamp,
                patient_hash,
                risk_score,
                risk_tier,
                model_version,
                inference_ms,
            ),
        )
        conn.commit()
        conn.close()
        print(f"Audit log recorded for patient hash: {patient_hash[:12]}")
    except Exception as e:
        print(f"WARNING: Audit logging failed: {e}")
