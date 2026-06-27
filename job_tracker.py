import sqlite3
import uuid
import time
import json
from structured_parser import get_db_connection


def init_job_table():
    """
    Creates the job_status table if it doesn't exist.
    Called once at app startup.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS job_status (
            job_id TEXT PRIMARY KEY,
            filename TEXT,
            status TEXT,
            progress_current INTEGER,
            progress_total INTEGER,
            stage TEXT,
            result TEXT,
            error TEXT,
            created_at REAL,
            updated_at REAL
        )
    """)
    conn.commit()
    conn.close()


def create_job(filename: str) -> str:
    """
    Creates a new job entry. Returns the job_id immediately.
    Status starts as 'queued' — the actual work hasn't started yet.
    """
    job_id = str(uuid.uuid4())
    now = time.time()

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO job_status
        (job_id, filename, status, progress_current, progress_total, stage, result, error, created_at, updated_at)
        VALUES (?, ?, 'queued', 0, 0, 'queued', NULL, NULL, ?, ?)
    """, (job_id, filename, now, now))
    conn.commit()
    conn.close()

    return job_id


def update_job(job_id: str, status: str = None, current: int = None,
                total: int = None, stage: str = None,
                result: dict = None, error: str = None):
    """
    Updates a job's progress. Called repeatedly during ingestion
    (e.g. once per N pages processed) so polling /status returns
    fresh numbers, not a frozen snapshot.

    Only updates fields that are explicitly passed — None means
    "don't touch this field".
    """
    conn = get_db_connection()
    cursor = conn.cursor()

    fields = []
    values = []

    if status is not None:
        fields.append("status = ?")
        values.append(status)
    if current is not None:
        fields.append("progress_current = ?")
        values.append(current)
    if total is not None:
        fields.append("progress_total = ?")
        values.append(total)
    if stage is not None:
        fields.append("stage = ?")
        values.append(stage)
    if result is not None:
        fields.append("result = ?")
        values.append(json.dumps(result))
    if error is not None:
        fields.append("error = ?")
        values.append(error)

    fields.append("updated_at = ?")
    values.append(time.time())
    values.append(job_id)

    cursor.execute(f"UPDATE job_status SET {', '.join(fields)} WHERE job_id = ?", values)
    conn.commit()
    conn.close()


def get_job(job_id: str) -> dict | None:
    """
    Returns the current state of a job.
    This is what the /status/{job_id} endpoint calls on every poll.
    """
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM job_status WHERE job_id = ?", (job_id,))
    row = cursor.fetchone()

    if row is None:
        conn.close()
        return None

    columns = [desc[0] for desc in cursor.description]
    job = dict(zip(columns, row))
    conn.close()

    # Parse the JSON result field back into a dict
    if job.get("result"):
        job["result"] = json.loads(job["result"])

    # Compute a percentage for the frontend progress bar
    if job["progress_total"] and job["progress_total"] > 0:
        job["progress_percent"] = round(
            (job["progress_current"] / job["progress_total"]) * 100, 1
        )
    else:
        job["progress_percent"] = 0

    return job