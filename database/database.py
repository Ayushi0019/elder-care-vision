"""
database/database.py — SQLite Database Manager
================================================
PURPOSE:
    Store event logs, activity history, and system status.

PRIVACY DESIGN (CORE REQUIREMENT):
    This database NEVER stores:
    - Raw video frames
    - Face images
    - Personal identifying information
    - Continuous video stream

    It ONLY stores:
    - Timestamps
    - Activity labels (text)
    - Confidence scores (numbers)
    - Event types (text)
    - Feature values (numbers)
    - Alert status (boolean/text)

    This is the privacy-preserving design described in the synopsis.

TABLES:
    events:
        Records significant events (falls, alerts)
        
    activity_logs:
        Records activity classification every N frames
        
    system_status:
        Records camera and privacy status changes
"""

import sqlite3
import os
import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
import config

logger = logging.getLogger(__name__)


class Database:
    """
    SQLite database manager for the elder care monitoring system.

    Provides methods to insert and query:
    - Fall/emergency events
    - Activity log entries
    - System status updates

    NEVER stores raw video or images.
    """

    def __init__(self, db_path: str = config.DATABASE_PATH):
        self.db_path = db_path

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        self.conn: Optional[sqlite3.Connection] = None
        self._connect()
        self._create_tables()

        logger.info(f"Database initialized at: {db_path}")

    def _connect(self) -> None:
        """Open SQLite connection."""
        try:
            self.conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,  # Allow multi-thread access
            )
            self.conn.row_factory = sqlite3.Row  # Rows as dict-like objects
            logger.debug(f"SQLite connected: {self.db_path}")
        except Exception as e:
            logger.error(f"Database connection error: {e}")
            raise

    def _create_tables(self) -> None:
        """Create all required tables if they don't exist."""
        cursor = self.conn.cursor()

        # --- events table: records significant events ---
        # PRIVACY NOTE: No video or image data stored here
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                event_id    TEXT PRIMARY KEY,
                timestamp   TEXT NOT NULL,
                event_type  TEXT NOT NULL,
                activity    TEXT,
                confidence  REAL,
                duration    REAL,
                irc_value   REAL,
                alert_sent  INTEGER DEFAULT 0,
                notes       TEXT
            )
        """)

        # --- activity_logs table: periodic activity records ---
        # Stores activity classification every N frames
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS activity_logs (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT NOT NULL,
                activity    TEXT NOT NULL,
                confidence  REAL,
                aspect_ratio REAL,
                kinetic_energy REAL
            )
        """)

        # --- system_status table: system health log ---
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS system_status (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT NOT NULL,
                camera_status   TEXT,
                privacy_status  TEXT,
                fps             REAL,
                notes           TEXT
            )
        """)

        self.conn.commit()
        logger.debug("Database tables created/verified.")

    # =========================================================================
    # EVENT OPERATIONS
    # =========================================================================

    def log_event(
        self,
        event_type: str,
        activity: str,
        confidence: float,
        duration: float = 0.0,
        irc_value: float = 0.0,
        alert_sent: bool = False,
        notes: str = "",
    ) -> str:
        """
        Log a significant event (e.g., fall detection).

        PRIVACY: No video data is stored. Only numerical/text metadata.

        Args:
            event_type:  Type of event ("FALL_DETECTED", "POTENTIAL_FALL", etc.)
            activity:    Activity at time of event
            confidence:  ML model confidence
            duration:    Duration of the event in seconds
            irc_value:   IRC value at time of event
            alert_sent:  True if alert was dispatched
            notes:       Optional description

        Returns:
            event_id string (timestamp-based unique ID)
        """
        # Generate a unique event ID using timestamp
        event_id = f"EVT_{int(time.time() * 1000)}"
        timestamp = datetime.now().isoformat()

        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT INTO events
                (event_id, timestamp, event_type, activity, confidence,
                 duration, irc_value, alert_sent, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id, timestamp, event_type, activity,
                    confidence, duration, irc_value,
                    1 if alert_sent else 0, notes
                )
            )
            self.conn.commit()
            logger.info(f"Event logged: {event_id} — {event_type}")
            return event_id
        except Exception as e:
            logger.error(f"Error logging event: {e}")
            return ""

    def log_activity(
        self,
        activity: str,
        confidence: float,
        aspect_ratio: float = 0.0,
        kinetic_energy: float = 0.0,
    ) -> None:
        """
        Log periodic activity classification.

        Called every N frames to maintain activity history.
        PRIVACY: Only stores text labels and numerical features.

        Args:
            activity:       Predicted activity label
            confidence:     Model confidence
            aspect_ratio:   Current aspect ratio feature
            kinetic_energy: Current kinetic energy proxy feature
        """
        timestamp = datetime.now().isoformat()
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT INTO activity_logs
                (timestamp, activity, confidence, aspect_ratio, kinetic_energy)
                VALUES (?, ?, ?, ?, ?)
                """,
                (timestamp, activity, confidence, aspect_ratio, kinetic_energy)
            )
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error logging activity: {e}")

    def log_system_status(
        self,
        camera_status: str,
        fps: float = 0.0,
        notes: str = "",
    ) -> None:
        """Log system status snapshot."""
        timestamp = datetime.now().isoformat()
        privacy_status = "RAW VIDEO STORAGE: DISABLED" if config.PRIVACY_MODE else "WARNING: Privacy mode off"
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                """
                INSERT INTO system_status
                (timestamp, camera_status, privacy_status, fps, notes)
                VALUES (?, ?, ?, ?, ?)
                """,
                (timestamp, camera_status, privacy_status, fps, notes)
            )
            self.conn.commit()
        except Exception as e:
            logger.error(f"Error logging status: {e}")

    # =========================================================================
    # QUERY OPERATIONS (for dashboard)
    # =========================================================================

    def get_recent_events(self, limit: int = config.MAX_EVENTS_IN_DASHBOARD) -> List[Dict]:
        """Return most recent events for dashboard display."""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT * FROM events ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error fetching events: {e}")
            return []

    def get_recent_activity_logs(self, limit: int = 100) -> List[Dict]:
        """Return recent activity history for dashboard."""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT * FROM activity_logs ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error fetching activity logs: {e}")
            return []

    def get_fall_events(self) -> List[Dict]:
        """Return only fall-related events."""
        try:
            cursor = self.conn.cursor()
            cursor.execute(
                "SELECT * FROM events WHERE event_type='FALL_DETECTED' ORDER BY timestamp DESC"
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error fetching fall events: {e}")
            return []

    def close(self) -> None:
        """Close database connection."""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed.")
