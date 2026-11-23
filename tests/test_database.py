"""
tests/test_database.py — Unit Tests for Database Module
========================================================
Tests verify:
1. Database creation and table setup
2. Logging events without raw video data
3. Querying events
4. Privacy: no raw frame columns exist
5. Activity log operations
6. System status logging
"""

import sys
import os
import unittest
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.database import Database


class TestDatabase(unittest.TestCase):
    """Tests for the SQLite database module."""

    def setUp(self):
        """Create a temporary database for each test."""
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "test.db")
        self.db = Database(db_path=self.db_path)

    def tearDown(self):
        """Close and remove temp database."""
        self.db.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_database_file_created(self):
        """Database file should be created on initialization."""
        self.assertTrue(
            os.path.exists(self.db_path),
            "Database file should be created."
        )

    def test_log_event_stores_metadata_only(self):
        """Event log must store only metadata, not image/video data."""
        event_id = self.db.log_event(
            event_type="FALL_DETECTED",
            activity="Fallen",
            confidence=0.87,
            duration=6.5,
            irc_value=0.31,
            alert_sent=True,
            notes="Test fall event",
        )
        self.assertNotEqual(event_id, "", "Event ID should not be empty.")

        # Verify it was stored
        events = self.db.get_recent_events(limit=5)
        self.assertGreater(len(events), 0, "Event should be stored.")

        stored = events[0]
        self.assertEqual(stored["event_type"], "FALL_DETECTED")
        self.assertAlmostEqual(stored["confidence"], 0.87, places=2)
        self.assertEqual(stored["alert_sent"], 1)

    def test_events_table_has_no_image_columns(self):
        """Events table must NOT have any image/video columns."""
        cursor = self.db.conn.cursor()
        cursor.execute("PRAGMA table_info(events)")
        columns = [row[1] for row in cursor.fetchall()]

        forbidden = ["image", "frame", "video", "rgb", "bgr", "pixel", "photo"]
        for col in columns:
            for bad in forbidden:
                self.assertNotIn(
                    bad, col.lower(),
                    f"Events table has suspicious column: {col}"
                )

    def test_activity_logs_table_has_no_image_columns(self):
        """Activity logs table must NOT have image/video columns."""
        cursor = self.db.conn.cursor()
        cursor.execute("PRAGMA table_info(activity_logs)")
        columns = [row[1] for row in cursor.fetchall()]
        forbidden = ["image", "frame", "video", "rgb", "pixel"]
        for col in columns:
            for bad in forbidden:
                self.assertNotIn(bad, col.lower(),
                                 f"activity_logs has suspicious column: {col}")

    def test_log_activity_and_retrieve(self):
        """Should be able to log and retrieve activity records."""
        self.db.log_activity(
            activity="Walking",
            confidence=0.92,
            aspect_ratio=0.38,
            kinetic_energy=0.025,
        )
        logs = self.db.get_recent_activity_logs(limit=10)
        self.assertGreater(len(logs), 0)
        self.assertEqual(logs[0]["activity"], "Walking")
        self.assertAlmostEqual(logs[0]["confidence"], 0.92, places=2)

    def test_get_fall_events_filters_correctly(self):
        """get_fall_events should return only FALL_DETECTED type events."""
        self.db.log_event("FALL_DETECTED", "Fallen", 0.90, 5.0)
        self.db.log_event("POTENTIAL_FALL", "Fallen", 0.70, 2.0)
        self.db.log_activity("Standing", 0.88)

        fall_events = self.db.get_fall_events()
        for event in fall_events:
            self.assertEqual(event["event_type"], "FALL_DETECTED",
                             "Should only return FALL_DETECTED events.")

    def test_multiple_events_ordered_by_time(self):
        """Events should be retrievable in descending time order."""
        for i in range(5):
            self.db.log_event(f"FALL_DETECTED", "Fallen", 0.80 + i * 0.02, float(i))

        events = self.db.get_recent_events(limit=10)
        # Most recent should be first (descending order)
        if len(events) >= 2:
            self.assertGreaterEqual(
                events[0]["timestamp"], events[1]["timestamp"],
                "Events should be ordered most-recent-first."
            )

    def test_system_status_logging(self):
        """System status should be logged without errors."""
        try:
            self.db.log_system_status(
                camera_status="ACTIVE",
                fps=28.5,
                notes="Test status"
            )
        except Exception as e:
            self.fail(f"log_system_status raised exception: {e}")

    def test_empty_database_returns_empty_lists(self):
        """Querying empty database should return empty lists, not errors."""
        events = self.db.get_recent_events()
        self.assertIsInstance(events, list)

        logs = self.db.get_recent_activity_logs()
        self.assertIsInstance(logs, list)

        falls = self.db.get_fall_events()
        self.assertIsInstance(falls, list)


class TestPrivacyDatabase(unittest.TestCase):
    """Specifically test privacy requirements on the database."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db = Database(db_path=os.path.join(self.test_dir, "test.db"))

    def tearDown(self):
        self.db.close()
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_all_tables_have_no_blob_columns(self):
        """No table should have BLOB type columns (would store binary/image data)."""
        cursor = self.db.conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]

        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            cols = cursor.fetchall()
            for col in cols:
                col_name = col[1]
                col_type = col[2].upper() if col[2] else ""
                self.assertNotEqual(
                    col_type, "BLOB",
                    f"Table '{table}' column '{col_name}' is BLOB type "
                    f"(could store binary/image data — privacy violation!)."
                )

    def test_event_confidence_is_numeric(self):
        """Confidence stored as a number, not as image."""
        self.db.log_event("FALL_DETECTED", "Fallen", 0.88, 3.0)
        events = self.db.get_recent_events()
        confidence = events[0]["confidence"]
        self.assertIsInstance(confidence, (int, float))


if __name__ == "__main__":
    print("Running Database Tests...")
    print("=" * 60)
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestDatabase))
    suite.addTests(loader.loadTestsFromTestCase(TestPrivacyDatabase))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if result.wasSuccessful():
        print("\n✅ All database tests passed!")
    else:
        print(f"\n❌ {len(result.failures)} test(s) failed.")
