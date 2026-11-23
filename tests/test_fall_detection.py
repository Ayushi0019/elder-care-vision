"""
tests/test_fall_detection.py — Unit Tests for Fall Detection
=============================================================
Run with: python -m pytest tests/ -v

Tests cover:
1. IRC threshold detection
2. Multi-signal fall voting
3. Temporal validator state machine
4. False positive scenarios (sitting, bending)
5. Recovery detection
6. Cooldown behavior
"""

import sys
import os
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from fall_detection.irc import IRCCalculator
from fall_detection.fall_detector import FallDetector
from fall_detection.temporal_validator import TemporalValidator, FallState


class TestIRCCalculator(unittest.TestCase):
    """Test the Instantaneous Rate of Change module."""

    def setUp(self):
        # Use small window and threshold for testing
        self.irc = IRCCalculator(window=3, threshold=0.10)

    def test_not_triggered_on_insufficient_history(self):
        """IRC should not trigger until buffer is full."""
        result = self.irc.update(0.5)
        self.assertFalse(result["above_threshold"])
        self.assertEqual(result["irc_value"], 0.0)

    def test_no_sudden_change_no_trigger(self):
        """Gradual change should not exceed threshold."""
        for val in [0.10, 0.11, 0.12, 0.13]:
            result = self.irc.update(val)
        # Change over 3 frames: |0.13 - 0.10| / 3 = 0.01 < 0.10 threshold
        self.assertFalse(result["above_threshold"],
                         "Gradual change should not trigger IRC.")

    def test_sudden_change_triggers(self):
        """Sudden large change should exceed threshold."""
        for val in [0.01, 0.01, 0.01]:
            result = self.irc.update(val)
        # Now sudden spike
        result = self.irc.update(0.80)
        # Change = (0.80 - 0.01) / 3 = 0.263 > 0.10 threshold
        self.assertTrue(result["above_threshold"],
                        "Sudden spike should trigger IRC.")

    def test_reset_clears_history(self):
        """Reset should clear all history."""
        for val in [0.1, 0.2, 0.3, 0.4]:
            self.irc.update(val)
        self.irc.reset()
        result = self.irc.update(0.5)
        self.assertEqual(result["history_length"], 1)
        self.assertFalse(result["above_threshold"])


class TestFallDetector(unittest.TestCase):
    """Test the multi-signal fall detector."""

    def setUp(self):
        self.detector = FallDetector()

    def _make_features(
        self,
        aspect_ratio=0.4,
        head_vy=0.01,
        head_accel=0.005,
        kinetic_energy=0.01,
        body_vy=0.01,
        movement_intensity=0.05,
        shoulder_vel=0.01,
        hip_vel=0.01,
        knee_vel=0.01,
    ):
        """Helper to create a feature vector."""
        return np.array([
            aspect_ratio, head_vy, head_accel,
            kinetic_energy, body_vy, movement_intensity,
            shoulder_vel, hip_vel, knee_vel,
        ], dtype=np.float32)

    def test_normal_standing_no_fall_signal(self):
        """Normal standing features should NOT trigger fall signal."""
        features = self._make_features(
            aspect_ratio=0.35,  # Tall = standing
            head_vy=0.001,       # Slow movement
            kinetic_energy=0.005,
        )
        result = self.detector.analyze("Standing", 0.90, features)
        self.assertFalse(result["fall_signal"],
                         "Standing person should not trigger fall signal.")

    def test_fallen_posture_with_ml_triggers(self):
        """ML saying 'Fallen' + high AR + high head velocity = fall signal."""
        features = self._make_features(
            aspect_ratio=1.5,   # Wide = horizontal (fallen)
            head_vy=0.10,       # Fast downward movement
            head_accel=0.05,    # Rapid acceleration
            kinetic_energy=0.08,
            body_vy=0.08,
        )
        result = self.detector.analyze("Fallen", 0.85, features)
        self.assertTrue(result["fall_signal"],
                        "Multiple fall signals should trigger detection.")
        self.assertGreaterEqual(result["vote_count"], 3)

    def test_sitting_does_not_trigger_fall(self):
        """Normal sitting should not create enough votes for fall."""
        features = self._make_features(
            aspect_ratio=0.55,  # Moderate (could be sitting)
            head_vy=0.005,
            head_accel=0.001,
            kinetic_energy=0.008,
        )
        result = self.detector.analyze("Sitting", 0.80, features)
        # Sitting: aspect ratio might be borderline but other signals low
        # Vote count should be low
        self.assertLess(result["vote_count"], 4,
                        "Normal sitting should have few fall votes.")

    def test_vote_count_bounds(self):
        """Vote count must be between 0 and total_votes."""
        features = self._make_features()
        result = self.detector.analyze("Standing", 0.7, features)
        self.assertGreaterEqual(result["vote_count"], 0)
        self.assertLessEqual(
            result["vote_count"], result["total_votes"]
        )

    def test_low_confidence_ml_still_can_trigger(self):
        """Low ML confidence + strong other signals can still vote."""
        features = self._make_features(
            aspect_ratio=1.8,
            head_vy=0.12,
            head_accel=0.08,
            kinetic_energy=0.10,
            body_vy=0.09,
        )
        # ML confidence < 0.5, so ML vote = False
        result = self.detector.analyze("Fallen", 0.45, features)
        # But other signals should still vote
        self.assertGreaterEqual(result["vote_count"], 2,
                                "Other signals should still vote even with low ML confidence.")


class TestTemporalValidator(unittest.TestCase):
    """Test the fall state machine."""

    def setUp(self):
        # Short times for faster testing
        self.validator = TemporalValidator(
            confirmation_seconds=0.3,
            recovery_window_seconds=0.5,
            cooldown_seconds=1.0,
            min_frames=2,
        )

    def test_starts_in_normal_state(self):
        """Validator must start in NORMAL state."""
        self.assertEqual(self.validator.get_state(), FallState.NORMAL)

    def test_no_fall_stays_normal(self):
        """No fall signal → stays in NORMAL."""
        for _ in range(10):
            result = self.validator.update(False, "Standing", 0.9)
        self.assertEqual(result["state"], "NORMAL")
        self.assertFalse(result["should_alert"])

    def test_persistent_fall_moves_to_potential(self):
        """Consecutive fall signals → POTENTIAL_FALL state."""
        # Send min_frames consecutive fall signals
        for _ in range(3):
            result = self.validator.update(True, "Fallen", 0.85)
        self.assertEqual(result["state"], "POTENTIAL_FALL",
                         "Should enter POTENTIAL_FALL after consecutive fall frames.")

    def test_recovery_resets_to_normal(self):
        """Recovery signal should reset to NORMAL from POTENTIAL_FALL."""
        # Enter potential fall
        for _ in range(3):
            self.validator.update(True, "Fallen", 0.85)

        # Recovery
        result = self.validator.update(False, "Standing", 0.90)
        self.assertEqual(result["state"], "NORMAL",
                         "Recovery should bring state back to NORMAL.")

    def test_confirmed_fall_sends_alert(self):
        """Fall persisting past confirmation time → should_alert = True."""
        # Enter potential fall
        for _ in range(3):
            self.validator.update(True, "Fallen", 0.85)

        # Wait longer than confirmation_seconds (0.3s)
        time.sleep(0.35)

        # Next fall update should confirm
        result = self.validator.update(True, "Fallen", 0.85)

        # It should now be in FALLEN_CONFIRMED → next update triggers alert
        alert_result = self.validator.update(True, "Fallen", 0.85)
        self.assertTrue(
            result["state"] in ["FALLEN_CONFIRMED", "ALERTING"]
            or alert_result.get("should_alert", False),
            "Fall should be confirmed after timeout."
        )

    def test_force_reset(self):
        """Force reset should return to NORMAL immediately."""
        for _ in range(3):
            self.validator.update(True, "Fallen", 0.85)
        self.validator.force_reset()
        self.assertEqual(self.validator.get_state(), FallState.NORMAL)

    def test_consecutive_frames_counter(self):
        """Consecutive frame counter should increment on each fall signal."""
        result1 = self.validator.update(True, "Fallen", 0.85)
        result2 = self.validator.update(True, "Fallen", 0.85)
        self.assertGreaterEqual(
            result2["consecutive_frames"],
            result1["consecutive_frames"],
        )

    def test_normal_sitting_not_confused_with_fall(self):
        """
        A single fall frame followed by normal frames should not alert.
        Tests false-positive reduction.
        """
        self.validator.update(True, "Fallen", 0.70)   # One fall frame
        result = self.validator.update(False, "Sitting", 0.85)  # Recovery
        self.assertEqual(result["state"], "NORMAL",
                         "Single fall frame + recovery should stay NORMAL.")


class TestPrivacyVerification(unittest.TestCase):
    """Verify privacy constraints are enforced at module boundaries."""

    def test_fall_detector_does_not_store_frames(self):
        """FallDetector should not have any image/frame storage attributes."""
        detector = FallDetector()
        # Check that no attribute names suggest image storage
        image_attrs = [
            attr for attr in dir(detector)
            if any(word in attr.lower() for word in
                   ["frame", "image", "video", "rgb", "bgr", "pixel"])
        ]
        self.assertEqual(image_attrs, [],
                         f"FallDetector has suspicious image attributes: {image_attrs}")

    def test_irc_stores_only_numbers(self):
        """IRCCalculator should only store numerical values, not frames."""
        irc = IRCCalculator(window=3, threshold=0.25)
        irc.update(0.5)
        irc.update(0.6)

        for item in irc.intensity_history:
            self.assertIsInstance(
                item, (int, float),
                f"IRC history should contain only numbers, got: {type(item)}"
            )


if __name__ == "__main__":
    print("Running Fall Detection Tests...")
    print("=" * 60)
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestIRCCalculator))
    suite.addTests(loader.loadTestsFromTestCase(TestFallDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestTemporalValidator))
    suite.addTests(loader.loadTestsFromTestCase(TestPrivacyVerification))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if result.wasSuccessful():
        print("\n✅ All fall detection tests passed!")
    else:
        print(f"\n❌ {len(result.failures)} test(s) failed.")
