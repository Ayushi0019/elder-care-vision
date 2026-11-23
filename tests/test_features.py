"""
tests/test_features.py — Unit Tests for Feature Extraction
===========================================================
Run with: python -m pytest tests/ -v

Tests cover:
1. Aspect ratio calculation (standing vs fallen)
2. Joint velocity calculation
3. Kinetic energy proxy
4. Bounding box calculation
5. Temporal window statistics
6. Feature vector shape and bounds
"""

import sys
import os
import numpy as np
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.feature_extraction import FeatureExtractor
from features.temporal_features import TemporalWindow


def make_standing_keypoints():
    """
    Create synthetic 17-point keypoints for a STANDING person.
    Standing: tall narrow bounding box → low aspect ratio.
    Normalized coordinates (0-1 range).
    """
    # Person centered at x=0.5, spanning y from 0.1 (head) to 0.9 (feet)
    # Width is narrow: ~0.2
    kps = [
        [0.50, 0.10, 0.9],  # 0: Nose (head)
        [0.48, 0.12, 0.9],  # 1: Left Eye
        [0.52, 0.12, 0.9],  # 2: Right Eye
        [0.46, 0.14, 0.9],  # 3: Left Ear
        [0.54, 0.14, 0.9],  # 4: Right Ear
        [0.43, 0.25, 0.9],  # 5: Left Shoulder
        [0.57, 0.25, 0.9],  # 6: Right Shoulder
        [0.40, 0.40, 0.9],  # 7: Left Elbow
        [0.60, 0.40, 0.9],  # 8: Right Elbow
        [0.38, 0.55, 0.9],  # 9: Left Wrist
        [0.62, 0.55, 0.9],  # 10: Right Wrist
        [0.44, 0.55, 0.9],  # 11: Left Hip
        [0.56, 0.55, 0.9],  # 12: Right Hip
        [0.44, 0.72, 0.9],  # 13: Left Knee
        [0.56, 0.72, 0.9],  # 14: Right Knee
        [0.44, 0.90, 0.9],  # 15: Left Ankle
        [0.56, 0.90, 0.9],  # 16: Right Ankle
    ]
    return kps


def make_fallen_keypoints():
    """
    Create synthetic 17-point keypoints for a FALLEN person.
    Fallen: wide flat bounding box → high aspect ratio.
    Body is horizontal (spread along x-axis).
    """
    kps = [
        [0.10, 0.50, 0.9],  # 0: Nose (left side)
        [0.11, 0.48, 0.9],  # 1: Left Eye
        [0.11, 0.52, 0.9],  # 2: Right Eye
        [0.12, 0.46, 0.9],  # 3: Left Ear
        [0.12, 0.54, 0.9],  # 4: Right Ear
        [0.25, 0.42, 0.9],  # 5: Left Shoulder
        [0.25, 0.58, 0.9],  # 6: Right Shoulder
        [0.42, 0.38, 0.9],  # 7: Left Elbow
        [0.42, 0.62, 0.9],  # 8: Right Elbow
        [0.55, 0.36, 0.9],  # 9: Left Wrist
        [0.55, 0.64, 0.9],  # 10: Right Wrist
        [0.55, 0.42, 0.9],  # 11: Left Hip
        [0.55, 0.58, 0.9],  # 12: Right Hip
        [0.72, 0.42, 0.9],  # 13: Left Knee
        [0.72, 0.58, 0.9],  # 14: Right Knee
        [0.88, 0.42, 0.9],  # 15: Left Ankle
        [0.88, 0.58, 0.9],  # 16: Right Ankle
    ]
    return kps


class TestAspectRatio(unittest.TestCase):
    """Test the core aspect ratio feature."""

    def setUp(self):
        self.extractor = FeatureExtractor()

    def test_standing_has_low_aspect_ratio(self):
        """Standing person's bounding box is tall → aspect ratio < 1."""
        kps = make_standing_keypoints()
        ratio = self.extractor.compute_aspect_ratio(kps)
        self.assertLess(ratio, 0.7,
                        f"Standing should have AR < 0.7, got {ratio:.3f}")

    def test_fallen_has_high_aspect_ratio(self):
        """Fallen person's bounding box is wide → aspect ratio > 1."""
        kps = make_fallen_keypoints()
        ratio = self.extractor.compute_aspect_ratio(kps)
        self.assertGreater(ratio, 1.0,
                           f"Fallen should have AR > 1.0, got {ratio:.3f}")

    def test_aspect_ratio_non_negative(self):
        """Aspect ratio must be ≥ 0."""
        kps = make_standing_keypoints()
        ratio = self.extractor.compute_aspect_ratio(kps)
        self.assertGreaterEqual(ratio, 0.0)

    def test_aspect_ratio_zero_for_invisible_keypoints(self):
        """If all keypoints invisible → aspect ratio should be 0."""
        # All visibility = 0
        kps = [[0.5, 0.5, 0.0] for _ in range(17)]
        ratio = self.extractor.compute_aspect_ratio(kps)
        self.assertEqual(ratio, 0.0)


class TestBoundingBox(unittest.TestCase):
    """Test bounding box calculations."""

    def setUp(self):
        self.extractor = FeatureExtractor()

    def test_bounding_box_contains_all_keypoints(self):
        """All visible keypoints must fall within the bounding box."""
        kps = make_standing_keypoints()
        bbox = self.extractor.compute_bounding_box(kps)

        for kp in kps:
            if kp[2] >= 0.3:  # Visible
                self.assertGreaterEqual(kp[0], bbox["x_min"] - 1e-6)
                self.assertLessEqual(kp[0], bbox["x_max"] + 1e-6)
                self.assertGreaterEqual(kp[1], bbox["y_min"] - 1e-6)
                self.assertLessEqual(kp[1], bbox["y_max"] + 1e-6)

    def test_bounding_box_has_positive_dimensions(self):
        """Width and height must be non-negative."""
        kps = make_standing_keypoints()
        bbox = self.extractor.compute_bounding_box(kps)
        self.assertGreaterEqual(bbox["width"], 0)
        self.assertGreaterEqual(bbox["height"], 0)


class TestJointVelocity(unittest.TestCase):
    """Test velocity calculations."""

    def setUp(self):
        self.extractor = FeatureExtractor()

    def test_zero_velocity_for_same_keypoint(self):
        """Identical keypoints → zero velocity."""
        kp = [0.5, 0.5, 0.9]
        vel = self.extractor.compute_joint_velocity(kp, kp)
        self.assertAlmostEqual(vel, 0.0, places=6)

    def test_velocity_positive_for_different_positions(self):
        """Different positions → positive velocity."""
        kp1 = [0.3, 0.3, 0.9]
        kp2 = [0.5, 0.5, 0.9]
        vel = self.extractor.compute_joint_velocity(kp2, kp1)
        self.assertGreater(vel, 0.0)

    def test_velocity_zero_for_invisible_keypoint(self):
        """If either keypoint invisible → velocity is 0."""
        kp_visible = [0.5, 0.5, 0.9]
        kp_invisible = [0.5, 0.5, 0.0]  # confidence = 0
        vel = self.extractor.compute_joint_velocity(kp_visible, kp_invisible)
        self.assertEqual(vel, 0.0)

    def test_velocity_none_keypoints(self):
        """None keypoints → velocity is 0."""
        vel = self.extractor.compute_joint_velocity(None, None)
        self.assertEqual(vel, 0.0)


class TestKineticEnergyProxy(unittest.TestCase):
    """Test kinetic energy proxy (RMS velocity)."""

    def setUp(self):
        self.extractor = FeatureExtractor()

    def test_zero_velocities_give_zero_ke(self):
        """All zero velocities → KE proxy is 0."""
        velocities = [0.0] * 17
        ke = self.extractor.compute_kinetic_energy_proxy(velocities)
        self.assertAlmostEqual(ke, 0.0, places=6)

    def test_ke_is_non_negative(self):
        """KE proxy must always be ≥ 0 (it's RMS = always positive)."""
        velocities = [0.1, 0.2, 0.0, 0.05, 0.15]
        ke = self.extractor.compute_kinetic_energy_proxy(velocities)
        self.assertGreaterEqual(ke, 0.0)

    def test_higher_velocities_give_higher_ke(self):
        """Higher velocities should give higher KE proxy."""
        low_vel = [0.01] * 17
        high_vel = [0.20] * 17
        ke_low = self.extractor.compute_kinetic_energy_proxy(low_vel)
        ke_high = self.extractor.compute_kinetic_energy_proxy(high_vel)
        self.assertGreater(ke_high, ke_low)

    def test_empty_velocities_give_zero(self):
        """Empty velocity list → 0."""
        ke = self.extractor.compute_kinetic_energy_proxy([])
        self.assertEqual(ke, 0.0)


class TestFeatureExtraction(unittest.TestCase):
    """Test the full feature vector extraction."""

    def setUp(self):
        self.extractor = FeatureExtractor()

    def test_first_frame_returns_none(self):
        """First frame has no previous frame → cannot compute velocity → None."""
        kps = make_standing_keypoints()
        result = self.extractor.extract(kps)
        self.assertIsNone(result, "First frame must return None (no history).")

    def test_second_frame_returns_vector(self):
        """Second frame should return a feature vector."""
        kps = make_standing_keypoints()
        self.extractor.extract(kps)   # First frame
        result = self.extractor.extract(kps)  # Second frame
        self.assertIsNotNone(result)

    def test_feature_vector_shape(self):
        """Feature vector must have exactly 9 elements."""
        kps = make_standing_keypoints()
        self.extractor.extract(kps)
        result = self.extractor.extract(kps)
        self.assertEqual(len(result), 9,
                         f"Expected 9 features, got {len(result)}")

    def test_feature_names_count(self):
        """Should return exactly 9 feature names."""
        names = self.extractor.get_feature_names()
        self.assertEqual(len(names), 9)

    def test_reset_clears_state(self):
        """After reset, first frame logic applies again."""
        kps = make_standing_keypoints()
        self.extractor.extract(kps)
        self.extractor.reset()
        result = self.extractor.extract(kps)
        self.assertIsNone(result, "After reset, first frame must return None.")


class TestTemporalWindow(unittest.TestCase):
    """Test the sliding temporal window."""

    def setUp(self):
        self.window = TemporalWindow(window_size=5, n_features=9)

    def test_not_ready_when_empty(self):
        """Empty window should not be ready."""
        self.assertFalse(self.window.is_ready())

    def test_ready_after_enough_frames(self):
        """Should be ready after window_size/2 frames."""
        vec = np.ones(9, dtype=np.float32)
        for _ in range(3):  # 3 >= 5//2 = 2
            self.window.add(vec)
        self.assertTrue(self.window.is_ready())

    def test_temporal_features_shape(self):
        """Temporal features should have 9×4 = 36 elements."""
        vec = np.ones(9, dtype=np.float32)
        for _ in range(5):
            self.window.add(vec)
        features = self.window.compute_temporal_features()
        self.assertIsNotNone(features)
        self.assertEqual(features.shape[0], 36)

    def test_temporal_features_none_when_empty(self):
        """Empty window → None temporal features."""
        result = self.window.compute_temporal_features()
        self.assertIsNone(result)

    def test_sliding_window_max_size(self):
        """Window should not grow beyond max size."""
        vec = np.ones(9, dtype=np.float32)
        for _ in range(20):  # Add more than window_size
            self.window.add(vec)
        self.assertLessEqual(len(self.window.window), 5)

    def test_feature_names_count(self):
        """Should return 36 temporal feature names."""
        names = self.window.get_temporal_feature_names()
        self.assertEqual(len(names), 36)


if __name__ == "__main__":
    print("Running Feature Extraction Tests...")
    print("=" * 60)
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestAspectRatio))
    suite.addTests(loader.loadTestsFromTestCase(TestBoundingBox))
    suite.addTests(loader.loadTestsFromTestCase(TestJointVelocity))
    suite.addTests(loader.loadTestsFromTestCase(TestKineticEnergyProxy))
    suite.addTests(loader.loadTestsFromTestCase(TestFeatureExtraction))
    suite.addTests(loader.loadTestsFromTestCase(TestTemporalWindow))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if result.wasSuccessful():
        print("\n✅ All feature tests passed!")
    else:
        print(f"\n❌ {len(result.failures)} test(s) failed.")
