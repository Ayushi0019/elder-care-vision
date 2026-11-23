"""
tests/test_motion.py — Unit Tests for Motion Detection Module
==============================================================
Run with: python -m pytest tests/ -v
Or:        python tests/test_motion.py

Tests cover:
1. Static scene → no motion
2. Moving scene → motion detected
3. Motion ratio bounds
4. DBF initialization
5. Morphology pipeline
6. Optical flow on static scene
"""

import sys
import os
import cv2
import numpy as np
import unittest

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from motion.motion_detection import MotionDetector
from motion.optical_flow import OpticalFlowAnalyzer
from motion.background_filter import DoubleBackgroundFilter
from motion.morphology import MorphologyProcessor


def make_gray_frame(height=480, width=640, value=100):
    """Create a flat grayscale frame (static scene)."""
    return np.full((height, width), value, dtype=np.uint8)


def make_moving_frame(height=480, width=640):
    """Create a frame with a bright blob (simulates person)."""
    frame = np.zeros((height, width), dtype=np.uint8)
    # Draw a white rectangle representing a moving object
    frame[100:300, 200:400] = 200
    return frame


class TestMotionDetector(unittest.TestCase):
    """Tests for temporal frame differencing."""

    def setUp(self):
        self.detector = MotionDetector(threshold=25)

    def test_first_frame_no_motion(self):
        """First frame should never report motion (no previous frame)."""
        frame_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        result = self.detector.detect(frame_bgr)
        self.assertFalse(result["motion_detected"],
                         "First frame should never have motion.")
        self.assertEqual(result["motion_ratio"], 0.0)

    def test_identical_frames_no_motion(self):
        """Two identical frames should show no motion."""
        frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        self.detector.detect(frame)   # Feed first frame
        result = self.detector.detect(frame)   # Same frame again
        self.assertFalse(result["motion_detected"],
                         "Identical frames should not trigger motion.")

    def test_very_different_frames_motion(self):
        """Dramatically different frames should trigger motion."""
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)    # All black
        frame2 = np.full((480, 640, 3), 200, dtype=np.uint8) # All bright

        self.detector.detect(frame1)
        result = self.detector.detect(frame2)

        # Large change should be detected
        self.assertGreater(result["motion_ratio"], 0.5,
                           "Major lighting change should have high ratio.")

    def test_motion_ratio_bounds(self):
        """Motion ratio must always be between 0.0 and 1.0."""
        frames = [
            np.zeros((480, 640, 3), dtype=np.uint8),
            np.full((480, 640, 3), 255, dtype=np.uint8),
            np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
        ]
        for frame in frames:
            result = self.detector.detect(frame)
            self.assertGreaterEqual(result["motion_ratio"], 0.0)
            self.assertLessEqual(result["motion_ratio"], 1.0)

    def test_mask_shape(self):
        """Motion mask should match frame dimensions."""
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        self.detector.detect(frame)
        result = self.detector.detect(frame)
        self.assertEqual(result["mask"].shape, (240, 320))

    def test_reset_clears_previous_frame(self):
        """After reset, first frame logic should apply again."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.detector.detect(frame)
        self.detector.reset()
        result = self.detector.detect(frame)
        self.assertFalse(result["motion_detected"],
                         "After reset, first frame should have no motion.")

    def test_small_movement_detected(self):
        """Small region moving should be detected if above min_ratio."""
        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
        frame2 = frame1.copy()
        # Move a small patch
        frame2[100:150, 100:200] = 200

        self.detector.detect(frame1)
        result = self.detector.detect(frame2)

        # Ratio = (50×100) / (480×640) ≈ 0.016 → above min_ratio 0.001
        self.assertTrue(result["motion_detected"] or result["motion_ratio"] > 0,
                        "Small movement should increase motion ratio.")


class TestDoubleBackgroundFilter(unittest.TestCase):
    """Tests for the DBF module."""

    def setUp(self):
        self.dbf = DoubleBackgroundFilter(short_window=3, long_window=5)

    def test_not_ready_until_buffers_full(self):
        """DBF should not be ready until both windows are full."""
        gray = make_gray_frame()
        result = self.dbf.update_and_filter(gray)
        self.assertFalse(result["dbf_ready"],
                         "Should not be ready after 1 frame.")

    def test_ready_after_long_window_frames(self):
        """Should be ready after LONG_WINDOW frames are collected."""
        gray = make_gray_frame()
        result = None
        for _ in range(5):  # long_window = 5
            result = self.dbf.update_and_filter(gray)
        self.assertTrue(result["dbf_ready"], "Should be ready after 5 frames.")

    def test_static_scene_minimal_foreground(self):
        """Static scene should produce near-zero foreground mask."""
        gray = make_gray_frame(value=120)  # Same frame
        for _ in range(6):  # Fill buffers
            result = self.dbf.update_and_filter(gray)

        # A completely static scene should have very few foreground pixels
        fg_pixels = np.count_nonzero(result["foreground_mask"])
        self.assertLess(fg_pixels, 100,
                        "Static scene should have minimal foreground.")

    def test_reset_clears_buffers(self):
        """Reset should empty both frame buffers."""
        gray = make_gray_frame()
        for _ in range(5):
            self.dbf.update_and_filter(gray)
        self.dbf.reset()
        self.assertEqual(len(self.dbf.short_buffer), 0)
        self.assertEqual(len(self.dbf.long_buffer), 0)


class TestMorphologyProcessor(unittest.TestCase):
    """Tests for morphological operations."""

    def setUp(self):
        self.morph = MorphologyProcessor(kernel_size=(5, 5))

    def test_process_returns_same_shape(self):
        """Output mask must have same shape as input."""
        mask = np.random.randint(0, 2, (480, 640), dtype=np.uint8) * 255
        result = self.morph.process(mask)
        self.assertEqual(result.shape, mask.shape)

    def test_opening_removes_single_pixels(self):
        """Opening should remove isolated noise pixels."""
        # Single pixel noise on black background
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[50, 50] = 255  # Single white pixel
        result = self.morph.opening(mask)
        # A single pixel should be erased by opening with 5x5 kernel
        self.assertEqual(np.count_nonzero(result), 0,
                         "Single pixel should be removed by opening.")

    def test_closing_fills_small_holes(self):
        """Closing should fill small holes in a large white region."""
        mask = np.ones((50, 50), dtype=np.uint8) * 255
        mask[25, 25] = 0  # Small hole in middle
        result = self.morph.closing(mask)
        # The hole should be filled
        self.assertEqual(result[25, 25], 255,
                         "Small hole should be filled by closing.")

    def test_process_empty_mask(self):
        """Processing an all-black mask should return all-black."""
        mask = np.zeros((480, 640), dtype=np.uint8)
        result = self.morph.process(mask)
        self.assertEqual(np.count_nonzero(result), 0,
                         "Empty mask processed should remain empty.")

    def test_largest_contour_area(self):
        """Should find largest contour area correctly."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(mask, (10, 10), (60, 60), 255, -1)  # 50x50 = 2500 px
        area = self.morph.largest_contour_area(mask)
        self.assertGreater(area, 1000,
                           "Should detect large rectangle as biggest contour.")


class TestOpticalFlow(unittest.TestCase):
    """Tests for Lucas-Kanade optical flow."""

    def setUp(self):
        self.flow = OpticalFlowAnalyzer()

    def test_static_scene_minimal_flow(self):
        """Identical frames should have near-zero optical flow."""
        gray = make_gray_frame()
        result = self.flow.calculate_optical_flow(gray, gray)
        self.assertAlmostEqual(result["mean_magnitude"], 0.0, places=1,
                               msg="Static scene should have ~0 flow magnitude.")

    def test_returns_required_keys(self):
        """Result must always contain the required keys."""
        gray = make_gray_frame()
        result = self.flow.calculate_optical_flow(gray, gray)
        required_keys = [
            "flow_vectors", "mean_vx", "mean_vy",
            "mean_magnitude", "mean_direction", "num_points", "has_flow"
        ]
        for key in required_keys:
            self.assertIn(key, result, f"Missing key: {key}")

    def test_moving_scene_has_flow(self):
        """A scene where objects move should show non-zero flow magnitude."""
        # Create a textured frame (checkerboard)
        gray1 = np.zeros((200, 200), dtype=np.uint8)
        for i in range(0, 200, 20):
            for j in range(0, 200, 20):
                if (i // 20 + j // 20) % 2 == 0:
                    gray1[i:i+20, j:j+20] = 200

        # Shift by 5 pixels horizontally
        M = np.float32([[1, 0, 5], [0, 1, 0]])
        gray2 = cv2.warpAffine(gray1, M, (200, 200))

        result = self.flow.calculate_optical_flow(gray1, gray2)
        self.assertGreater(result["mean_magnitude"], 0.5,
                           "Shifted scene should show non-trivial flow.")


if __name__ == "__main__":
    print("Running Motion Detection Tests...")
    print("=" * 60)
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    suite.addTests(loader.loadTestsFromTestCase(TestMotionDetector))
    suite.addTests(loader.loadTestsFromTestCase(TestDoubleBackgroundFilter))
    suite.addTests(loader.loadTestsFromTestCase(TestMorphologyProcessor))
    suite.addTests(loader.loadTestsFromTestCase(TestOpticalFlow))

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if result.wasSuccessful():
        print("\n✅ All motion tests passed!")
    else:
        print(f"\n❌ {len(result.failures)} test(s) failed.")
