"""
motion/motion_detection.py — Temporal Frame Differencing
==========================================================
PURPOSE:
    Detect whether meaningful human motion is occurring in the scene.

HOW IT WORKS:
    1. Convert current and previous frame to grayscale.
    2. Apply Gaussian blur to reduce sensor noise.
    3. Calculate absolute pixel difference between frames:
           diff = |I(x,y,t) - I(x,y,t-1)|
    4. Threshold the difference to create a binary mask.
    5. Calculate what fraction of pixels changed (motion ratio).
    6. If motion ratio is in the expected range → motion detected.

WHY GRAYSCALE?
    Color information is not needed for motion detection.
    Grayscale is faster to process and reduces memory usage.

WHY BLUR FIRST?
    Camera noise creates tiny pixel differences even in a still scene.
    Blurring smooths these out before differencing.

WHY MOTION RATIO RANGE?
    Too low  → probably noise, not real movement.
    Too high → probably a sudden lighting change or camera shake.
"""

import cv2
import numpy as np
import logging
from typing import Optional, Dict, Any, Tuple
import config

logger = logging.getLogger(__name__)


class MotionDetector:
    """
    Detects motion using temporal frame differencing.

    Compares consecutive frames to identify where movement occurred.
    Returns a motion mask and a motion ratio indicating how much of
    the scene is moving.
    """

    def __init__(
        self,
        threshold: int = config.MOTION_THRESHOLD,
        blur_kernel: Tuple[int, int] = config.MOTION_BLUR_KERNEL,
        min_ratio: float = config.MOTION_RATIO_MIN,
        max_ratio: float = config.MOTION_RATIO_MAX,
        use_adaptive: bool = config.USE_ADAPTIVE_THRESHOLD,
    ):
        """
        Args:
            threshold:    Pixel intensity difference to count as motion (0-255)
            blur_kernel:  Gaussian blur kernel size, e.g., (5,5)
            min_ratio:    Minimum fraction of pixels that must change
            max_ratio:    Maximum fraction (beyond this = noise/lighting)
            use_adaptive: Use OpenCV adaptive threshold instead of fixed
        """
        self.threshold = threshold
        self.blur_kernel = blur_kernel
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.use_adaptive = use_adaptive

        # Previous frame stored for differencing
        self.prev_gray: Optional[np.ndarray] = None

        # Tracking statistics
        self._detection_count = 0
        self._frame_count = 0

    def preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Convert a BGR frame to blurred grayscale.

        Step 1: BGR → Grayscale  (removes color, reduces data)
        Step 2: Gaussian Blur    (removes sensor noise)

        Args:
            frame: Raw BGR camera frame (numpy array)

        Returns:
            Blurred grayscale image (numpy array, single channel)
        """
        # Step 1: Convert color to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Step 2: Blur to reduce noise
        # GaussianBlur smooths pixel values using a weighted average
        # of surrounding pixels. Kernel (5,5) means 5×5 neighborhood.
        blurred = cv2.GaussianBlur(gray, self.blur_kernel, 0)

        return blurred

    def compute_difference(
        self,
        current_gray: np.ndarray,
        previous_gray: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate absolute pixel difference between two frames.

        For each pixel (x, y):
            diff(x, y) = |current_gray(x,y) - previous_gray(x,y)|

        High values = large change = possible motion.
        Low values = little change = static area.

        Args:
            current_gray:   Current blurred grayscale frame
            previous_gray:  Previous blurred grayscale frame

        Returns:
            Difference image (numpy array, values 0–255)
        """
        diff = cv2.absdiff(current_gray, previous_gray)
        return diff

    def apply_threshold(self, diff: np.ndarray) -> np.ndarray:
        """
        Convert continuous difference image to binary mask.

        Pixels where diff > threshold → 255 (white = motion)
        Pixels where diff ≤ threshold → 0   (black = static)

        Two modes:
        - Fixed threshold: Simple, fast, works in consistent lighting.
        - Adaptive threshold: Smarter, adjusts per region, better for
          uneven lighting but slightly slower.

        Args:
            diff: Difference image from compute_difference()

        Returns:
            Binary mask (0 or 255 per pixel)
        """
        if self.use_adaptive:
            # Adaptive threshold: threshold is calculated per small region
            # Good for scenes with uneven lighting (e.g., window + dark corner)
            mask = cv2.adaptiveThreshold(
                diff,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                blockSize=11,
                C=-2,
            )
        else:
            # Fixed threshold: simple global cutoff
            # Any pixel with difference > self.threshold becomes white
            _, mask = cv2.threshold(diff, self.threshold, 255, cv2.THRESH_BINARY)

        return mask

    def calculate_motion_ratio(self, mask: np.ndarray) -> float:
        """
        Calculate what fraction of the frame is in motion.

        motion_ratio = (white pixels in mask) / (total pixels)

        This tells us:
        - 0.0  = completely static
        - 1.0  = entire frame is "moving" (usually noise or camera moved)
        - 0.05 = 5% of frame has motion (typical for a person walking)

        Args:
            mask: Binary motion mask (values 0 or 255)

        Returns:
            Float between 0.0 and 1.0
        """
        total_pixels = mask.shape[0] * mask.shape[1]   # height × width
        moving_pixels = np.count_nonzero(mask)           # pixels that are 255
        ratio = moving_pixels / total_pixels
        return float(ratio)

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Main method: Run motion detection on a new frame.

        Call this every frame. It automatically stores the previous frame.

        Args:
            frame: Raw BGR frame from camera (numpy array)

        Returns:
            Dictionary with:
                motion_detected (bool):   True if meaningful motion found
                motion_ratio    (float):  Fraction of pixels moving (0-1)
                mask            (ndarray): Binary motion mask
                current_gray    (ndarray): Preprocessed current frame
                                          (needed by optical flow module)
        """
        self._frame_count += 1

        # Step 1: Preprocess (grayscale + blur)
        current_gray = self.preprocess(frame)

        # Step 2: On the very first frame, no previous frame exists.
        #         Store it and return "no motion" for this frame.
        if self.prev_gray is None:
            self.prev_gray = current_gray
            return {
                "motion_detected": False,
                "motion_ratio": 0.0,
                "mask": np.zeros_like(current_gray),
                "current_gray": current_gray,
            }

        # Step 3: Compute absolute frame difference
        diff = self.compute_difference(current_gray, self.prev_gray)

        # Step 4: Threshold to binary mask
        mask = self.apply_threshold(diff)

        # Step 5: Calculate motion ratio
        ratio = self.calculate_motion_ratio(mask)

        # Step 6: Decide if this is "real" motion
        # We check both a minimum (to filter noise) and maximum (to filter
        # sudden global changes like someone turning on a light)
        motion_detected = self.min_ratio < ratio < self.max_ratio

        if motion_detected:
            self._detection_count += 1

        # Step 7: Update previous frame for next iteration
        self.prev_gray = current_gray

        result = {
            "motion_detected": motion_detected,
            "motion_ratio": ratio,
            "mask": mask,
            "current_gray": current_gray,
        }

        logger.debug(
            f"Motion: {motion_detected} | Ratio: {ratio:.4f}"
        )

        return result

    def reset(self) -> None:
        """Reset the detector (clear stored previous frame)."""
        self.prev_gray = None
        logger.info("MotionDetector reset.")

    def get_stats(self) -> Dict[str, Any]:
        """Return detection statistics."""
        return {
            "total_frames": self._frame_count,
            "motion_frames": self._detection_count,
            "motion_rate": (
                self._detection_count / self._frame_count
                if self._frame_count > 0
                else 0.0
            ),
        }
