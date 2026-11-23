"""
motion/background_filter.py — Double Background Filtering (DBF)
================================================================
PURPOSE:
    Remove environmental background interference while preserving
    genuine human foreground movement.

THE PROBLEM:
    Simple frame differencing detects ALL changes between frames.
    This includes:
    - Fans rotating
    - Curtains swaying in breeze
    - Shadows moving when clouds pass
    - Light flickering
    - Screen glow from TV

    These are environmental changes, not human movement.
    If we don't filter them out, we get false motion detections
    which can lead to false fall alarms.

HOW DBF WORKS:
    DBF uses TWO temporal accumulators with different time scales:

    SHORT-TERM accumulator (3 frames):
        Captures recent fast changes.
        Sensitive to quick environmental changes (fans, flickering).

    LONG-TERM accumulator (5 frames):
        Captures slower, more stable background.
        Less sensitive to fast changes.

    FOREGROUND MASK:
        We compare the SHORT-TERM background to the LONG-TERM background.
        
        If a region has changed significantly in the SHORT-TERM but
        matches the LONG-TERM background, it's probably environmental
        (e.g., a fan that completed a cycle).
        
        If a region has MOVED compared to BOTH backgrounds, it's more
        likely a genuine foreground object (human body).

    This "double filtering" reduces environmental interference while
    preserving the human body movement that we care about.

    Conceptually:
        short_bg = average of last 3 frames
        long_bg  = average of last 5 frames
        foreground = |current - short_bg| AND |current - long_bg|

NOTE:
    This is the project-specific DBF concept described in the synopsis.
    It is a simplified temporal accumulation approach, not a full
    Gaussian Mixture Model (GMM) like OpenCV's MOG2.
    The advantage: simpler, faster, easier to understand.

CONFIGURABLE WINDOW SIZES:
    SHORT_WINDOW and LONG_WINDOW can be changed in config.py.
"""

import cv2
import numpy as np
import logging
from collections import deque
from typing import Tuple, Optional
import config

logger = logging.getLogger(__name__)


class DoubleBackgroundFilter:
    """
    Implements Double Background Filtering (DBF) as described in the
    project synopsis.

    Uses two sliding windows of different lengths to separate genuine
    human foreground movement from environmental background changes.
    """

    def __init__(
        self,
        short_window: int = config.DBF_SHORT_WINDOW,
        long_window: int = config.DBF_LONG_WINDOW,
        diff_threshold: int = config.DBF_DIFF_THRESHOLD,
        frame_size: Optional[Tuple[int, int]] = None,
    ):
        """
        Args:
            short_window:    Number of frames for short-term background
            long_window:     Number of frames for long-term background
            diff_threshold:  Pixel intensity difference threshold
            frame_size:      (height, width) — set automatically on first frame
        """
        self.short_window = short_window
        self.long_window = long_window
        self.diff_threshold = diff_threshold

        # deque (double-ended queue): automatically discards old frames
        # when it reaches max length. Perfect for sliding window!
        self.short_buffer: deque = deque(maxlen=short_window)
        self.long_buffer: deque = deque(maxlen=long_window)

        logger.info(
            f"DoubleBackgroundFilter initialized: "
            f"short_window={short_window}, long_window={long_window}, "
            f"threshold={diff_threshold}"
        )

    def _compute_background(self, buffer: deque) -> Optional[np.ndarray]:
        """
        Compute background estimate as the mean of frames in buffer.

        Mean (average) of N frames: reduces noise while capturing
        the stable background that persists across frames.

        Args:
            buffer: deque of grayscale frames

        Returns:
            Mean frame as uint8 numpy array, or None if buffer is empty.
        """
        if len(buffer) == 0:
            return None

        # Stack all frames into a 3D array: shape (N, H, W)
        stacked = np.stack(list(buffer), axis=0).astype(np.float32)

        # Compute mean along the frame axis (axis=0)
        mean_bg = np.mean(stacked, axis=0).astype(np.uint8)

        return mean_bg

    def update_and_filter(self, gray_frame: np.ndarray) -> dict:
        """
        Add current frame to both buffers and compute DBF foreground mask.

        Call this every frame with the preprocessed grayscale image.

        DBF Logic:
        1. Add current frame to both buffers.
        2. Compute short-term background (avg of last SHORT_WINDOW frames).
        3. Compute long-term background  (avg of last LONG_WINDOW frames).
        4. Compute foreground masks from both backgrounds.
        5. Combine masks with AND operation:
               Only keep regions that differ from BOTH backgrounds.
               This removes environmental patterns that appear in either bg.

        Args:
            gray_frame: Preprocessed grayscale frame

        Returns:
            Dictionary with:
                foreground_mask  (ndarray): Binary DBF mask (0 or 255)
                short_bg         (ndarray): Short-term background estimate
                long_bg          (ndarray): Long-term background estimate
                short_diff       (ndarray): Difference from short-term bg
                long_diff        (ndarray): Difference from long-term bg
                dbf_ready        (bool):    False until both buffers filled
        """
        # Add current frame to BOTH buffers
        self.short_buffer.append(gray_frame.copy())
        self.long_buffer.append(gray_frame.copy())

        # Check if buffers have enough frames
        dbf_ready = (
            len(self.short_buffer) >= self.short_window
            and len(self.long_buffer) >= self.long_window
        )

        if not dbf_ready:
            # Not enough frames yet — return empty mask
            h, w = gray_frame.shape[:2]
            return {
                "foreground_mask": np.zeros((h, w), dtype=np.uint8),
                "short_bg": gray_frame.copy(),
                "long_bg": gray_frame.copy(),
                "short_diff": np.zeros((h, w), dtype=np.uint8),
                "long_diff": np.zeros((h, w), dtype=np.uint8),
                "dbf_ready": False,
            }

        # Compute background estimates
        short_bg = self._compute_background(self.short_buffer)
        long_bg = self._compute_background(self.long_buffer)

        # Compute absolute difference from each background
        # |current - short_bg|: How different is current from recent past?
        short_diff = cv2.absdiff(gray_frame, short_bg)

        # |current - long_bg|: How different is current from longer past?
        long_diff = cv2.absdiff(gray_frame, long_bg)

        # Threshold both differences to binary masks
        _, short_mask = cv2.threshold(
            short_diff, self.diff_threshold, 255, cv2.THRESH_BINARY
        )
        _, long_mask = cv2.threshold(
            long_diff, self.diff_threshold, 255, cv2.THRESH_BINARY
        )

        # DBF Combination: AND of both masks
        # A pixel is "foreground" only if it differs from BOTH backgrounds.
        # 
        # Example: A fan blade appears in short_mask (it moved recently)
        #   but NOT in long_mask (its average position is stable over
        #   5 frames). So it gets filtered out by the AND operation.
        #
        # A human arm appears in BOTH masks (it wasn't there before,
        #   and it's still different from both backgrounds).
        #   So it survives the AND and is kept as foreground.
        foreground_mask = cv2.bitwise_and(short_mask, long_mask)

        logger.debug(
            f"DBF: short_buffer={len(self.short_buffer)}/{self.short_window}, "
            f"long_buffer={len(self.long_buffer)}/{self.long_window}, "
            f"fg_pixels={np.count_nonzero(foreground_mask)}"
        )

        return {
            "foreground_mask": foreground_mask,
            "short_bg": short_bg,
            "long_bg": long_bg,
            "short_diff": short_diff,
            "long_diff": long_diff,
            "dbf_ready": True,
        }

    def reset(self) -> None:
        """Clear both frame buffers (e.g., when camera is restarted)."""
        self.short_buffer.clear()
        self.long_buffer.clear()
        logger.info("DoubleBackgroundFilter buffers cleared.")
