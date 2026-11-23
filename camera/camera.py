"""
camera/camera.py — Webcam Capture Module
=========================================
PURPOSE:
    Opens the webcam, reads frames continuously, and provides
    them to the rest of the pipeline.

PRIVACY:
    Frames are processed IN MEMORY only.
    Raw RGB frames are NEVER written to disk here.
    The caller is responsible for discarding the frame after use.

USAGE:
    cam = CameraCapture()
    cam.start()
    frame = cam.read_frame()
    cam.stop()

    OR use as a context manager:
    with CameraCapture() as cam:
        frame = cam.read_frame()
"""

import cv2
import time
import logging
from typing import Optional, Tuple
import config

logger = logging.getLogger(__name__)


class CameraCapture:
    """
    Manages webcam connection and frame reading.

    This class wraps OpenCV's VideoCapture so the rest of the
    system doesn't need to deal with camera errors directly.
    """

    def __init__(
        self,
        camera_index=config.CAMERA_INDEX,
        width: int = config.FRAME_WIDTH,
        height: int = config.FRAME_HEIGHT,
        fps: int = config.TARGET_FPS,
    ):
        """
        Initialize camera settings.

        Args:
            camera_index: Which camera to open (0 = built-in webcam) or video file path
            width:        Frame width in pixels
            height:       Frame height in pixels
            fps:          Target frames per second
        """
        if isinstance(camera_index, str) and camera_index.isdigit():
            self.camera_index = int(camera_index)
        else:
            self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False

        # Performance tracking
        self._frame_count = 0
        self._start_time = 0.0
        self._actual_fps = 0.0

    def start(self) -> bool:
        """
        Open the webcam.

        Returns:
            True if camera opened successfully, False otherwise.
        """
        logger.info(f"Opening camera index={self.camera_index} ...")

        self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap.isOpened():
            logger.error(
                f"❌ Cannot open camera index={self.camera_index}. "
                f"Check if camera is connected and not used by another app."
            )
            return False

        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Read back actual values (camera may not support all settings)
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS)

        logger.info(
            f"✅ Camera opened. "
            f"Resolution: {actual_width}x{actual_height} @ {actual_fps:.1f} FPS"
        )

        self.is_running = True
        self._start_time = time.time()
        self._frame_count = 0
        return True

    def read_frame(self) -> Tuple[bool, Optional[object]]:
        """
        Read one frame from the webcam.

        Returns:
            (success, frame)
            success: True if frame was read successfully
            frame:   OpenCV BGR image (numpy array), or None on failure

        IMPORTANT: The returned frame is a raw RGB image.
                   The caller MUST process it and NOT store it to disk.
        """
        if not self.is_running or self.cap is None:
            logger.warning("read_frame() called but camera is not started.")
            return False, None

        ret, frame = self.cap.read()

        if not ret:
            logger.error("❌ Failed to read frame from camera.")
            return False, None

        # Track frame count for FPS calculation
        self._frame_count += 1

        return True, frame

    def get_fps(self) -> float:
        """
        Calculate the actual frames per second being processed.

        Returns:
            Measured FPS as a float.
        """
        elapsed = time.time() - self._start_time
        if elapsed > 0:
            self._actual_fps = self._frame_count / elapsed
        return self._actual_fps

    def get_frame_count(self) -> int:
        """Return total frames read since camera started."""
        return self._frame_count

    def get_resolution(self) -> Tuple[int, int]:
        """Return actual camera resolution as (width, height)."""
        if self.cap is None:
            return (0, 0)
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (w, h)

    def stop(self) -> None:
        """
        Release the webcam and clean up.

        Call this when the monitoring session ends.
        """
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            logger.info(
                f"Camera stopped. "
                f"Total frames processed: {self._frame_count}, "
                f"Average FPS: {self.get_fps():.2f}"
            )
        self.is_running = False

    # -------------------------------------------------------------------------
    # Context manager support: allows "with CameraCapture() as cam:"
    # -------------------------------------------------------------------------
    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
        return False   # Don't suppress exceptions

    def __repr__(self):
        status = "running" if self.is_running else "stopped"
        return (
            f"CameraCapture(index={self.camera_index}, "
            f"resolution={self.get_resolution()}, "
            f"status={status})"
        )
