"""
pose/pose_estimation.py — MediaPipe Skeletal Pose Estimation
=============================================================
PURPOSE:
    Extract the 17-point skeletal representation of a person from a
    camera frame. After extraction, the raw RGB frame is discarded.

WHY MEDIAPIPE?
    - Runs efficiently on CPU (no GPU required)
    - Provides 33 landmarks but we use 17 key joints
    - Open source (Google)
    - Lightweight enough for Raspberry Pi
    - Well-documented API
    - No identity recognition (privacy-preserving)

THE 17-POINT SKELETON:
    Our project uses 17 key joints that match standard human pose
    representation (compatible with COCO format):

    Head region:
        0  = Nose
        1  = Left Eye
        2  = Right Eye
        3  = Left Ear
        4  = Right Ear

    Upper body:
        5  = Left Shoulder
        6  = Right Shoulder
        7  = Left Elbow
        8  = Right Elbow
        9  = Left Wrist
        10 = Right Wrist

    Lower body:
        11 = Left Hip
        12 = Right Hip
        13 = Left Knee
        14 = Right Knee
        15 = Left Ankle
        16 = Right Ankle

WHY 17 KEYPOINTS?
    MediaPipe actually gives 33 landmarks. We select 17 because:
    - These 17 match the COCO body keypoint standard
    - They capture all important joints for fall detection
    - Fewer points = faster feature extraction
    - Reduces complexity while keeping all necessary information

PRIVACY:
    - No face recognition is performed
    - No identity is extracted
    - The output is only numerical coordinates (x, y, confidence)
    - Raw frames are not stored in this module

MODULARITY:
    This module can be replaced with YOLO-Pose by:
    1. Keeping the same output format: List of 17 [x, y, confidence] 
    2. Replacing the internal MediaPipe code
    The rest of the system doesn't need to change.
"""

import cv2
import numpy as np
import logging
from typing import Optional, List, Tuple, Dict, Any
import config

logger = logging.getLogger(__name__)

# Try importing mediapipe — give helpful message if not installed
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.error(
        "MediaPipe not found! Install it with: pip install mediapipe"
    )


# MediaPipe landmark indices that map to our 17 keypoints
# MediaPipe has 33 landmarks (0-32)
# We pick these 17 to match COCO format
MEDIAPIPE_TO_OUR_17 = [
    0,   # 0: Nose
    2,   # 1: Left Eye  (MediaPipe index 2)
    5,   # 2: Right Eye (MediaPipe index 5)
    7,   # 3: Left Ear  (MediaPipe index 7)
    8,   # 4: Right Ear (MediaPipe index 8)
    11,  # 5: Left Shoulder
    12,  # 6: Right Shoulder
    13,  # 7: Left Elbow
    14,  # 8: Right Elbow
    15,  # 9: Left Wrist
    16,  # 10: Right Wrist
    23,  # 11: Left Hip
    24,  # 12: Right Hip
    25,  # 13: Left Knee
    26,  # 14: Right Knee
    27,  # 15: Left Ankle
    28,  # 16: Right Ankle
]


class PoseEstimator:
    """
    Extracts 17-point skeletal keypoints from camera frames using MediaPipe.

    IMPORTANT: This module only READS the frame.
    It does NOT store the frame.
    Only numerical keypoint data is returned.
    The caller should discard the raw frame after calling estimate().
    """

    def __init__(
        self,
        min_detection_confidence: float = config.POSE_MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence: float = config.POSE_MIN_TRACKING_CONFIDENCE,
    ):
        """
        Args:
            min_detection_confidence: Minimum score to detect a person (0-1)
            min_tracking_confidence:  Minimum score to keep tracking (0-1)
        """
        if not MEDIAPIPE_AVAILABLE:
            raise RuntimeError(
                "MediaPipe is required for pose estimation. "
                "Install with: pip install mediapipe"
            )

        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence

        # Initialize MediaPipe Pose
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils

        self.pose = self.mp_pose.Pose(
            static_image_mode=False,          # Video mode (tracks across frames)
            model_complexity=1,               # 0=lite, 1=full, 2=heavy
            smooth_landmarks=True,            # Reduces jitter between frames
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self._frame_count = 0
        self._detection_count = 0

        logger.info(
            f"PoseEstimator initialized: "
            f"detection_conf={min_detection_confidence}, "
            f"tracking_conf={min_tracking_confidence}"
        )

    def estimate(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Extract 17-point skeleton from a raw BGR frame.

        PRIVACY IMPORTANT:
            This function reads the frame and returns ONLY numerical data.
            The frame should be discarded by the caller immediately after.

        Args:
            frame: Raw BGR camera frame (numpy array)

        Returns:
            Dictionary with:
                person_detected (bool):        True if a person was found
                keypoints       (list):        17 keypoints as [x, y, confidence]
                                               Coordinates are normalized 0-1
                raw_landmarks   (object):      MediaPipe landmark object (internal use)
                frame_shape     (tuple):       (height, width) of input frame
        """
        self._frame_count += 1

        h, w = frame.shape[:2]

        empty_result = {
            "person_detected": False,
            "keypoints": None,
            "raw_landmarks": None,
            "frame_shape": (h, w),
        }

        # MediaPipe requires RGB input (not BGR which is OpenCV's default)
        # We convert here. This conversion is temporary and in-memory.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Run MediaPipe pose estimation
        # MediaPipe processes the frame and returns landmark positions
        results = self.pose.process(frame_rgb)

        # frame_rgb is no longer needed — release reference
        del frame_rgb

        # Check if a person was detected
        if not results.pose_landmarks:
            logger.debug("No person detected in frame.")
            return empty_result

        self._detection_count += 1

        # Extract our 17 keypoints from MediaPipe's 33 landmarks
        keypoints = self._extract_17_keypoints(
            results.pose_landmarks, w, h
        )

        logger.debug(
            f"Person detected. Keypoints extracted: {len(keypoints)}"
        )

        return {
            "person_detected": True,
            "keypoints": keypoints,
            "raw_landmarks": results.pose_landmarks,
            "frame_shape": (h, w),
        }

    def _extract_17_keypoints(
        self,
        landmarks,
        frame_width: int,
        frame_height: int,
    ) -> List[List[float]]:
        """
        Convert MediaPipe's 33 landmarks to our 17-point representation.

        Each keypoint format: [x_normalized, y_normalized, visibility]

        x_normalized: 0.0 = left edge, 1.0 = right edge
        y_normalized: 0.0 = top edge,  1.0 = bottom edge
        visibility:   0.0 = not visible, 1.0 = fully visible

        IMPORTANT: Coordinates are NORMALIZED (0-1 range).
        This means the system works regardless of camera resolution.
        A person standing on the left half has x ≈ 0.25,
        regardless of whether camera is 480p or 1080p.

        Args:
            landmarks:     MediaPipe pose landmarks (33 points)
            frame_width:   Frame width in pixels (for reference)
            frame_height:  Frame height in pixels (for reference)

        Returns:
            List of 17 keypoints, each as [x, y, confidence]
        """
        all_landmarks = landmarks.landmark  # All 33 MediaPipe landmarks

        keypoints = []
        for mp_idx in MEDIAPIPE_TO_OUR_17:
            lm = all_landmarks[mp_idx]

            # MediaPipe already provides normalized coordinates (0-1 range)
            x = float(lm.x)
            y = float(lm.y)
            confidence = float(lm.visibility)  # How confident MediaPipe is

            keypoints.append([x, y, confidence])

        return keypoints  # 17 keypoints total

    def draw_skeleton_privacy(
        self,
        frame_shape: Tuple[int, int],
        keypoints: List[List[float]],
        activity_label: str = "Unknown",
        confidence: float = 0.0,
    ) -> np.ndarray:
        """
        Draw a PRIVACY-PRESERVING skeletal visualization.

        Instead of showing the raw camera image, this function creates
        a dark background image with ONLY the skeleton drawn on it.

        This is the privacy-preserving visualization for the dashboard.
        No facial features, no clothing, no background is shown.

        Args:
            frame_shape:    (height, width) for the output image size
            keypoints:      17 keypoints [[x, y, conf], ...]
            activity_label: Current activity label (shown on screen)
            confidence:     Model confidence (shown on screen)

        Returns:
            Dark image with skeleton drawn (NOT the raw camera image)
        """
        h, w = frame_shape

        # Create dark background — NOT the real camera frame
        vis = np.full((h, w, 3), config.SKELETON_BG_COLOR, dtype=np.uint8)

        if keypoints is None or len(keypoints) < 17:
            # No skeleton to draw
            cv2.putText(
                vis, "No person detected",
                (10, h // 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (150, 150, 150), 2
            )
            return vis

        # Convert normalized coordinates (0-1) to pixel coordinates
        points_px = []
        for kp in keypoints:
            x_px = int(kp[0] * w)
            y_px = int(kp[1] * h)
            vis_conf = kp[2]
            points_px.append((x_px, y_px, vis_conf))

        # Draw skeleton connections (bones)
        for (joint_a, joint_b) in config.SKELETON_CONNECTIONS:
            if joint_a >= len(points_px) or joint_b >= len(points_px):
                continue
            ax, ay, a_conf = points_px[joint_a]
            bx, by, b_conf = points_px[joint_b]

            # Only draw if both joints are sufficiently visible
            if a_conf < config.MIN_VISIBILITY or b_conf < config.MIN_VISIBILITY:
                continue

            cv2.line(
                vis,
                (ax, ay), (bx, by),
                config.SKELETON_COLOR_BONE,
                config.SKELETON_LINE_THICKNESS
            )

        # Draw joint circles
        for (x_px, y_px, vis_conf) in points_px:
            if vis_conf < config.MIN_VISIBILITY:
                continue
            cv2.circle(
                vis,
                (x_px, y_px),
                config.SKELETON_DOT_RADIUS,
                config.SKELETON_COLOR_JOINT,
                -1  # Filled circle
            )

        # Draw activity label and confidence on the skeleton image
        label_text = f"Activity: {activity_label}"
        conf_text = f"Confidence: {confidence * 100:.1f}%"
        privacy_text = "[ PRIVACY MODE — NO VIDEO STORED ]"

        cv2.putText(vis, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 180), 2)
        cv2.putText(vis, conf_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 150), 1)
        cv2.putText(vis, privacy_text, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (100, 100, 255), 1)

        return vis

    def draw_on_frame(self, frame: np.ndarray, raw_landmarks) -> np.ndarray:
        """
        Draw MediaPipe skeleton on the original frame.

        USE ONLY FOR DEBUG/DEVELOPMENT — NOT for the production dashboard.
        In production, use draw_skeleton_privacy() instead.

        Args:
            frame:         Original BGR camera frame
            raw_landmarks: raw_landmarks from estimate() result

        Returns:
            Frame with skeleton overlay (copy of original)
        """
        debug_frame = frame.copy()

        if raw_landmarks is not None:
            self.mp_drawing.draw_landmarks(
                debug_frame,
                raw_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
            )

        return debug_frame

    def get_detection_rate(self) -> float:
        """Return fraction of frames where a person was detected."""
        if self._frame_count == 0:
            return 0.0
        return self._detection_count / self._frame_count

    def close(self) -> None:
        """Release MediaPipe resources."""
        self.pose.close()
        logger.info("PoseEstimator closed.")
