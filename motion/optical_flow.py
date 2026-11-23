"""
motion/optical_flow.py — Lucas-Kanade Optical Flow
====================================================
PURPOSE:
    Estimate how pixels/feature points are MOVING between frames.
    This gives us velocity information: how fast and in what direction
    parts of the scene are moving.

WHY OPTICAL FLOW?
    Simple frame differencing (Module 2) only tells us WHERE motion is.
    Optical flow tells us HOW FAST and IN WHICH DIRECTION motion is happening.
    This is important for:
    - Measuring the speed of falling motion
    - Detecting the direction of movement
    - Providing the velocity input to the kinetic energy proxy feature

LUCAS-KANADE METHOD:
    Developed by Bruce D. Lucas and Takeo Kanade (1981).
    
    Assumptions:
    1. Pixels in a small neighborhood move together (local smoothness)
    2. Movement is small between consecutive frames (small displacement)
    3. Pixel brightness doesn't change much (brightness constancy)
    
    Works by tracking "feature points" — distinctive corners/edges
    detected using the Shi-Tomasi corner detector.

WHAT ARE Vx AND Vy?
    Vx = horizontal velocity (pixels per frame, positive = rightward)
    Vy = vertical velocity   (pixels per frame, positive = downward)
    
    Magnitude = sqrt(Vx² + Vy²) = speed of movement
    Direction = atan2(Vy, Vx)   = angle of movement
"""

import cv2
import numpy as np
import logging
from typing import Optional, Dict, Any, Tuple
import config

logger = logging.getLogger(__name__)


class OpticalFlowAnalyzer:
    """
    Computes Lucas-Kanade sparse optical flow between consecutive frames.

    Sparse optical flow = tracks specific feature points (not every pixel).
    This is faster than dense optical flow (like Farneback) and suitable
    for edge devices.
    """

    def __init__(
        self,
        max_corners: int = config.LK_MAX_CORNERS,
        quality_level: float = config.LK_QUALITY_LEVEL,
        min_distance: int = config.LK_MIN_DISTANCE,
        block_size: int = config.LK_BLOCK_SIZE,
        win_size: Tuple[int, int] = config.LK_WIN_SIZE,
        max_level: int = config.LK_MAX_LEVEL,
        magnitude_threshold: float = config.FLOW_MAGNITUDE_THRESHOLD,
    ):
        """
        Args:
            max_corners:         Maximum number of feature points to track
            quality_level:       Minimum quality for corner detection (0-1)
            min_distance:        Minimum pixel distance between corners
            block_size:          Size of neighborhood for corner detection
            win_size:            LK window size (affects tracking accuracy)
            max_level:           Pyramid levels (handles larger movements)
            magnitude_threshold: Minimum flow magnitude to consider moving
        """
        # Shi-Tomasi corner detection parameters
        self.feature_params = dict(
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
            blockSize=block_size,
        )

        # Lucas-Kanade optical flow parameters
        self.lk_params = dict(
            winSize=win_size,
            maxLevel=max_level,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                10,   # Max iterations
                0.03, # Convergence epsilon
            ),
        )

        self.magnitude_threshold = magnitude_threshold

        # Store previous points for next frame
        self._prev_points: Optional[np.ndarray] = None
        self._frame_count = 0

    def detect_feature_points(self, gray_frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect "good features to track" (corners) in a grayscale frame.

        Uses Shi-Tomasi corner detection — finds points where the image
        has strong gradients in multiple directions (corners are stable
        and easy to track).

        Args:
            gray_frame: Grayscale image (numpy array, single channel)

        Returns:
            Array of (x, y) corner points, or None if none found.
            Shape: (N, 1, 2) where N = number of corners found
        """
        points = cv2.goodFeaturesToTrack(gray_frame, mask=None, **self.feature_params)
        return points

    def calculate_optical_flow(
        self,
        previous_gray: np.ndarray,
        current_gray: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Calculate optical flow between two consecutive frames.

        Args:
            previous_gray: Previous frame (blurred grayscale)
            current_gray:  Current frame  (blurred grayscale)

        Returns:
            Dictionary with:
                flow_vectors    (ndarray): Displacement vectors for each point
                mean_vx         (float):  Average horizontal velocity
                mean_vy         (float):  Average vertical velocity
                mean_magnitude  (float):  Average movement speed
                mean_direction  (float):  Average movement direction (degrees)
                num_points      (int):    Number of points successfully tracked
                has_flow        (bool):   True if any significant flow detected
        """
        self._frame_count += 1

        # Default result if flow cannot be computed
        empty_result = {
            "flow_vectors": np.array([]),
            "mean_vx": 0.0,
            "mean_vy": 0.0,
            "mean_magnitude": 0.0,
            "mean_direction": 0.0,
            "num_points": 0,
            "has_flow": False,
        }

        # Step 1: Detect feature points in the PREVIOUS frame
        prev_points = self.detect_feature_points(previous_gray)

        if prev_points is None or len(prev_points) == 0:
            logger.debug("No feature points detected for optical flow.")
            return empty_result

        # Step 2: Track where those points MOVED TO in the current frame
        # cv2.calcOpticalFlowPyrLK returns:
        # - next_pts: where each point ended up in the current frame
        # - status:   1 if point was successfully tracked, 0 if lost
        # - err:      tracking error for each point
        next_points, status, err = cv2.calcOpticalFlowPyrLK(
            previous_gray,
            current_gray,
            prev_points,
            None,
            **self.lk_params,
        )

        if next_points is None:
            return empty_result

        # Step 3: Keep only successfully tracked points (status == 1)
        good_mask = status.ravel() == 1
        good_prev = prev_points[good_mask]
        good_next = next_points[good_mask]

        if len(good_prev) == 0:
            return empty_result

        # Step 4: Calculate displacement vectors (Vx, Vy) for each point
        # displacement = where it went - where it was
        displacement = good_next - good_prev  # Shape: (N, 1, 2)
        displacement = displacement.reshape(-1, 2)  # Shape: (N, 2)

        vx = displacement[:, 0]  # Horizontal velocity (x component)
        vy = displacement[:, 1]  # Vertical velocity   (y component)

        # Step 5: Calculate magnitude and direction for each tracked point
        # Magnitude = sqrt(Vx² + Vy²) = speed of movement
        magnitudes = np.sqrt(vx**2 + vy**2)

        # Direction = atan2(Vy, Vx) converted from radians to degrees
        directions = np.degrees(np.arctan2(vy, vx))

        # Step 6: Filter out very small movements (below threshold = noise)
        significant_mask = magnitudes > self.magnitude_threshold
        num_significant = int(np.sum(significant_mask))

        # Step 7: Compute summary statistics
        mean_vx = float(np.mean(vx)) if len(vx) > 0 else 0.0
        mean_vy = float(np.mean(vy)) if len(vy) > 0 else 0.0
        mean_magnitude = float(np.mean(magnitudes)) if len(magnitudes) > 0 else 0.0
        mean_direction = float(np.mean(directions)) if len(directions) > 0 else 0.0

        has_flow = num_significant > 0 and mean_magnitude > self.magnitude_threshold

        result = {
            "flow_vectors": displacement,
            "mean_vx": mean_vx,
            "mean_vy": mean_vy,
            "mean_magnitude": mean_magnitude,
            "mean_direction": mean_direction,
            "num_points": len(good_prev),
            "has_flow": has_flow,
            "good_prev_points": good_prev,  # For visualization
            "good_next_points": good_next,  # For visualization
        }

        logger.debug(
            f"Optical flow: Vx={mean_vx:.2f}, Vy={mean_vy:.2f}, "
            f"Magnitude={mean_magnitude:.2f}, Points={len(good_prev)}"
        )

        return result

    def draw_flow_visualization(
        self,
        frame: np.ndarray,
        flow_result: Dict[str, Any],
        color: Tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        """
        Draw optical flow arrows on a frame copy for visualization.

        PRIVACY NOTE: This draws onto a COPY of the frame for debug purposes.
        In the production dashboard, only skeleton visualization is shown.

        Args:
            frame:       Frame to draw on (will be copied, not modified)
            flow_result: Result from calculate_optical_flow()
            color:       Arrow color in BGR

        Returns:
            Frame copy with flow arrows drawn.
        """
        vis_frame = frame.copy()

        if not flow_result.get("has_flow", False):
            return vis_frame

        prev_pts = flow_result.get("good_prev_points")
        next_pts = flow_result.get("good_next_points")

        if prev_pts is None or next_pts is None:
            return vis_frame

        for prev, nxt in zip(prev_pts, next_pts):
            p_x, p_y = int(prev[0][0]), int(prev[0][1])
            n_x, n_y = int(nxt[0][0]), int(nxt[0][1])

            # Draw circle at starting point
            cv2.circle(vis_frame, (p_x, p_y), 3, color, -1)

            # Draw arrow showing movement direction
            cv2.arrowedLine(
                vis_frame, (p_x, p_y), (n_x, n_y), color, 1, tipLength=0.3
            )

        return vis_frame
