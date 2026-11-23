"""
features/feature_extraction.py — Spatial-Temporal Feature Extraction
======================================================================
PURPOSE:
    Convert 17 skeletal keypoints into a compact numerical feature vector
    that captures the POSTURE and MOVEMENT of the person.

    The ML model uses these features — not raw frames or raw keypoints.

WHY FEATURES INSTEAD OF RAW KEYPOINTS?
    Raw keypoints have 17×3 = 51 values per frame.
    These raw values depend heavily on:
    - Where the person is standing in the frame
    - Camera resolution
    - Person's actual height
    
    By computing derived features (ratios, velocities, accelerations),
    we get values that are:
    - Position-independent (same for left/right side of frame)
    - Scale-independent (works for tall and short people)
    - More meaningful for fall vs standing classification

FEATURE LIST:
    [0] aspect_ratio            — body width / body height (falls increase this)
    [1] head_vertical_velocity  — how fast head is moving up/down
    [2] head_vertical_accel     — rate of change of head velocity
    [3] kinetic_energy_proxy    — normalized proxy for total body movement
    [4] body_vertical_velocity  — how fast body center is moving down
    [5] movement_intensity      — total skeletal movement across joints
    [6] shoulder_velocity       — how fast shoulders are moving
    [7] hip_velocity            — how fast hips are moving
    [8] knee_velocity           — how fast knees are moving

NOTES ON KINETIC ENERGY PROXY:
    We do NOT claim this is the physical kinetic energy (½mv²).
    We don't know the person's actual mass or absolute velocity.
    This is a NORMALIZED PROXY: sqrt(mean of squared joint displacements).
    It correlates with actual kinetic energy but is not the same.
    This is clearly documented here and in the project report.
"""

import numpy as np
import logging
from typing import List, Optional, Dict, Any
import config

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extracts spatial-temporal features from consecutive skeletal keypoints.

    Requires two consecutive frames of keypoints to compute velocities.
    Uses keypoint indices defined in config.py.
    """

    def __init__(self):
        # Store previous keypoints for velocity calculation
        self._prev_keypoints: Optional[List[List[float]]] = None
        self._prev_velocity: Optional[float] = None  # For acceleration
        self._frame_time = 1.0 / config.TARGET_FPS   # Seconds per frame

    # =========================================================================
    # GEOMETRY FEATURES (from single frame)
    # =========================================================================

    def compute_bounding_box(
        self, keypoints: List[List[float]]
    ) -> Dict[str, float]:
        """
        Calculate the bounding box of the skeleton.

        We look at ALL visible keypoints and find:
        - minimum x → left edge
        - maximum x → right edge
        - minimum y → top edge
        - maximum y → bottom edge
        - width  = right - left
        - height = bottom - top

        Args:
            keypoints: 17 keypoints [[x, y, conf], ...]

        Returns:
            Dictionary: {x_min, x_max, y_min, y_max, width, height}
        """
        visible = [
            kp for kp in keypoints
            if kp[2] >= config.MIN_VISIBILITY   # Only use visible joints
        ]

        if len(visible) < 3:
            return {"x_min": 0, "x_max": 0, "y_min": 0, "y_max": 0,
                    "width": 0, "height": 0}

        xs = [kp[0] for kp in visible]
        ys = [kp[1] for kp in visible]

        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        return {
            "x_min": x_min,
            "x_max": x_max,
            "y_min": y_min,
            "y_max": y_max,
            "width": x_max - x_min,
            "height": y_max - y_min,
        }

    def compute_aspect_ratio(self, keypoints: List[List[float]]) -> float:
        """
        Compute width-to-height ratio of the skeleton bounding box.

        WHY THIS IS IMPORTANT FOR FALL DETECTION:
            Standing person: body is taller than wide → ratio < 1.0
            Fallen person:   body is wider than tall  → ratio > 1.0

            Example:
            Standing: width=0.2, height=0.8 → ratio = 0.25
            Fallen:   width=0.7, height=0.3 → ratio = 2.33

        Args:
            keypoints: 17 keypoints

        Returns:
            Aspect ratio (width / height). Returns 0 if invalid.
        """
        bbox = self.compute_bounding_box(keypoints)
        height = bbox["height"]

        if height < 1e-6:
            return 0.0

        ratio = bbox["width"] / height
        return float(ratio)

    def get_keypoint(
        self,
        keypoints: List[List[float]],
        idx: int,
    ) -> Optional[List[float]]:
        """
        Get a specific keypoint if it's sufficiently visible.

        Args:
            keypoints: 17 keypoints
            idx:       Index (0-16, see config.py KEYPOINT_NAMES)

        Returns:
            [x, y, confidence] or None if not visible enough.
        """
        if idx >= len(keypoints):
            return None
        kp = keypoints[idx]
        if kp[2] < config.MIN_VISIBILITY:
            return None
        return kp

    def compute_midpoint(
        self,
        kp_a: Optional[List[float]],
        kp_b: Optional[List[float]],
    ) -> Optional[List[float]]:
        """
        Compute midpoint between two keypoints.

        Used for body center = midpoint between left and right hip.

        Returns:
            [x_mid, y_mid, min_confidence] or None if either point missing.
        """
        if kp_a is None or kp_b is None:
            return None
        return [
            (kp_a[0] + kp_b[0]) / 2,
            (kp_a[1] + kp_b[1]) / 2,
            min(kp_a[2], kp_b[2]),  # Use lower confidence of the two
        ]

    # =========================================================================
    # VELOCITY FEATURES (require previous frame)
    # =========================================================================

    def compute_joint_velocity(
        self,
        curr_kp: Optional[List[float]],
        prev_kp: Optional[List[float]],
    ) -> float:
        """
        Compute velocity of a single joint between two consecutive frames.

        velocity = distance moved / time
                 = sqrt((x2-x1)² + (y2-y1)²) / (1/FPS)

        Since coordinates are normalized (0-1), velocity is in
        "normalized units per second."

        Args:
            curr_kp: Current keypoint [x, y, conf]
            prev_kp: Previous keypoint [x, y, conf]

        Returns:
            Velocity magnitude (float). 0 if either point missing.
        """
        if curr_kp is None or prev_kp is None:
            return 0.0

        dx = curr_kp[0] - prev_kp[0]
        dy = curr_kp[1] - prev_kp[1]
        distance = np.sqrt(dx**2 + dy**2)

        # velocity = distance per frame (we'll normalize by FPS if needed)
        return float(distance)

    def compute_all_joint_velocities(
        self,
        curr_kps: List[List[float]],
        prev_kps: List[List[float]],
    ) -> List[float]:
        """
        Compute velocity magnitude for all 17 joints.

        Args:
            curr_kps: Current frame keypoints (17 points)
            prev_kps: Previous frame keypoints (17 points)

        Returns:
            List of 17 velocity values.
        """
        velocities = []
        for i in range(min(len(curr_kps), len(prev_kps))):
            vel = self.compute_joint_velocity(curr_kps[i], prev_kps[i])
            velocities.append(vel)
        return velocities

    def compute_kinetic_energy_proxy(self, velocities: List[float]) -> float:
        """
        Compute a normalized kinetic energy proxy.

        DISCLAIMER:
        This is NOT the physical kinetic energy (½mv²).
        We don't know the person's mass or absolute velocity.
        This is a dimensionless proxy that correlates with movement intensity.

        Formula: sqrt(mean(velocity²)) — Root Mean Square velocity

        Higher values = more energetic/rapid movement.
        Falls tend to have high values during the fall event.

        Args:
            velocities: List of joint velocity magnitudes

        Returns:
            RMS velocity as kinetic energy proxy.
        """
        if not velocities:
            return 0.0
        squared = [v**2 for v in velocities]
        return float(np.sqrt(np.mean(squared)))

    # =========================================================================
    # MAIN FEATURE EXTRACTION
    # =========================================================================

    def extract(
        self,
        current_keypoints: List[List[float]],
    ) -> Optional[np.ndarray]:
        """
        Extract the full feature vector from current keypoints.

        Requires previous frame's keypoints (stored internally).
        On the first frame, returns None (no velocity can be computed).

        Args:
            current_keypoints: 17 keypoints [[x, y, conf], ...] for current frame

        Returns:
            Feature vector as numpy array of shape (9,), or None if
            not enough data yet.
        """
        if self._prev_keypoints is None:
            # First frame — save and wait for next frame
            self._prev_keypoints = current_keypoints
            return None

        prev_kps = self._prev_keypoints
        curr_kps = current_keypoints

        # ---- FEATURE 0: Aspect Ratio ----------------------------------------
        aspect_ratio = self.compute_aspect_ratio(curr_kps)

        # ---- FEATURE 1: Head (Nose) Vertical Velocity -----------------------
        # Positive Vy = moving DOWN (y increases downward in image)
        head_curr = self.get_keypoint(curr_kps, config.IDX_NOSE)
        head_prev = self.get_keypoint(prev_kps, config.IDX_NOSE)

        if head_curr is not None and head_prev is not None:
            head_vy = head_curr[1] - head_prev[1]   # Positive = falling downward
        else:
            head_vy = 0.0

        # ---- FEATURE 2: Head Vertical Acceleration --------------------------
        # Change in vertical velocity compared to last computation
        if self._prev_velocity is not None:
            head_accel = head_vy - self._prev_velocity
        else:
            head_accel = 0.0
        self._prev_velocity = head_vy

        # ---- FEATURE 3: Kinetic Energy Proxy --------------------------------
        all_velocities = self.compute_all_joint_velocities(curr_kps, prev_kps)
        kinetic_energy = self.compute_kinetic_energy_proxy(all_velocities)

        # ---- FEATURE 4: Body Center Vertical Velocity -----------------------
        # Body center = midpoint of left hip and right hip
        l_hip_curr = self.get_keypoint(curr_kps, config.IDX_LEFT_HIP)
        r_hip_curr = self.get_keypoint(curr_kps, config.IDX_RIGHT_HIP)
        l_hip_prev = self.get_keypoint(prev_kps, config.IDX_LEFT_HIP)
        r_hip_prev = self.get_keypoint(prev_kps, config.IDX_RIGHT_HIP)

        center_curr = self.compute_midpoint(l_hip_curr, r_hip_curr)
        center_prev = self.compute_midpoint(l_hip_prev, r_hip_prev)

        if center_curr is not None and center_prev is not None:
            body_vy = center_curr[1] - center_prev[1]
        else:
            body_vy = 0.0

        # ---- FEATURE 5: Movement Intensity ----------------------------------
        # Total sum of all joint velocities (overall body activity)
        movement_intensity = float(np.sum(all_velocities))

        # ---- FEATURE 6: Shoulder Velocity -----------------------------------
        l_sh_curr = self.get_keypoint(curr_kps, config.IDX_LEFT_SHOULDER)
        r_sh_curr = self.get_keypoint(curr_kps, config.IDX_RIGHT_SHOULDER)
        l_sh_prev = self.get_keypoint(prev_kps, config.IDX_LEFT_SHOULDER)
        r_sh_prev = self.get_keypoint(prev_kps, config.IDX_RIGHT_SHOULDER)

        sh_vel_l = self.compute_joint_velocity(l_sh_curr, l_sh_prev)
        sh_vel_r = self.compute_joint_velocity(r_sh_curr, r_sh_prev)
        shoulder_velocity = (sh_vel_l + sh_vel_r) / 2.0

        # ---- FEATURE 7: Hip Velocity ----------------------------------------
        hip_vel_l = self.compute_joint_velocity(l_hip_curr, l_hip_prev)
        hip_vel_r = self.compute_joint_velocity(r_hip_curr, r_hip_prev)
        hip_velocity = (hip_vel_l + hip_vel_r) / 2.0

        # ---- FEATURE 8: Knee Velocity ----------------------------------------
        l_kn_curr = self.get_keypoint(curr_kps, config.IDX_LEFT_KNEE)
        r_kn_curr = self.get_keypoint(curr_kps, config.IDX_RIGHT_KNEE)
        l_kn_prev = self.get_keypoint(prev_kps, config.IDX_LEFT_KNEE)
        r_kn_prev = self.get_keypoint(prev_kps, config.IDX_RIGHT_KNEE)

        kn_vel_l = self.compute_joint_velocity(l_kn_curr, l_kn_prev)
        kn_vel_r = self.compute_joint_velocity(r_kn_curr, r_kn_prev)
        knee_velocity = (kn_vel_l + kn_vel_r) / 2.0

        # ---- ASSEMBLE FEATURE VECTOR ----------------------------------------
        feature_vector = np.array([
            aspect_ratio,          # [0] Width/height ratio
            head_vy,               # [1] Head vertical velocity
            head_accel,            # [2] Head vertical acceleration
            kinetic_energy,        # [3] Kinetic energy proxy (RMS velocity)
            body_vy,               # [4] Body center vertical velocity
            movement_intensity,    # [5] Total movement across all joints
            shoulder_velocity,     # [6] Shoulder joint velocity
            hip_velocity,          # [7] Hip joint velocity
            knee_velocity,         # [8] Knee joint velocity
        ], dtype=np.float32)

        # Update stored keypoints for next frame's velocity calculation
        self._prev_keypoints = current_keypoints

        logger.debug(
            f"Features: AR={aspect_ratio:.3f}, head_vy={head_vy:.4f}, "
            f"KE={kinetic_energy:.4f}, MI={movement_intensity:.4f}"
        )

        return feature_vector

    def get_feature_names(self) -> List[str]:
        """Return names of each feature in the vector (for CSV headers)."""
        return [
            "aspect_ratio",
            "head_vertical_velocity",
            "head_vertical_accel",
            "kinetic_energy_proxy",
            "body_vertical_velocity",
            "movement_intensity",
            "shoulder_velocity",
            "hip_velocity",
            "knee_velocity",
        ]

    def reset(self) -> None:
        """Reset stored previous frame (e.g., when tracking is lost)."""
        self._prev_keypoints = None
        self._prev_velocity = None
