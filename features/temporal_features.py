"""
features/temporal_features.py — Sliding Temporal Window
=========================================================
PURPOSE:
    Instead of classifying from a single frame's features, we collect
    features over a window of consecutive frames and compute statistics.

WHY TEMPORAL WINDOW?
    A fall is an EVENT over time, not just one frame's posture.
    
    Consider:
    - Sitting DOWN quickly: momentary high velocity + position change
    - Actual FALL:          very high velocity + sustained new position
    
    With a single frame, both look similar!
    With a temporal window, we see the PATTERN:
    - Normal sitting: velocity rises then falls as person settles
    - Fall: velocity spikes, then person stays in horizontal position

HOW IT WORKS:
    We maintain a deque (sliding window) of the last N feature vectors.
    
    When the window is full, we compute statistical summaries:
    - mean   (average value across time)
    - std    (variability — falls have high std during event)
    - max    (peak value — falls have very high peaks)
    - min    (baseline)
    
    The temporal feature vector is the CONCATENATION of all these stats.
    
    For 9 base features with 4 stats each = 36 temporal features.

CONFIGURATION:
    TEMPORAL_WINDOW_SIZE in config.py (default: 15 frames)
    At 30 FPS, 15 frames = 0.5 seconds of context
"""

import numpy as np
import logging
from collections import deque
from typing import Optional, List
import config

logger = logging.getLogger(__name__)


class TemporalWindow:
    """
    Maintains a sliding window of feature vectors and computes
    temporal statistics for ML classification.
    """

    def __init__(
        self,
        window_size: int = config.TEMPORAL_WINDOW_SIZE,
        n_features: int = 9,  # Must match FeatureExtractor output
    ):
        """
        Args:
            window_size: Number of frames to keep in the window
            n_features:  Number of features per frame
        """
        self.window_size = window_size
        self.n_features = n_features

        # Sliding window of feature vectors
        # deque automatically removes oldest when full
        self.window: deque = deque(maxlen=window_size)

        logger.info(
            f"TemporalWindow: window_size={window_size}, "
            f"n_features={n_features}"
        )

    def add(self, feature_vector: np.ndarray) -> None:
        """
        Add a new frame's feature vector to the window.

        Args:
            feature_vector: numpy array of shape (n_features,)
        """
        if feature_vector is not None and len(feature_vector) == self.n_features:
            self.window.append(feature_vector.copy())

    def is_ready(self) -> bool:
        """
        Return True when the window has enough frames for classification.
        We need at least half the window filled to classify.
        """
        return len(self.window) >= max(3, self.window_size // 2)

    def compute_temporal_features(self) -> Optional[np.ndarray]:
        """
        Compute statistical summary of features in the current window.

        For each of the 9 base features, we compute:
        - mean: average value (what's the typical state?)
        - std:  standard deviation (how variable is it?)
        - max:  maximum value (what's the peak?)
        - min:  minimum value (what's the baseline?)

        Result: 9 features × 4 stats = 36 temporal features.

        Returns:
            numpy array of shape (36,), or None if window not ready.
        """
        if not self.is_ready():
            return None

        # Stack window frames into matrix: shape (N_frames, N_features)
        window_matrix = np.stack(list(self.window), axis=0)

        # Compute statistics along the time axis (axis=0)
        means = np.mean(window_matrix, axis=0)    # Shape: (9,)
        stds = np.std(window_matrix, axis=0)      # Shape: (9,)
        maxs = np.max(window_matrix, axis=0)      # Shape: (9,)
        mins = np.min(window_matrix, axis=0)      # Shape: (9,)

        # Concatenate all stats into one long vector
        temporal_features = np.concatenate([means, stds, maxs, mins])
        # Shape: (36,)

        return temporal_features.astype(np.float32)

    def get_latest_features(self) -> Optional[np.ndarray]:
        """Return the most recent frame's feature vector (no temporal stats)."""
        if len(self.window) == 0:
            return None
        return self.window[-1].copy()

    def get_temporal_feature_names(self) -> List[str]:
        """
        Return names for all 36 temporal features.
        Used for CSV column headers and debugging.
        """
        base_names = [
            "aspect_ratio", "head_vy", "head_accel",
            "kinetic_energy", "body_vy", "movement_intensity",
            "shoulder_vel", "hip_vel", "knee_vel"
        ]
        names = []
        for stat in ["mean", "std", "max", "min"]:
            for feat in base_names:
                names.append(f"{feat}_{stat}")
        return names  # 36 names total

    def reset(self) -> None:
        """Clear the window (e.g., when tracking is lost)."""
        self.window.clear()
        logger.debug("TemporalWindow cleared.")
