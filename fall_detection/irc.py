"""
fall_detection/irc.py — Instantaneous Rate of Change (IRC)
============================================================
PURPOSE:
    The IRC measures HOW SUDDENLY movement intensity changes.

    A normal activity change (standing → sitting) happens gradually.
    A fall happens SUDDENLY with a very rapid change in movement intensity.

    IRC = (current_intensity - past_intensity) / time_elapsed

HOW IT WORKS:
    We compare current movement intensity to the intensity N frames ago.
    A large positive IRC = sudden increase in movement = possible fall event.
    A large negative IRC = sudden decrease = possible collapse/freezing.

    Example:
        Frame 1: intensity = 0.02 (person standing still)
        Frame 2: intensity = 0.03 (slight movement)
        Frame 3: intensity = 0.40 (SUDDEN LARGE MOVEMENT — possible fall!)
        
        IRC between frame 1 and 3 = (0.40 - 0.02) / 2 = 0.19

THRESHOLD:
    config.IRC_THRESHOLD = 0.25 (proposed value from synopsis)
    
    IMPORTANT: This is a PROTOTYPE threshold.
    It has NOT been scientifically validated on a large dataset.
    It should be tuned using real validation data in future work.
    
    When IRC > threshold → flags potential sudden movement event.
    This is ONE of multiple signals used in fall detection.
    It is NOT sufficient alone to declare a fall.

CONFIGURABLE:
    IRC_THRESHOLD  in config.py
    IRC_WINDOW     in config.py (number of frames to look back)
"""

import numpy as np
import logging
from collections import deque
from typing import Dict, Any
import config

logger = logging.getLogger(__name__)


class IRCCalculator:
    """
    Computes the Instantaneous Rate of Change of movement intensity.

    Used as one of multiple signals in the fall detection pipeline.
    """

    def __init__(
        self,
        window: int = config.IRC_WINDOW,
        threshold: float = config.IRC_THRESHOLD,
    ):
        """
        Args:
            window:    Number of past frames to compare against
            threshold: IRC value above which sudden movement is flagged
        """
        self.window = window
        self.threshold = threshold

        # History of recent movement intensity values
        self.intensity_history: deque = deque(maxlen=window + 1)

        logger.info(
            f"IRCCalculator: window={window}, threshold={threshold} "
            f"[PROTOTYPE — not scientifically validated]"
        )

    def update(self, movement_intensity: float) -> Dict[str, Any]:
        """
        Update IRC with the latest movement intensity value.

        Args:
            movement_intensity: Current frame's movement intensity
                                (from feature_extraction.py feature[5])

        Returns:
            Dictionary:
                irc_value           (float): Computed IRC value
                sudden_movement     (bool):  True if IRC > threshold
                above_threshold     (bool):  Same as sudden_movement
                history_length      (int):   Frames collected so far
        """
        self.intensity_history.append(movement_intensity)

        if len(self.intensity_history) < self.window + 1:
            # Not enough history yet
            return {
                "irc_value": 0.0,
                "sudden_movement": False,
                "above_threshold": False,
                "history_length": len(self.intensity_history),
            }

        # Get oldest and newest intensity in the window
        history_list = list(self.intensity_history)
        oldest = history_list[0]       # N frames ago
        newest = history_list[-1]      # Current frame

        # IRC = change in intensity / number of frames
        # This gives us "change per frame" (rate of change)
        irc_value = (newest - oldest) / float(self.window)
        
        # We use absolute IRC to catch both sudden increases (fall)
        # and sudden decreases (collapse/freeze)
        irc_abs = abs(irc_value)

        sudden_movement = irc_abs > self.threshold

        if sudden_movement:
            logger.debug(
                f"IRC triggered: {irc_abs:.4f} > threshold {self.threshold}"
            )

        return {
            "irc_value": float(irc_value),
            "irc_abs": float(irc_abs),
            "sudden_movement": sudden_movement,
            "above_threshold": sudden_movement,
            "history_length": len(self.intensity_history),
        }

    def reset(self) -> None:
        """Clear IRC history."""
        self.intensity_history.clear()
