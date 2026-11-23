"""
fall_detection/fall_detector.py — Multi-Signal Fall Detection
==============================================================
PURPOSE:
    Combines MULTIPLE signals to decide if a fall has occurred.
    Using only one signal (e.g., just ML classification) causes too
    many false positives. Using multiple signals together increases
    reliability.

THE 8 SIGNALS USED:
    1. ML Activity Classification  → "Fallen" label from Random Forest
    2. Aspect Ratio                → body width > height = horizontal
    3. Head Vertical Velocity      → rapid downward head movement
    4. Head Vertical Acceleration  → sudden change in head speed
    5. Kinetic Energy Proxy        → high total body movement
    6. IRC                         → sudden change in movement intensity
    7. Body Vertical Displacement  → overall body dropped significantly
    8. Post-event Inactivity       → person stays horizontal = more serious

VOTING MECHANISM:
    Each signal casts a "vote" (True/False).
    If votes >= VOTE_THRESHOLD → potential fall detected.
    This vote goes to the TemporalValidator which then decides
    whether to confirm and alert.

    This is a PROTOTYPE threshold-based approach.
    In future work, a proper validation study would determine
    optimal thresholds and weights.

ACADEMIC HONESTY NOTE:
    Threshold values in this file are PROPOSED PROTOTYPE values.
    They have not been validated on a large clinical dataset.
    Real deployment would require proper testing with elderly participants.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional
import config

logger = logging.getLogger(__name__)


class FallDetector:
    """
    Multi-signal fall detector.

    Combines ML classification, geometric features, and movement
    analysis to generate a fall signal for the TemporalValidator.
    """

    # Number of votes required to flag a potential fall
    # (out of 6 total signals checked)
    VOTE_THRESHOLD = 3

    def __init__(self):
        # Thresholds (all are prototype values — document clearly in report)
        self.aspect_ratio_threshold = config.FALL_ASPECT_RATIO_THRESHOLD
        self.head_velocity_threshold = config.HEAD_VELOCITY_THRESHOLD
        self.head_accel_threshold = config.HEAD_ACCELERATION_THRESHOLD

        logger.info(
            "FallDetector initialized. "
            "[NOTE: All thresholds are prototype values, not clinically validated]"
        )

    def analyze(
        self,
        activity: str,
        confidence: float,
        features: Optional[np.ndarray],
        irc_result: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Analyze current frame for fall indicators.

        Args:
            activity:    ML-predicted activity string (e.g., "Fallen")
            confidence:  ML model confidence (0-1)
            features:    Feature vector from FeatureExtractor [9 values]
            irc_result:  Result from IRCCalculator.update()

        Returns:
            Dictionary:
                fall_signal      (bool): True if fall likely
                vote_count       (int):  How many signals voted "fall"
                votes            (dict): Individual signal votes
                reason           (str):  Human-readable explanation
        """
        votes = {}
        reasons = []

        # ---- SIGNAL 1: ML Classification ------------------------------------
        # ML says "Fallen" with sufficient confidence
        ml_fallen = (activity == "Fallen" and confidence >= 0.5)
        votes["ml_classification"] = ml_fallen
        if ml_fallen:
            reasons.append(f"ML: '{activity}' ({confidence:.0%})")

        if features is not None and len(features) >= 9:
            # ---- SIGNAL 2: Aspect Ratio -------------------------------------
            aspect_ratio = features[0]
            ar_fallen = aspect_ratio > self.aspect_ratio_threshold
            votes["aspect_ratio"] = ar_fallen
            if ar_fallen:
                reasons.append(f"AR={aspect_ratio:.2f}>{self.aspect_ratio_threshold}")

            # ---- SIGNAL 3: Head Vertical Velocity ---------------------------
            # Positive = head moving DOWN quickly
            head_vy = features[1]
            hvy_fallen = head_vy > self.head_velocity_threshold
            votes["head_velocity"] = hvy_fallen
            if hvy_fallen:
                reasons.append(f"HeadVy={head_vy:.4f}")

            # ---- SIGNAL 4: Head Vertical Acceleration -----------------------
            head_accel = features[2]
            haccel_fallen = abs(head_accel) > self.head_accel_threshold
            votes["head_acceleration"] = haccel_fallen
            if haccel_fallen:
                reasons.append(f"HeadAccel={head_accel:.4f}")

            # ---- SIGNAL 5: Kinetic Energy Proxy ------------------------------
            # High kinetic energy during fall event
            kinetic_energy = features[3]
            ke_fallen = kinetic_energy > 0.02  # Prototype threshold
            votes["kinetic_energy"] = ke_fallen
            if ke_fallen:
                reasons.append(f"KE={kinetic_energy:.4f}")

            # ---- SIGNAL 6: Body Vertical Displacement ------------------------
            body_vy = features[4]
            bvy_fallen = body_vy > self.head_velocity_threshold
            votes["body_displacement"] = bvy_fallen
            if bvy_fallen:
                reasons.append(f"BodyVy={body_vy:.4f}")
        else:
            # No features available — mark as False
            votes["aspect_ratio"] = False
            votes["head_velocity"] = False
            votes["head_acceleration"] = False
            votes["kinetic_energy"] = False
            votes["body_displacement"] = False

        # ---- SIGNAL 7: IRC (Instantaneous Rate of Change) -------------------
        irc_fallen = False
        if irc_result:
            irc_fallen = irc_result.get("sudden_movement", False)
        votes["irc"] = irc_fallen
        if irc_fallen:
            reasons.append(f"IRC={irc_result.get('irc_abs', 0):.4f}")

        # ---- COUNT VOTES ----------------------------------------------------
        vote_count = sum(1 for v in votes.values() if v)
        fall_signal = vote_count >= self.VOTE_THRESHOLD

        reason_str = " | ".join(reasons) if reasons else "No fall indicators"

        if fall_signal:
            logger.debug(
                f"Fall signal: {vote_count}/{len(votes)} votes. "
                f"Reasons: {reason_str}"
            )

        return {
            "fall_signal": fall_signal,
            "vote_count": vote_count,
            "total_votes": len(votes),
            "votes": votes,
            "reason": reason_str,
        }
