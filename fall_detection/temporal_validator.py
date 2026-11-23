"""
fall_detection/temporal_validator.py — Temporal Fall Validation
================================================================
PURPOSE:
    Avoid false alarms by requiring that a "fall" signal PERSISTS
    for a sufficient duration before triggering an emergency alert.

THE PROBLEM WITH INSTANT CLASSIFICATION:
    If we immediately alert on the first "Fallen" prediction:
    - Sitting down quickly → looks like a fall briefly
    - Bending down to pick something → body goes horizontal
    - Playing with a child on the floor → lying down intentionally
    
    All of these would trigger false alarms.

HOW TEMPORAL VALIDATION WORKS:
    
    STATE MACHINE with these states:
    
    NORMAL → POTENTIAL_FALL → FALLEN_CONFIRMED → ALERTING → COOLDOWN

    NORMAL:
        All good. No fall detected.
        
    POTENTIAL_FALL:
        A fall-like event was detected.
        We START a timer.
        Give the person a chance to recover (get up).
        If they recover within RECOVERY_WINDOW seconds → back to NORMAL.
        
    FALLEN_CONFIRMED:
        Fall signal persisted for FALL_CONFIRMATION_SECONDS.
        Person did NOT recover.
        This is now considered a genuine fall.
        Trigger the emergency alert.
        
    ALERTING:
        Alert has been sent.
        Continue monitoring.
        
    COOLDOWN:
        After alert, wait before allowing another alert.
        Prevents repeated alerts for the same fall.

CONFIGURATION:
    FALL_CONFIRMATION_SECONDS  in config.py (default: 5 seconds)
    RECOVERY_WINDOW_SECONDS    in config.py (default: 10 seconds)
    ALERT_COOLDOWN_SECONDS     in config.py (default: 30 seconds)
    FALL_MIN_FRAMES            in config.py (default: 5 frames)
"""

import time
import logging
from enum import Enum
from typing import Dict, Any, Optional
import config

logger = logging.getLogger(__name__)


class FallState(Enum):
    """States in the temporal fall validation state machine."""
    NORMAL = "NORMAL"
    POTENTIAL_FALL = "POTENTIAL_FALL"
    FALLEN_CONFIRMED = "FALLEN_CONFIRMED"
    ALERTING = "ALERTING"
    COOLDOWN = "COOLDOWN"


class TemporalValidator:
    """
    State machine that validates potential fall events over time.

    Receives per-frame fall signals and decides when to actually
    trigger an emergency alert.
    """

    def __init__(
        self,
        confirmation_seconds: float = config.FALL_CONFIRMATION_SECONDS,
        recovery_window_seconds: float = config.RECOVERY_WINDOW_SECONDS,
        cooldown_seconds: float = config.ALERT_COOLDOWN_SECONDS,
        min_frames: int = config.FALL_MIN_FRAMES,
    ):
        self.confirmation_seconds = confirmation_seconds
        self.recovery_window_seconds = recovery_window_seconds
        self.cooldown_seconds = cooldown_seconds
        self.min_frames = min_frames

        self.state = FallState.NORMAL

        # Timing
        self._fall_start_time: Optional[float] = None
        self._alert_time: Optional[float] = None
        self._cooldown_start: Optional[float] = None

        # Frame counters
        self._consecutive_fall_frames = 0
        self._total_alerts_sent = 0

        logger.info(
            f"TemporalValidator: confirmation={confirmation_seconds}s, "
            f"recovery={recovery_window_seconds}s, "
            f"cooldown={cooldown_seconds}s"
        )

    def update(
        self,
        fall_signal: bool,
        activity: str,
        confidence: float,
        features: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Update the state machine with the latest frame's fall signal.

        Args:
            fall_signal:  True if this frame's analysis indicates a fall
            activity:     Current classified activity (e.g., "Fallen")
            confidence:   ML model confidence for this prediction
            features:     Optional dictionary of current frame features

        Returns:
            Dictionary:
                state              (str):  Current state name
                should_alert       (bool): True if alert should be triggered NOW
                fall_duration      (float): Seconds since fall started (0 if normal)
                consecutive_frames (int):  Consecutive fall frames count
                message            (str):  Human-readable state description
        """
        now = time.time()
        should_alert = False
        fall_duration = 0.0

        # ---- STATE TRANSITIONS -----------------------------------------------

        if self.state == FallState.NORMAL:
            if fall_signal:
                self._consecutive_fall_frames += 1
                if self._consecutive_fall_frames >= self.min_frames:
                    # Enough consecutive fall frames → enter POTENTIAL_FALL
                    self.state = FallState.POTENTIAL_FALL
                    self._fall_start_time = now
                    logger.info(
                        f"⚠️  Potential fall detected! Activity={activity}, "
                        f"Confidence={confidence:.2f}. "
                        f"Starting {self.confirmation_seconds}s validation..."
                    )
            else:
                self._consecutive_fall_frames = 0

        elif self.state == FallState.POTENTIAL_FALL:
            fall_duration = now - self._fall_start_time

            if not fall_signal:
                # Person recovered! Back to normal.
                logger.info(
                    f"✅ Person recovered after {fall_duration:.1f}s. "
                    f"Resetting fall state."
                )
                self._reset_fall_state()

            elif fall_duration >= self.confirmation_seconds:
                # Fall persisted long enough — confirm it
                self.state = FallState.FALLEN_CONFIRMED
                logger.warning(
                    f"🚨 FALL CONFIRMED after {fall_duration:.1f}s of "
                    f"continuous fall signal!"
                )

        elif self.state == FallState.FALLEN_CONFIRMED:
            fall_duration = now - self._fall_start_time
            should_alert = True
            self.state = FallState.ALERTING
            self._alert_time = now
            self._total_alerts_sent += 1
            logger.warning(f"🚨 ALERT TRIGGERED! Alert #{self._total_alerts_sent}")

        elif self.state == FallState.ALERTING:
            fall_duration = now - self._fall_start_time

            if not fall_signal:
                # Person recovered after alert was sent
                logger.info("✅ Person appears to have recovered after alert.")

            # Move to cooldown after some time
            if self._alert_time and (now - self._alert_time) > 5.0:
                self.state = FallState.COOLDOWN
                self._cooldown_start = now

        elif self.state == FallState.COOLDOWN:
            cooldown_elapsed = now - self._cooldown_start
            if cooldown_elapsed >= self.cooldown_seconds:
                logger.info(
                    f"Cooldown complete ({cooldown_elapsed:.1f}s). "
                    f"Returning to NORMAL monitoring."
                )
                self._reset_fall_state()

        # Build result
        state_messages = {
            FallState.NORMAL: "✅ Normal monitoring active",
            FallState.POTENTIAL_FALL: f"⚠️  Potential fall — validating ({fall_duration:.1f}s / {self.confirmation_seconds}s)",
            FallState.FALLEN_CONFIRMED: "🚨 Fall confirmed — generating alert",
            FallState.ALERTING: f"🚨 Alert sent — monitoring ({fall_duration:.1f}s elapsed)",
            FallState.COOLDOWN: "🔄 Cooldown period",
        }

        return {
            "state": self.state.value,
            "should_alert": should_alert,
            "fall_duration": fall_duration,
            "consecutive_frames": self._consecutive_fall_frames,
            "message": state_messages.get(self.state, "Unknown"),
            "total_alerts": self._total_alerts_sent,
        }

    def _reset_fall_state(self) -> None:
        """Reset to NORMAL state."""
        self.state = FallState.NORMAL
        self._fall_start_time = None
        self._alert_time = None
        self._cooldown_start = None
        self._consecutive_fall_frames = 0

    def force_reset(self) -> None:
        """Force reset (e.g., when camera is restarted)."""
        self._reset_fall_state()
        logger.info("TemporalValidator force-reset to NORMAL.")

    def get_state(self) -> FallState:
        """Return current state."""
        return self.state

    def is_in_alert(self) -> bool:
        """True if currently in ALERTING or FALLEN_CONFIRMED state."""
        return self.state in [FallState.FALLEN_CONFIRMED, FallState.ALERTING]
