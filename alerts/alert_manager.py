"""
alerts/alert_manager.py — Emergency Alert System
==================================================
PURPOSE:
    When a fall is confirmed by the temporal validator, this module
    triggers the emergency alert through multiple channels.

ALERT CHANNELS:
    1. Dashboard Alert    (via shared state that Streamlit reads)
    2. Desktop Notification  (using plyer library)
    3. Console/Log Warning
    4. Optional: Webhook (send HTTP POST to caregiver's server)

PRIVACY:
    Alerts contain ONLY:
    - Time
    - Activity classification
    - Confidence
    - Duration
    - Event ID

    Alerts do NOT contain:
    - Camera images
    - Raw video
    - Face images
    - Any identifying information

DISCLAIMER:
    This is a prototype system.
    It does NOT replace professional medical monitoring or emergency services.
"""

import time
import logging
import threading
from datetime import datetime
from typing import Dict, Any, Optional
import config

logger = logging.getLogger(__name__)

# Shared alert state (read by the dashboard)
# In a production system, use a proper message queue
_latest_alert: Dict[str, Any] = {}
_alert_lock = threading.Lock()


def get_latest_alert() -> Dict[str, Any]:
    """Return the most recent alert (thread-safe)."""
    with _alert_lock:
        return _latest_alert.copy()


def clear_alert() -> None:
    """Clear the current alert (called by dashboard after display)."""
    global _latest_alert
    with _alert_lock:
        _latest_alert = {}


class AlertManager:
    """Manages emergency alert dispatch for the elder care system."""

    def __init__(self):
        self._alert_count = 0
        self._last_alert_time = 0.0
        self._cooldown = config.ALERT_COOLDOWN_SECONDS

        # Try importing plyer for desktop notifications
        try:
            from plyer import notification
            self._plyer_available = True
        except ImportError:
            self._plyer_available = False
            logger.warning(
                "plyer not installed. Desktop notifications disabled. "
                "Install with: pip install plyer"
            )

    def send_alert(
        self,
        activity: str,
        confidence: float,
        duration: float,
        irc_value: float = 0.0,
        event_id: str = "",
        replay_info: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Dispatch emergency alert through all configured channels.

        PRIVACY NOTE: No raw video/pixel data included in alerts.
        Only metadata and privacy-preserving skeletal vector replay animation.

        Args:
            activity:    Activity at time of fall
            confidence:  ML confidence
            duration:    How long the fall has been ongoing
            irc_value:   IRC value at time of detection
            event_id:    Unique event identifier
            replay_info: Dictionary containing skeletal vector replay animation paths

        Returns:
            True if alert was successfully sent.
        """
        # Check cooldown
        now = time.time()
        if now - self._last_alert_time < self._cooldown:
            remaining = self._cooldown - (now - self._last_alert_time)
            logger.warning(
                f"Alert skipped (cooldown: {remaining:.1f}s remaining)"
            )
            return False

        self._alert_count += 1
        self._last_alert_time = now
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        alert_data = {
            "active": True,
            "event_id": event_id or f"ALERT_{self._alert_count}",
            "timestamp": timestamp,
            "activity": activity,
            "confidence": confidence,
            "duration": duration,
            "irc_value": irc_value,
            "alert_number": self._alert_count,
            "replay_html": replay_info.get("html_path") if replay_info else None,
            "replay_gif": replay_info.get("gif_path") if replay_info else None,
            "privacy_guaranteed": True,
        }

        # 1. Update shared dashboard state
        self._update_dashboard_state(alert_data)

        # 2. Desktop notification
        if config.ENABLE_DESKTOP_NOTIFICATION:
            self._send_desktop_notification(alert_data)

        # 3. Console warning
        self._print_console_alert(alert_data)

        # 4. Optional webhook
        if config.WEBHOOK_URL:
            threading.Thread(
                target=self._send_webhook,
                args=(alert_data,),
                daemon=True,
            ).start()

        return True

    def _update_dashboard_state(self, alert_data: Dict) -> None:
        """Update the shared alert state for the Streamlit dashboard."""
        global _latest_alert
        with _alert_lock:
            _latest_alert = alert_data
        logger.info(f"Dashboard alert state updated: {alert_data['event_id']}")

    def _send_desktop_notification(self, alert_data: Dict) -> None:
        """Send a desktop popup notification."""
        if not self._plyer_available:
            return
        try:
            from plyer import notification
            notification.notify(
                title="🚨 ELDER CARE ALERT — FALL DETECTED",
                message=(
                    f"Time: {alert_data['timestamp']}\n"
                    f"Activity: {alert_data['activity']}\n"
                    f"Confidence: {alert_data['confidence']*100:.1f}%\n"
                    f"Duration: {alert_data['duration']:.1f}s\n"
                    f"Event: {alert_data['event_id']}"
                ),
                timeout=30,  # Show for 30 seconds
            )
            logger.info("Desktop notification sent.")
        except Exception as e:
            logger.error(f"Desktop notification error: {e}")

    def _print_console_alert(self, alert_data: Dict) -> None:
        """Print a clear warning to the console."""
        line = "=" * 60
        logger.warning(f"\n{line}")
        logger.warning("🚨 EMERGENCY ALERT — FALL/DISTRESS DETECTED 🚨")
        logger.warning(f"  Event ID:   {alert_data['event_id']}")
        logger.warning(f"  Time:       {alert_data['timestamp']}")
        logger.warning(f"  Activity:   {alert_data['activity']}")
        logger.warning(f"  Confidence: {alert_data['confidence']*100:.1f}%")
        logger.warning(f"  Duration:   {alert_data['duration']:.1f} seconds")
        logger.warning(f"  IRC Value:  {alert_data['irc_value']:.4f}")
        logger.warning(
            "\n  ⚠️  DISCLAIMER: This is a prototype monitoring system.")
        logger.warning(
            "     It does not replace professional medical care.")
        logger.warning(f"{line}\n")

    def _send_webhook(self, alert_data: Dict) -> None:
        """
        Send alert data via HTTP POST to a webhook URL.

        Optional feature for remote caregiver notification.
        PRIVACY: Only sends metadata, not video.
        Only runs if WEBHOOK_URL is configured in config.py.
        """
        try:
            import requests
            response = requests.post(
                config.WEBHOOK_URL,
                json=alert_data,
                timeout=10,
            )
            logger.info(f"Webhook sent: HTTP {response.status_code}")
        except Exception as e:
            logger.error(f"Webhook error: {e}")
