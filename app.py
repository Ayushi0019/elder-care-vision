"""
app.py — Main Elder Care Monitoring Application
================================================
ENTRY POINT for the complete live monitoring system.

HOW TO RUN:
    python app.py

This integrates all modules in the correct pipeline order:

CAMERA
→ FRAME CAPTURE
→ MOTION DETECTION
→ OPTICAL FLOW
→ DOUBLE BACKGROUND FILTERING
→ MORPHOLOGICAL PROCESSING
→ POSE ESTIMATION (17 keypoints)
→ RAW RGB FRAME DISCARDED (privacy)
→ FEATURE EXTRACTION
→ TEMPORAL WINDOW
→ ML CLASSIFICATION
→ IRC CALCULATION
→ MULTI-SIGNAL FALL DETECTION
→ TEMPORAL VALIDATION
→ ALERT (if confirmed)
→ DATABASE LOGGING
→ SKELETON VISUALIZATION

PRIVACY DESIGN:
    - Raw RGB frames are NEVER saved to disk
    - Only skeleton visualization is shown to user
    - Database stores only text/numbers
    - No face recognition

DISCLAIMER:
    This is a prototype system for academic purposes.
    It does NOT replace professional medical monitoring.
"""

import cv2
import os
import sys
import time
import math
import logging
import threading
from datetime import datetime
from typing import Optional

# Setup logging first
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("logs/system.log"),
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)

import config

# Import all modules
import argparse
from camera import CameraCapture
from motion import MotionDetector, OpticalFlowAnalyzer, DoubleBackgroundFilter, MorphologyProcessor
from pose import PoseEstimator
from features import FeatureExtractor, TemporalWindow
from ml import ActivityClassifier
from fall_detection import FallDetector, TemporalValidator, IRCCalculator
from alerts import AlertManager, SkeletalReplayBuffer
from alerts.alert_manager import get_latest_alert
from database import Database

# Shared state for dashboard (thread-safe)
import threading
_monitor_state = {
    "running": False,
    "activity": "Unknown",
    "confidence": 0.0,
    "fps": 0.0,
    "fall_state": "NORMAL",
    "fall_message": "",
    "motion_detected": False,
    "person_detected": False,
    "privacy_status": "RAW VIDEO STORAGE: DISABLED",
    "timestamp": "",
    "skeleton_image": None,  # Latest skeleton visualization
    "irc_value": 0.0,
    "vote_count": 0,
    "aspect_ratio": 0.5,
    "head_accel": 0.0,
    "kinetic_energy": 0.0,
    "replay_html": None,
    "replay_gif": None,
}
_state_lock = threading.Lock()


def get_monitor_state() -> dict:
    """Thread-safe read of monitoring state (for dashboard)."""
    with _state_lock:
        state = _monitor_state.copy()
        state.pop("skeleton_image", None)  # Don't copy image in dict
    return state


def update_state(**kwargs) -> None:
    """Thread-safe update of monitoring state."""
    with _state_lock:
        _monitor_state.update(kwargs)


class ElderCareMonitor:
    """
    Main monitoring application.

    Runs the complete pipeline in a loop, processing camera frames or demo feed.
    """

    def __init__(self, camera_source=config.CAMERA_INDEX, demo_mode: str = "none"):
        logger.info("="*60)
        logger.info("  Elder Care Monitoring System — Starting Up")
        logger.info(f"  Mode: {'DEMO SIMULATION (' + demo_mode + ')' if demo_mode != 'none' else 'LIVE CAMERA'}")
        logger.info("  PROTOTYPE SYSTEM — Not for medical use")
        logger.info("  Privacy Mode: ACTIVE — No raw video stored")
        logger.info("="*60)

        self.demo_mode = demo_mode
        self.camera_source = camera_source
        self.camera = CameraCapture(camera_index=camera_source)
        self.motion_detector = MotionDetector()
        self.optical_flow = OpticalFlowAnalyzer()
        self.dbf = DoubleBackgroundFilter()
        self.morphology = MorphologyProcessor()
        self.pose_estimator = PoseEstimator()
        self.feature_extractor = FeatureExtractor()
        self.temporal_window = TemporalWindow()
        self.classifier = ActivityClassifier()
        self.fall_detector = FallDetector()
        self.temporal_validator = TemporalValidator()
        self.irc_calculator = IRCCalculator()
        self.alert_manager = AlertManager()
        self.replay_buffer = SkeletalReplayBuffer(max_frames=60)
        self.database = Database()

        # Performance tracking
        self._frame_count = 0
        self._start_time = 0.0
        self._last_db_log_frame = 0
        self._running = False

        logger.info(
            f"Classifier ready: {self.classifier.is_ready} "
            f"({'model' if self.classifier.is_ready else 'rule-based fallback'})"
        )

    def run(self) -> None:
        """Start the monitoring loop."""
        if self.demo_mode != "none":
            self._run_demo()
            return

        if not self.camera.start():
            logger.error("❌ Cannot open camera. Exiting.")
            return

        self._running = True
        self._start_time = time.time()
        update_state(running=True, privacy_status="RAW VIDEO STORAGE: DISABLED")

        self.database.log_system_status("STARTED", notes="Monitoring begun")

        logger.info("✅ Monitoring started. Press Q to stop.")

        prev_gray = None  # For optical flow

        try:
            while self._running:
                # ================================================================
                # STEP 1: CAPTURE FRAME
                # ================================================================
                t_frame_start = time.time()
                ret, frame = self.camera.read_frame()

                if not ret or frame is None:
                    logger.warning("Frame read failed. Retrying...")
                    time.sleep(0.033)
                    continue

                self._frame_count += 1
                h, w = frame.shape[:2]

                # ================================================================
                # STEP 2: MOTION DETECTION (Temporal Frame Differencing)
                # ================================================================
                motion_result = self.motion_detector.detect(frame)
                motion_detected = motion_result["motion_detected"]
                current_gray = motion_result["current_gray"]

                # ================================================================
                # STEP 3: OPTICAL FLOW (Lucas-Kanade)
                # ================================================================
                flow_result = {"has_flow": False, "mean_magnitude": 0.0}
                if prev_gray is not None:
                    flow_result = self.optical_flow.calculate_optical_flow(
                        prev_gray, current_gray
                    )
                prev_gray = current_gray

                # ================================================================
                # STEP 4: DOUBLE BACKGROUND FILTERING (DBF)
                # ================================================================
                dbf_result = self.dbf.update_and_filter(current_gray)

                # ================================================================
                # STEP 5: MORPHOLOGICAL PROCESSING
                # ================================================================
                clean_mask = self.morphology.process(dbf_result["foreground_mask"])

                # If no motion, we can skip pose estimation to save CPU
                # (still run every 10th frame for status updates)
                if not motion_detected and self._frame_count % 10 != 0:
                    update_state(motion_detected=False)
                    # Still need to show skeleton — just use previous
                    self._show_and_handle_key()
                    continue

                # ================================================================
                # STEP 6: POSE ESTIMATION (17 Skeletal Keypoints)
                # ================================================================
                pose_result = self.pose_estimator.estimate(frame)

                # ================================================================
                # STEP 7: PRIVACY — DISCARD RAW RGB FRAME
                # The frame variable is explicitly set to None.
                # After this point, NO raw pixel data is used.
                # Only skeleton keypoints proceed through the pipeline.
                # ================================================================
                frame = None  # RAW FRAME DISCARDED ← PRIVACY ENFORCEMENT

                person_detected = pose_result["person_detected"]
                keypoints = pose_result.get("keypoints")

                # ================================================================
                # STEP 8: FEATURE EXTRACTION
                # ================================================================
                feature_vector = None
                aspect_ratio = 0.5

                if person_detected and keypoints:
                    feature_vector = self.feature_extractor.extract(keypoints)

                    if feature_vector is not None:
                        self.temporal_window.add(feature_vector)
                        aspect_ratio = float(feature_vector[0])

                    # Buffer skeletal coordinates for privacy animation replay
                    self.replay_buffer.add_frame(
                        keypoints=keypoints,
                        activity=activity if 'activity' in locals() else "Unknown",
                        confidence=confidence if 'confidence' in locals() else 0.0,
                        aspect_ratio=aspect_ratio,
                        kinetic_energy=float(feature_vector[3]) if feature_vector is not None else 0.0,
                        is_fall_state=(fall_state in ["POTENTIAL_FALL", "FALLEN_CONFIRMED", "ALERTING"]) if 'fall_state' in locals() else False,
                    )

                # ================================================================
                # STEP 9: ML CLASSIFICATION
                # ================================================================
                temporal_features = self.temporal_window.compute_temporal_features()
                classification = self.classifier.predict(temporal_features, aspect_ratio)
                activity = classification["activity"]
                confidence = classification["confidence"]

                # ================================================================
                # STEP 10: IRC CALCULATION
                # ================================================================
                movement_intensity = (
                    float(feature_vector[5]) if feature_vector is not None else 0.0
                )
                irc_result = self.irc_calculator.update(movement_intensity)

                # ================================================================
                # STEP 11: MULTI-SIGNAL FALL DETECTION
                # ================================================================
                fall_analysis = self.fall_detector.analyze(
                    activity=activity,
                    confidence=confidence,
                    features=feature_vector,
                    irc_result=irc_result,
                )

                # ================================================================
                # STEP 12: TEMPORAL VALIDATION
                # ================================================================
                validation_result = self.temporal_validator.update(
                    fall_signal=fall_analysis["fall_signal"],
                    activity=activity,
                    confidence=confidence,
                )

                fall_state = validation_result["state"]
                fall_message = validation_result["message"]

                # ================================================================
                # STEP 13: ALERT (if confirmed fall)
                # ================================================================
                if validation_result["should_alert"]:
                    logger.warning("🚨 Triggering emergency alert!")
                    event_id = f"FALL_{int(time.time())}"
                    replay_info = self.replay_buffer.generate_replay(event_id=event_id)
                    alert_sent = self.alert_manager.send_alert(
                        activity=activity,
                        confidence=confidence,
                        duration=validation_result["fall_duration"],
                        irc_value=irc_result.get("irc_abs", 0.0),
                        event_id=event_id,
                        replay_info=replay_info,
                    )

                    if alert_sent:
                        self.database.log_event(
                            event_type="FALL_DETECTED",
                            activity=activity,
                            confidence=confidence,
                            duration=validation_result["fall_duration"],
                            irc_value=irc_result.get("irc_abs", 0.0),
                            alert_sent=True,
                        )
                        update_state(
                            replay_html=replay_info.get("html_path"),
                            replay_gif=replay_info.get("gif_path"),
                        )

                # ================================================================
                # STEP 14: DATABASE LOGGING (every N frames)
                # ================================================================
                if self._frame_count - self._last_db_log_frame >= config.LOG_ACTIVITY_EVERY_N_FRAMES:
                    self.database.log_activity(
                        activity=activity,
                        confidence=confidence,
                        aspect_ratio=aspect_ratio,
                        kinetic_energy=(
                            float(feature_vector[3]) if feature_vector is not None else 0.0
                        ),
                    )
                    self._last_db_log_frame = self._frame_count

                # ================================================================
                # STEP 15: SKELETON VISUALIZATION (Privacy-Preserving)
                # NOT the raw camera image — only skeleton on dark background
                # ================================================================
                skeleton_img = self.pose_estimator.draw_skeleton_privacy(
                    frame_shape=(h, w),
                    keypoints=keypoints,
                    activity_label=f"{activity} [{fall_state}]",
                    confidence=confidence,
                )

                # Draw additional info on skeleton image
                fps_actual = self.camera.get_fps()
                self._draw_hud(
                    skeleton_img, activity, confidence, fall_state,
                    fall_message, fps_actual, irc_result.get("irc_abs", 0.0),
                    motion_detected, person_detected,
                    fall_analysis.get("vote_count", 0),
                )

                # ================================================================
                # STEP 16: UPDATE SHARED STATE (for dashboard)
                # ================================================================
                update_state(
                    activity=activity,
                    confidence=confidence,
                    fps=fps_actual,
                    fall_state=fall_state,
                    fall_message=fall_message,
                    motion_detected=motion_detected,
                    person_detected=person_detected,
                    timestamp=datetime.now().isoformat(),
                    irc_value=irc_result.get("irc_abs", 0.0),
                    vote_count=fall_analysis.get("vote_count", 0),
                    aspect_ratio=aspect_ratio,
                    head_accel=float(feature_vector[2]) if feature_vector is not None else 0.0,
                    kinetic_energy=float(feature_vector[3]) if feature_vector is not None else 0.0,
                )

                with _state_lock:
                    _monitor_state["skeleton_image"] = skeleton_img

                # ================================================================
                # DISPLAY + KEY HANDLING
                # ================================================================
                cv2.imshow(
                    "Elder Care Monitor [Skeleton View — Privacy Mode]",
                    skeleton_img
                )

                # Measure per-frame latency
                t_frame_end = time.time()
                latency_ms = (t_frame_end - t_frame_start) * 1000

                if self._frame_count % 30 == 0:
                    logger.info(
                        f"FPS: {fps_actual:.1f} | Latency: {latency_ms:.1f}ms | "
                        f"Activity: {activity} ({confidence:.1%}) | "
                        f"Fall state: {fall_state}"
                    )

                # Handle keypresses
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == ord("Q"):
                    logger.info("Q pressed — stopping monitoring.")
                    break
                elif key == ord("r") or key == ord("R"):
                    # Manual reset
                    self.temporal_validator.force_reset()
                    logger.info("Fall state manually reset.")

        except KeyboardInterrupt:
            logger.info("Interrupted by user.")
        finally:
            self._shutdown()

    def _draw_hud(
        self, img, activity, confidence, fall_state, fall_message,
        fps, irc_val, motion, person, votes
    ):
        """Draw heads-up display information on the skeleton image."""
        h, w = img.shape[:2]
        right_x = w - 280

        # Right panel info
        lines = [
            f"FPS: {fps:.1f}",
            f"Motion: {'YES' if motion else 'NO'}",
            f"Person: {'YES' if person else 'NO'}",
            f"Activity: {activity}",
            f"Confidence: {confidence*100:.1f}%",
            f"IRC: {irc_val:.4f}",
            f"Votes: {votes}/7",
            f"Fall: {fall_state}",
        ]

        # Background rect
        cv2.rectangle(img, (right_x - 5, 0), (w, len(lines) * 22 + 20), (0, 0, 0), -1)

        for i, line in enumerate(lines):
            color = (255, 200, 0) if "YES" in line or "FALL" in fall_state else (200, 200, 200)
            if "FALL_DETECT" in fall_state or "ALERT" in fall_state:
                color = (0, 0, 255)
            cv2.putText(img, line, (right_x, 20 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        # Fall state prominent display
        if fall_state not in ["NORMAL", "COOLDOWN"]:
            overlay = img.copy()
            cv2.rectangle(overlay, (0, h - 50), (w, h), (0, 0, 200), -1)
            cv2.addWeighted(overlay, 0.4, img, 0.6, 0, img)
            cv2.putText(img, f"⚠ {fall_message}", (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    def _show_and_handle_key(self):
        """Show the current skeleton image and handle key input."""
        with _state_lock:
            img = _monitor_state.get("skeleton_image")
        if img is not None:
            cv2.imshow("Elder Care Monitor [Skeleton View — Privacy Mode]", img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == ord("Q"):
            self._running = False

    def _run_demo(self) -> None:
        """Runs an examiner-ready simulation demo without needing a physical camera."""
        self._running = True
        self._start_time = time.time()
        update_state(running=True, privacy_status="RAW VIDEO STORAGE: DISABLED (SIMULATION)")
        self.database.log_system_status("STARTED", notes=f"Demo simulation: {self.demo_mode}")
        logger.info(f"▶ Running Examiner Demonstration Mode: '{self.demo_mode}'")

        # 17 normalized keypoints: [x, y, confidence]
        standing_kps = [
            [0.50, 0.12, 0.95], [0.48, 0.10, 0.95], [0.52, 0.10, 0.95],
            [0.46, 0.12, 0.95], [0.54, 0.12, 0.95], [0.42, 0.22, 0.95],
            [0.58, 0.22, 0.95], [0.38, 0.38, 0.95], [0.62, 0.38, 0.95],
            [0.36, 0.52, 0.95], [0.64, 0.52, 0.95], [0.44, 0.52, 0.95],
            [0.56, 0.52, 0.95], [0.44, 0.70, 0.95], [0.56, 0.70, 0.95],
            [0.44, 0.88, 0.95], [0.56, 0.88, 0.95]
        ]
        fallen_kps = [
            [0.18, 0.75, 0.95], [0.17, 0.74, 0.95], [0.17, 0.76, 0.95],
            [0.16, 0.73, 0.95], [0.16, 0.77, 0.95], [0.28, 0.72, 0.95],
            [0.28, 0.78, 0.95], [0.40, 0.70, 0.95], [0.40, 0.80, 0.95],
            [0.52, 0.68, 0.95], [0.52, 0.82, 0.95], [0.58, 0.73, 0.95],
            [0.58, 0.77, 0.95], [0.72, 0.72, 0.95], [0.72, 0.78, 0.95],
            [0.86, 0.72, 0.95], [0.86, 0.78, 0.95]
        ]

        h, w = config.FRAME_HEIGHT, config.FRAME_WIDTH
        sim_frames = 150
        fall_start = 30
        fall_end = 45

        for frame_idx in range(sim_frames):
            if not self._running:
                break
            self._frame_count += 1
            t_start = time.time()

            # Generate scenario keypoints
            if self.demo_mode == "fall":
                if frame_idx < fall_start:
                    # Standing naturally with slight motion
                    t = frame_idx * 0.1
                    kps = [[p[0] + 0.005 * math.sin(t), p[1] + 0.003 * math.cos(t), p[2]] for p in standing_kps]
                elif frame_idx <= fall_end:
                    # Rapid fall trajectory
                    alpha = (frame_idx - fall_start) / float(fall_end - fall_start)
                    alpha_accel = alpha ** 2
                    kps = [
                        [
                            (1.0 - alpha_accel) * p_s[0] + alpha_accel * p_f[0],
                            (1.0 - alpha_accel) * p_s[1] + alpha_accel * p_f[1],
                            0.95
                        ]
                        for p_s, p_f in zip(standing_kps, fallen_kps)
                    ]
                else:
                    # Post-fall immobility (inert on floor)
                    kps = [list(p) for p in fallen_kps]
            else:
                # Normal standing/sitting
                t = frame_idx * 0.08
                kps = [[p[0] + 0.008 * math.sin(t), p[1], p[2]] for p in standing_kps]

            # Process through pipeline
            feature_vector = self.feature_extractor.extract(kps)
            aspect_ratio = float(feature_vector[0]) if feature_vector is not None else 0.5
            if feature_vector is not None:
                self.temporal_window.add(feature_vector)

            temporal_features = self.temporal_window.compute_temporal_features()
            classification = self.classifier.predict(temporal_features, aspect_ratio)
            activity = classification["activity"]
            confidence = classification["confidence"]

            movement_intensity = float(feature_vector[5]) if feature_vector is not None else 0.0
            irc_result = self.irc_calculator.update(movement_intensity)

            fall_analysis = self.fall_detector.analyze(
                activity=activity,
                confidence=confidence,
                features=feature_vector,
                irc_result=irc_result,
            )

            validation_result = self.temporal_validator.update(
                fall_signal=fall_analysis["fall_signal"],
                activity=activity,
                confidence=confidence,
            )

            fall_state = validation_result["state"]
            fall_message = validation_result["message"]

            # Add to replay buffer
            self.replay_buffer.add_frame(
                keypoints=kps,
                activity=activity,
                confidence=confidence,
                aspect_ratio=aspect_ratio,
                kinetic_energy=float(feature_vector[3]) if feature_vector is not None else 0.0,
                is_fall_state=(fall_state in ["POTENTIAL_FALL", "FALLEN_CONFIRMED", "ALERTING"]),
            )

            if validation_result["should_alert"]:
                logger.warning("🚨 DEMO: Confirmed Fall! Triggering alert and replay generation...")
                event_id = f"DEMO_FALL_{int(time.time())}"
                replay_info = self.replay_buffer.generate_replay(event_id=event_id)
                self.alert_manager.send_alert(
                    activity=activity,
                    confidence=confidence,
                    duration=validation_result["fall_duration"],
                    irc_value=irc_result.get("irc_abs", 0.0),
                    event_id=event_id,
                    replay_info=replay_info,
                )
                self.database.log_event(
                    event_type="DEMO_FALL_DETECTED",
                    activity=activity,
                    confidence=confidence,
                    duration=validation_result["fall_duration"],
                    irc_value=irc_result.get("irc_abs", 0.0),
                    alert_sent=True,
                )
                update_state(
                    replay_html=replay_info.get("html_path"),
                    replay_gif=replay_info.get("gif_path"),
                )

            # Draw preview
            skeleton_img = self.pose_estimator.draw_skeleton_privacy(
                frame_shape=(h, w),
                keypoints=kps,
                activity_label=f"DEMO: {activity} [{fall_state}]",
                confidence=confidence,
            )
            self._draw_hud(
                skeleton_img, activity, confidence, fall_state,
                fall_message, 30.0, irc_result.get("irc_abs", 0.0),
                True, True, fall_analysis.get("vote_count", 0)
            )

            with _state_lock:
                _monitor_state["skeleton_image"] = skeleton_img

            update_state(
                activity=activity,
                confidence=confidence,
                fps=30.0,
                fall_state=fall_state,
                fall_message=fall_message,
                motion_detected=True,
                person_detected=True,
                timestamp=datetime.now().isoformat(),
                irc_value=irc_result.get("irc_abs", 0.0),
                vote_count=fall_analysis.get("vote_count", 0),
                aspect_ratio=aspect_ratio,
                head_accel=float(feature_vector[2]) if feature_vector is not None else 0.0,
                kinetic_energy=float(feature_vector[3]) if feature_vector is not None else 0.0,
            )

            self._show_and_handle_key()
            elapsed = time.time() - t_start
            time.sleep(max(0.01, 0.04 - elapsed))

        logger.info("Demo sequence completed.")
        self._shutdown()

    def _shutdown(self) -> None:
        """Clean up all resources."""
        logger.info("Shutting down...")
        self._running = False
        self.camera.stop()
        self.pose_estimator.close()
        self.database.log_system_status("STOPPED")
        self.database.close()
        cv2.destroyAllWindows()
        update_state(running=False)
        logger.info("System stopped cleanly.")


def main():
    parser = argparse.ArgumentParser(description="CareVision: Privacy-Preserving Elder Care Monitoring")
    parser.add_argument("--source", default=config.CAMERA_INDEX, help="Camera index (0, 1) or video path")
    parser.add_argument("--demo", choices=["none", "fall", "sitting", "normal"], default="none",
                        help="Run an automated examiner simulation demonstration")
    args = parser.parse_args()

    monitor = ElderCareMonitor(camera_source=args.source, demo_mode=args.demo)
    monitor.run()


if __name__ == "__main__":
    main()
