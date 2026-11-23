"""
collect_dataset.py — Dataset Collection Tool
=============================================
PURPOSE:
    Record labeled training data for the ML model.

HOW TO USE:
    1. Run: python collect_dataset.py
    2. Select a label when prompted (e.g., "Standing")
    3. Perform the activity in front of the camera
    4. Press SPACE to start/stop recording
    5. Press Q to quit

DATA COLLECTED:
    Only feature vectors are saved — NOT raw video.
    Output: data/features/features.csv

PRIVACY:
    Raw frames are processed in memory and immediately discarded.
    Only numerical features are written to disk.

LABELS TO COLLECT:
    1. Standing
    2. Walking
    3. Sitting
    4. Reclining
    5. Fallen

COLLECTION TIPS:
    - Collect at least 100 frames per class
    - Vary your position in the frame
    - Move naturally (don't be stiff)
    - For "Fallen": lie down on the floor slowly and carefully
    - Collect multiple sessions for better generalization
"""

import cv2
import os
import sys
import csv
import logging
import time
from datetime import datetime
from typing import List, Optional

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from camera import CameraCapture
from motion import MotionDetector, OpticalFlowAnalyzer, DoubleBackgroundFilter, MorphologyProcessor
from pose import PoseEstimator
from features import FeatureExtractor, TemporalWindow

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

LABELS = config.ACTIVITY_CLASSES


def ensure_dataset_dir():
    """Create output directory if it doesn't exist."""
    os.makedirs(config.DATASET_DIR, exist_ok=True)


def get_csv_writer(csv_path: str, feature_names: List[str]):
    """
    Open CSV file for appending and return writer.
    If file doesn't exist, write header.
    """
    file_exists = os.path.exists(csv_path)
    f = open(csv_path, "a", newline="")
    writer = csv.writer(f)

    if not file_exists:
        # Write header
        headers = ["timestamp"] + feature_names + ["label"]
        writer.writerow(headers)
        logger.info(f"Created new dataset file: {csv_path}")
    else:
        logger.info(f"Appending to existing dataset: {csv_path}")

    return f, writer


def show_menu() -> str:
    """Show label selection menu and return chosen label."""
    print("\n" + "="*50)
    print("DATASET COLLECTION — Select Activity Label")
    print("="*50)
    for i, label in enumerate(LABELS, 1):
        print(f"  {i}. {label}")
    print("  Q. Quit")
    print("="*50)

    while True:
        choice = input("Enter number (1-5) or Q: ").strip().upper()
        if choice == "Q":
            return None
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(LABELS):
                return LABELS[idx]
        except ValueError:
            pass
        print("Invalid choice. Try again.")


def main():
    ensure_dataset_dir()

    # Initialize all modules
    camera = CameraCapture()
    motion_detector = MotionDetector()
    optical_flow = OpticalFlowAnalyzer()
    dbf = DoubleBackgroundFilter()
    morphology = MorphologyProcessor()
    pose_estimator = PoseEstimator()
    feature_extractor = FeatureExtractor()
    temporal_window = TemporalWindow()

    if not camera.start():
        logger.error("Failed to open camera. Exiting.")
        return

    feature_names = temporal_window.get_temporal_feature_names()
    csv_file, csv_writer = get_csv_writer(config.DATASET_FILE, feature_names)

    logger.info(
        "\n=== DATASET COLLECTION STARTED ===\n"
        "Controls:\n"
        "  SPACE  = Start/Stop recording for current label\n"
        "  N      = Next label (stop recording + choose new)\n"
        "  Q      = Quit\n"
    )

    current_label = show_menu()
    if current_label is None:
        camera.stop()
        csv_file.close()
        return

    recording = False
    frame_count_saved = 0
    last_activity_time = time.time()

    try:
        while True:
            ret, frame = camera.read_frame()
            if not ret or frame is None:
                continue

            h, w = frame.shape[:2]

            # --- Run pipeline ---
            motion_result = motion_detector.detect(frame)
            gray = motion_result["current_gray"]

            dbf_result = dbf.update_and_filter(gray)
            clean_mask = morphology.process(dbf_result["foreground_mask"])

            pose_result = pose_estimator.estimate(frame)

            # PRIVACY: Discard raw frame reference
            # (frame goes out of scope at end of loop iteration)

            feature_vector = None
            aspect_ratio = 0.5
            if pose_result["person_detected"] and pose_result["keypoints"]:
                kps = pose_result["keypoints"]
                feature_vector = feature_extractor.extract(kps)
                if feature_vector is not None:
                    temporal_window.add(feature_vector)
                    aspect_ratio = float(feature_vector[0])

            # --- Create visualization (skeleton only, not raw video) ---
            skeleton_img = pose_estimator.draw_skeleton_privacy(
                (h, w),
                pose_result.get("keypoints"),
                activity_label=f"Collecting: {current_label}",
                confidence=1.0,
            )

            # --- Draw UI overlay on skeleton image ---
            status_text = "🔴 RECORDING" if recording else "⏸ Press SPACE to record"
            status_color = (0, 0, 255) if recording else (100, 100, 100)
            cv2.putText(skeleton_img, status_text, (10, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            cv2.putText(skeleton_img, f"Label: {current_label}", (10, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
            cv2.putText(skeleton_img, f"Saved: {frame_count_saved} frames", (10, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(skeleton_img, f"FPS: {camera.get_fps():.1f}", (10, 190),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

            cv2.imshow("Data Collection (Skeleton View — Privacy Mode)", skeleton_img)

            # --- Save features if recording ---
            if recording and temporal_window.is_ready():
                temp_features = temporal_window.compute_temporal_features()
                if temp_features is not None:
                    timestamp = datetime.now().isoformat()
                    row = [timestamp] + temp_features.tolist() + [current_label]
                    csv_writer.writerow(row)
                    csv_file.flush()
                    frame_count_saved += 1

                    if frame_count_saved % 50 == 0:
                        logger.info(
                            f"Saved {frame_count_saved} temporal feature frames for '{current_label}'"
                        )

            # --- Handle key presses ---
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == ord("Q"):
                logger.info("Quitting dataset collection.")
                break

            elif key == ord(" "):  # SPACE = toggle recording
                recording = not recording
                if recording:
                    logger.info(f"▶ Recording started for: '{current_label}'")
                else:
                    logger.info(
                        f"⏸ Recording stopped. "
                        f"Total saved: {frame_count_saved} frames"
                    )

            elif key == ord("n") or key == ord("N"):
                recording = False
                logger.info(
                    f"Finished '{current_label}': {frame_count_saved} frames saved"
                )
                new_label = show_menu()
                if new_label is None:
                    break
                current_label = new_label
                feature_extractor.reset()
                temporal_window.reset()
                logger.info(f"Now collecting: '{current_label}'")

    finally:
        camera.stop()
        csv_file.close()
        pose_estimator.close()
        cv2.destroyAllWindows()
        logger.info(f"\n✅ Dataset collection complete.")
        logger.info(f"   Total frames saved: {frame_count_saved}")
        logger.info(f"   Dataset file: {config.DATASET_FILE}")
        logger.info(f"\n   Next step: python -m ml.train_model")


if __name__ == "__main__":
    main()
