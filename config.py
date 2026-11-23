"""
config.py — Central Configuration for Elder Care Monitoring System
=================================================================
All tunable parameters are here.
Change values here instead of touching the code files.

IMPORTANT: This is a prototype system.
It does NOT replace professional medical monitoring.
"""

import os

# =============================================================================
# CAMERA SETTINGS
# =============================================================================
CAMERA_INDEX = 0                # 0 = default webcam, 1 = external camera
FRAME_WIDTH = 640               # Frame width in pixels
FRAME_HEIGHT = 480              # Frame height in pixels
TARGET_FPS = 30                 # Target frames per second

# =============================================================================
# MOTION DETECTION SETTINGS
# =============================================================================
MOTION_THRESHOLD = 25           # Pixel intensity threshold for motion (0-255)
MOTION_RATIO_MIN = 0.001        # Minimum fraction of frame that must move
MOTION_RATIO_MAX = 0.95         # Maximum (too much = noise/lighting change)
MOTION_BLUR_KERNEL = (5, 5)     # Gaussian blur kernel to reduce noise
USE_ADAPTIVE_THRESHOLD = False  # True = smarter but slower threshold

# =============================================================================
# OPTICAL FLOW SETTINGS (Lucas-Kanade)
# =============================================================================
LK_MAX_CORNERS = 100            # Max feature points to track
LK_QUALITY_LEVEL = 0.3         # Quality of corner detection (0-1)
LK_MIN_DISTANCE = 7             # Minimum distance between corners (pixels)
LK_BLOCK_SIZE = 7               # Block size for corner detection
LK_WIN_SIZE = (15, 15)          # Lucas-Kanade window size
LK_MAX_LEVEL = 2                # Pyramid levels for LK flow
FLOW_MAGNITUDE_THRESHOLD = 1.0  # Min flow magnitude to count as movement

# =============================================================================
# DOUBLE BACKGROUND FILTERING (DBF) SETTINGS
# =============================================================================
# Short-term window: catches fast environmental changes (fans, curtains)
DBF_SHORT_WINDOW = 3            # Frames in short-term accumulation
# Long-term window: catches slow environmental changes (lighting shift)
DBF_LONG_WINDOW = 5             # Frames in long-term accumulation
DBF_DIFF_THRESHOLD = 30         # Threshold for DBF foreground mask

# =============================================================================
# MORPHOLOGICAL OPERATION SETTINGS
# =============================================================================
MORPH_KERNEL_SIZE = (5, 5)      # Kernel for morphological operations
MORPH_DILATION_ITER = 2         # Number of dilation iterations

# =============================================================================
# POSE ESTIMATION SETTINGS
# =============================================================================
POSE_MIN_DETECTION_CONFIDENCE = 0.5   # Min confidence to detect a person
POSE_MIN_TRACKING_CONFIDENCE = 0.5    # Min confidence to keep tracking
NUM_KEYPOINTS = 17              # 17-point skeletal representation

# MediaPipe keypoint indices (maps to our 17-point system)
# 0=Nose, 1=L-Eye, 2=R-Eye, 3=L-Ear, 4=R-Ear,
# 5=L-Shoulder, 6=R-Shoulder, 7=L-Elbow, 8=R-Elbow,
# 9=L-Wrist, 10=R-Wrist, 11=L-Hip, 12=R-Hip,
# 13=L-Knee, 14=R-Knee, 15=L-Ankle, 16=R-Ankle

KEYPOINT_NAMES = [
    "Nose", "Left_Eye", "Right_Eye", "Left_Ear", "Right_Ear",
    "Left_Shoulder", "Right_Shoulder", "Left_Elbow", "Right_Elbow",
    "Left_Wrist", "Right_Wrist", "Left_Hip", "Right_Hip",
    "Left_Knee", "Right_Knee", "Left_Ankle", "Right_Ankle"
]

# Skeleton connections for visualization (joint A connects to joint B)
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),          # Head connections
    (5, 6),                                    # Shoulder line
    (5, 7), (7, 9),                            # Left arm
    (6, 8), (8, 10),                           # Right arm
    (5, 11), (6, 12),                          # Torso
    (11, 12),                                  # Hip line
    (11, 13), (13, 15),                        # Left leg
    (12, 14), (14, 16),                        # Right leg
]

# =============================================================================
# FEATURE EXTRACTION SETTINGS
# =============================================================================
TEMPORAL_WINDOW_SIZE = 15       # Number of frames for temporal features
MIN_VISIBILITY = 0.3            # Minimum keypoint visibility to use it

# Important keypoint indices for feature extraction
IDX_NOSE = 0
IDX_LEFT_SHOULDER = 5
IDX_RIGHT_SHOULDER = 6
IDX_LEFT_HIP = 11
IDX_RIGHT_HIP = 12
IDX_LEFT_KNEE = 13
IDX_RIGHT_KNEE = 14
IDX_LEFT_ANKLE = 15
IDX_RIGHT_ANKLE = 16

# =============================================================================
# MACHINE LEARNING SETTINGS
# =============================================================================
# Activity classes (must match training labels)
ACTIVITY_CLASSES = ["Standing", "Walking", "Sitting", "Reclining", "Fallen"]
NUM_CLASSES = len(ACTIVITY_CLASSES)

# Random Forest hyperparameters
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 10
RF_RANDOM_STATE = 42
RF_CLASS_WEIGHT = "balanced"    # Important: handles imbalanced dataset

# Model file path
MODEL_PATH = "ml/model.pkl"
SCALER_PATH = "ml/scaler.pkl"
LABEL_ENCODER_PATH = "ml/label_encoder.pkl"

# Train/test split ratio
TEST_SIZE = 0.2

# =============================================================================
# FALL DETECTION SETTINGS
# =============================================================================
# Aspect ratio: width/height. Falls make this go UP (body becomes horizontal)
FALL_ASPECT_RATIO_THRESHOLD = 0.8   # Above this = potentially horizontal
NORMAL_ASPECT_RATIO = 0.4            # Typical standing aspect ratio

# Head movement thresholds
HEAD_VELOCITY_THRESHOLD = 0.05       # Normalized units per frame
HEAD_ACCELERATION_THRESHOLD = 0.03   # Normalized acceleration per frame

# Post-fall inactivity (seconds): how long before emergency confirmed
FALL_CONFIRMATION_SECONDS = 5.0

# Recovery window: if person gets up within this time, cancel alert
RECOVERY_WINDOW_SECONDS = 10.0

# Minimum consecutive frames to call it a "fall" before temporal validation
FALL_MIN_FRAMES = 5

# =============================================================================
# INSTANTANEOUS RATE OF CHANGE (IRC) SETTINGS
# =============================================================================
# IRC measures sudden change in movement intensity
# Proposed threshold from synopsis: 0.25 (prototype value — needs tuning)
IRC_THRESHOLD = 0.25
IRC_WINDOW = 5                       # Number of frames to compare

# =============================================================================
# ALERT SETTINGS
# =============================================================================
ALERT_COOLDOWN_SECONDS = 30          # Minimum time between alerts
ALERT_SOUND_FILE = "alerts/alert.wav"
ENABLE_SOUND_ALERT = True
ENABLE_DESKTOP_NOTIFICATION = True

# Optional: Webhook for remote notification (leave empty to disable)
WEBHOOK_URL = ""                     # e.g., "https://your-server.com/alert"
FIREBASE_ENABLED = False

# =============================================================================
# DATABASE SETTINGS
# =============================================================================
DATABASE_PATH = "database/elder_care.db"
LOG_ACTIVITY_EVERY_N_FRAMES = 30    # Log activity to DB every N frames
MAX_EVENTS_IN_DASHBOARD = 50        # Recent events to show on dashboard

# =============================================================================
# PRIVACY SETTINGS (CORE REQUIREMENT)
# =============================================================================
SAVE_RAW_FRAMES = False              # MUST remain False for privacy
UPLOAD_RAW_FRAMES = False            # MUST remain False for privacy
STORE_FACE_DATA = False              # MUST remain False for privacy
PRIVACY_MODE = True                  # Privacy indicator for dashboard

# =============================================================================
# DATASET COLLECTION SETTINGS
# =============================================================================
DATASET_DIR = "data/features"
DATASET_FILE = "data/features/features.csv"
COLLECT_FRAMES_PER_LABEL = 200      # Frames to collect per activity label

# =============================================================================
# PERFORMANCE SETTINGS
# =============================================================================
ENABLE_PERFORMANCE_LOGGING = True
PERFORMANCE_LOG_FILE = "logs/performance.log"

# =============================================================================
# LOGGING
# =============================================================================
LOG_DIR = "logs"
LOG_FILE = "logs/system.log"
LOG_LEVEL = "INFO"                   # DEBUG, INFO, WARNING, ERROR

# =============================================================================
# VISUALIZATION (Optional Privacy-Preserving Skeleton Display)
# =============================================================================
SHOW_SKELETON_PREVIEW = True         # Show skeleton (NOT raw video)
SHOW_RAW_VIDEO_IN_DASHBOARD = False  # MUST remain False for privacy
SKELETON_DOT_RADIUS = 5
SKELETON_LINE_THICKNESS = 2
SKELETON_COLOR_JOINT = (0, 255, 0)   # Green joints
SKELETON_COLOR_BONE = (0, 200, 255)  # Cyan bones
SKELETON_BG_COLOR = (30, 30, 30)     # Dark background

# Dashboard refresh rate (seconds)
DASHBOARD_REFRESH_SECONDS = 2
