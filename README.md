# 🏠 Privacy-Preserving Elder Care Monitoring System

**Edge-Based Skeletal Pose Estimation and Anomaly Detection**

> **B.Tech CSE Major Project** | Computer Vision + Edge AI + Machine Learning

---

## ⚠️ Disclaimer

> This system is an **academic prototype** intended for research and learning purposes.  
> It does **NOT** replace professional medical monitoring or emergency services.  
> All threshold values are prototype values that have not been clinically validated.

---

## 📌 Problem Statement

Over **700 million elderly people** worldwide live alone or with limited assistance. The key challenges:

- **Safety**: Falls are a leading cause of injury in the elderly. Many go undetected for hours.
- **Privacy**: Traditional camera-based systems continuously record and store video, which is intrusive and creates data security risks.
- **Accessibility**: Remote caregivers need status updates without requiring expensive infrastructure.

---

## 💡 Motivation

Existing elder care monitoring solutions either:
1. Store continuous video (privacy violation)
2. Require cloud processing (latency + cost + connectivity)
3. Use expensive specialized hardware

This project proposes a **privacy-preserving, edge-based solution** that runs entirely on a local computer (or Raspberry Pi), extracts only skeletal keypoints, immediately discards raw video, and uses machine learning to detect falls.

---

## 🎯 Objectives

1. Capture video locally using a standard webcam
2. Detect human movement using temporal frame differencing
3. Suppress environmental noise using Double Background Filtering (DBF)
4. Extract 17-point skeletal keypoints using MediaPipe Pose
5. **Discard raw RGB frames immediately after keypoint extraction**
6. Compute spatial-temporal features from skeletal data
7. Classify activities using Random Forest (Standing / Walking / Sitting / Reclining / Fallen)
8. Detect falls using multi-signal analysis (ML + geometry + velocity + IRC)
9. Use temporal validation to reduce false alarms
10. Send emergency alerts when a fall is confirmed
11. Log only numerical/metadata to SQLite (no raw video)
12. Display results on a Streamlit dashboard

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              PRIVACY-PRESERVING ELDER CARE PIPELINE             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   WEBCAM → Frame Capture → Motion Detection (Frame Diff)        │
│              ↓                                                  │
│         Optical Flow (Lucas-Kanade) → Vx, Vy, Magnitude         │
│              ↓                                                  │
│     Double Background Filtering (DBF) → Foreground Mask         │
│              ↓                                                  │
│    Morphological Processing (Opening→Closing→Dilation)          │
│              ↓                                                  │
│     MediaPipe Pose Estimation → 17 Skeletal Keypoints           │
│              ↓                                                  │
│   ██ RAW RGB FRAME DISCARDED HERE ██ (Privacy Enforcement)      │
│              ↓                                                  │
│     Feature Extraction [9 features per frame]                   │
│              ↓                                                  │
│     Temporal Window [15 frames] → 36 Statistical Features       │
│              ↓                                                  │
│     Random Forest Classifier → Activity + Confidence            │
│              ↓                                                  │
│     IRC Calculator → Instantaneous Rate of Change               │
│              ↓                                                  │
│     Multi-Signal Fall Detector [7 votes]                        │
│              ↓                                                  │
│     Temporal Validator (State Machine)                          │
│              ↓                                                  │
│     Emergency Alert + SQLite Logging + Dashboard                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Technology Stack

| Component | Technology | Reason |
|---|---|---|
| Language | Python 3.10+ | Ecosystem support for CV + ML |
| Camera | OpenCV VideoCapture | Reliable, cross-platform |
| Motion Detection | OpenCV (frame differencing) | Fast, simple, effective |
| Optical Flow | Lucas-Kanade (OpenCV) | Lightweight, CPU-friendly |
| Background Filter | Custom DBF (temporal accumulation) | Project-specific design |
| Pose Estimation | MediaPipe Pose | CPU-efficient, open-source, privacy-safe |
| ML Classifier | scikit-learn Random Forest | Works on small datasets, no GPU |
| Database | SQLite | Lightweight, local, no server needed |
| Dashboard | Streamlit | Rapid Python web UI |
| Charts | Plotly | Interactive visualizations |
| Alerts | plyer (desktop) + optional webhook | Cross-platform notifications |

---

## 📁 Folder Structure

```
elder-care-monitoring/
│
├── app.py                    ← Main monitoring application (RUN THIS)
├── collect_dataset.py        ← Dataset collection tool
├── config.py                 ← All configuration parameters
├── requirements.txt          ← Python dependencies
├── .gitignore
│
├── camera/
│   ├── __init__.py
│   └── camera.py             ← Webcam capture with privacy design
│
├── motion/
│   ├── __init__.py
│   ├── motion_detection.py   ← Temporal frame differencing
│   ├── optical_flow.py       ← Lucas-Kanade optical flow
│   ├── background_filter.py  ← Double Background Filtering (DBF)
│   └── morphology.py         ← Opening, closing, dilation
│
├── pose/
│   ├── __init__.py
│   └── pose_estimation.py    ← MediaPipe 17-point skeleton + privacy visualization
│
├── features/
│   ├── __init__.py
│   ├── feature_extraction.py ← 9 spatial-temporal features
│   └── temporal_features.py  ← Sliding window (36 statistical features)
│
├── ml/
│   ├── __init__.py
│   ├── classifier.py         ← Activity classifier (model + fallback)
│   ├── train_model.py        ← Training script
│   └── evaluate_model.py     ← Evaluation with real metrics
│
├── fall_detection/
│   ├── __init__.py
│   ├── fall_detector.py      ← Multi-signal 7-vote fall analysis
│   ├── temporal_validator.py ← State machine: NORMAL→POTENTIAL→CONFIRMED→ALERT
│   └── irc.py                ← Instantaneous Rate of Change
│
├── alerts/
│   ├── __init__.py
│   └── alert_manager.py      ← Desktop + console + optional webhook alerts
│
├── database/
│   ├── __init__.py
│   └── database.py           ← SQLite (events, activity_logs, system_status)
│
├── dashboard/
│   ├── __init__.py
│   └── dashboard.py          ← Streamlit monitoring dashboard
│
├── tests/
│   ├── test_motion.py        ← 20+ motion module tests
│   ├── test_features.py      ← Feature extraction tests
│   ├── test_fall_detection.py← Fall detection + state machine tests
│   ├── test_database.py      ← Database + privacy verification tests
│   └── run_all_tests.py      ← Run everything at once
│
├── data/
│   └── features/
│       └── features.csv      ← Generated by collect_dataset.py
│
└── logs/
    └── system.log            ← Auto-generated at runtime
```

---

## ⚙️ Installation

### Step 1: Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/elder-care-monitoring.git
cd elder-care-monitoring
```

### Step 2: Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows:
venv\Scripts\activate

# Linux/Mac:
source venv/bin/activate
```

### Step 3: Install all dependencies

```bash
pip install -r requirements.txt
```

> **Note:** MediaPipe installation may take a few minutes. This is normal.

### Step 4: Verify camera access

```python
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera OK' if cap.isOpened() else 'Camera FAILED'); cap.release()"
```

---

## 📊 Dataset Collection (Phase 9)

Before training the model, you must collect labeled training data.

```bash
python collect_dataset.py
```

**Instructions:**
1. Select an activity label (1=Standing, 2=Walking, 3=Sitting, 4=Reclining, 5=Fallen)
2. Press **SPACE** to start recording
3. Perform the activity naturally in front of the camera
4. Press **SPACE** again to stop
5. Press **N** to switch to a new label
6. Collect at least **150–200 frames per class** for better accuracy
7. Press **Q** to quit

> **Privacy:** Only skeletal features are saved — no raw video is stored.

The output file is: `data/features/features.csv`

---

## 🤖 Model Training (Phase 10)

After collecting data:

```bash
python -m ml.train_model
```

This will:
- Load `data/features/features.csv`
- Split 80% train / 20% test
- Train a Random Forest classifier
- Print real accuracy and classification report
- Save: `ml/model.pkl`, `ml/scaler.pkl`, `ml/label_encoder.pkl`

---

## 📈 Model Evaluation (Phase 11)

```bash
python -m ml.evaluate_model
```

Generates:
- Classification report with actual Precision, Recall, F1
- Confusion matrix → saved to `data/confusion_matrix.png`
- Feature importance chart → saved to `data/feature_importance.png`
- 5-fold cross-validation results

> All metrics are measured on real data — never invented.

---

## 🚀 Running Live Monitoring (Phase 12+)

```bash
python app.py
```

**What you'll see:**
- A dark window showing only the skeleton visualization (not raw video)
- Activity label and confidence in the top-right corner
- Fall state indicator
- FPS counter
- Press **Q** to stop, **R** to reset fall state

---

## 📊 Dashboard (Phase 17)

In a **separate terminal**, while `app.py` is running:

```bash
streamlit run dashboard/dashboard.py
```

Open browser at: **http://localhost:8501**

Dashboard shows:
- 🟢 Monitoring status and FPS
- Current activity + confidence
- Fall detection state
- 🚨 Emergency alert panel (when fall detected)
- Activity timeline chart
- Recent events table
- Privacy status indicators

---

## 🚨 Alert Mechanism

When a fall is confirmed after temporal validation:

| Channel | How it works |
|---|---|
| **Dashboard Alert** | Red panel appears on Streamlit dashboard |
| **Desktop Notification** | Popup via `plyer` library |
| **Console Warning** | Prominent log message with all details |
| **Optional Webhook** | HTTP POST to URL set in `config.py` |

Alert data sent:
```json
{
  "event_id": "EVT_1234567890",
  "timestamp": "2024-10-05 22:30:00",
  "activity": "Fallen",
  "confidence": 0.87,
  "duration": 6.5,
  "irc_value": 0.31
}
```
> **No camera images or video are included in alerts.**

---

## 🔒 Privacy Design

This system implements privacy-by-design at every layer:

| Layer | Privacy Mechanism |
|---|---|
| **Processing** | All computation runs locally on the edge device |
| **Frame Handling** | Raw RGB frames are explicitly set to `None` after keypoint extraction |
| **Database** | Only numerical metadata stored; no BLOB columns |
| **Alerts** | Send only text metadata; no images |
| **Visualization** | Dashboard shows skeleton on dark background; NOT raw video |
| **Storage** | `SAVE_RAW_FRAMES = False` enforced in `config.py` |

**Privacy indicators on dashboard:**
```
✅ RAW VIDEO STORAGE: DISABLED
✅ FACE RECOGNITION: DISABLED  
✅ CLOUD VIDEO UPLOAD: DISABLED
```

---

## 🧪 Testing

Run all tests:
```bash
python tests/run_all_tests.py
```

Or with pytest (more verbose):
```bash
pip install pytest
python -m pytest tests/ -v
```

Test coverage:
- `test_motion.py` — 20+ tests for motion detection, DBF, morphology, optical flow
- `test_features.py` — Aspect ratio, velocity, KE proxy, temporal window
- `test_fall_detection.py` — IRC, fall state machine, false positive checks
- `test_database.py` — Storage operations, privacy column verification

---

## ⚙️ Configuration

All parameters are in `config.py`. Key settings:

```python
# Camera
CAMERA_INDEX = 0             # Change to 1 for external camera

# Motion Detection  
MOTION_THRESHOLD = 25        # Lower = more sensitive

# Double Background Filtering
DBF_SHORT_WINDOW = 3         # Frames in short-term window
DBF_LONG_WINDOW = 5          # Frames in long-term window

# Fall Detection
FALL_ASPECT_RATIO_THRESHOLD = 0.8  # Prototype value
IRC_THRESHOLD = 0.25               # Prototype value — needs tuning

# Temporal Validation
FALL_CONFIRMATION_SECONDS = 5.0    # Wait before sending alert
RECOVERY_WINDOW_SECONDS = 10.0     # Recovery time window

# Privacy (MUST NOT CHANGE)
SAVE_RAW_FRAMES = False
UPLOAD_RAW_FRAMES = False
```

---

## 📐 Key Algorithms

### Double Background Filtering (DBF)
```
short_bg = mean(last 3 frames)
long_bg  = mean(last 5 frames)
short_mask = |current - short_bg| > threshold
long_mask  = |current - long_bg|  > threshold
foreground = short_mask AND long_mask
```
Purpose: Remove fans, curtains, lighting changes while preserving human movement.

### 17-Point Skeletal Representation
```
Head:    Nose, L-Eye, R-Eye, L-Ear, R-Ear
Upper:   L-Shoulder, R-Shoulder, L-Elbow, R-Elbow, L-Wrist, R-Wrist
Lower:   L-Hip, R-Hip, L-Knee, R-Knee, L-Ankle, R-Ankle
```

### Feature Vector (9 features)
```
[0] aspect_ratio              — body width/height
[1] head_vertical_velocity    — head y-displacement per frame
[2] head_vertical_accel       — rate of change of head velocity
[3] kinetic_energy_proxy      — RMS joint velocity (normalized proxy)
[4] body_vertical_velocity    — hip center y-displacement per frame
[5] movement_intensity        — sum of all joint velocities
[6] shoulder_velocity         — average shoulder joint velocity
[7] hip_velocity              — average hip joint velocity
[8] knee_velocity             — average knee joint velocity
```

### IRC (Instantaneous Rate of Change)
```
IRC = (current_intensity - past_intensity) / window_frames
Threshold: IRC > 0.25 → sudden movement flagged
[PROTOTYPE VALUE — not scientifically validated]
```

### Temporal Validation State Machine
```
NORMAL → POTENTIAL_FALL (≥5 consecutive fall frames)
       → FALLEN_CONFIRMED (fall persists >5 seconds)
       → ALERTING (alert sent)
       → COOLDOWN (30 seconds)
       → NORMAL (cycle repeats)
Recovery: Any non-fall frame during POTENTIAL_FALL resets to NORMAL.
```

---

## ⚡ Performance

Expected on a modern laptop:

| Metric | Expected Value |
|---|---|
| Processing FPS | 15–25 FPS |
| Latency per frame | 40–70 ms |
| Memory usage | ~300–500 MB (MediaPipe) |
| Storage per day | ~1–2 MB (SQLite only) |

> **Note:** These are measured estimates — actual values depend on hardware.

---

## 🚧 Limitations

1. **Small dataset** — prototype results may not generalize to all body types, clothing, lighting conditions
2. **Threshold sensitivity** — IRC and fall detection thresholds are prototype values needing clinical validation
3. **Single person** — system is designed for monitoring one person at a time
4. **Lighting dependency** — very dark or very bright environments may affect pose detection
5. **Side/partial occlusion** — MediaPipe may lose tracking if camera angle is bad
6. **False positives** — rapid sitting, bending, or lying down may temporarily trigger potential-fall state

---

## 🔮 Future Scope

1. **Dataset expansion** — Collect from real elderly participants with ethics approval
2. **Clinical validation** — Validate thresholds against fall reference dataset
3. **YOLO-Pose** — Replace MediaPipe for better multi-person handling
4. **Raspberry Pi deployment** — Test on actual edge hardware
5. **SVM comparison** — Benchmark SVM vs Random Forest on real data
6. **LSTM/GRU** — Deep temporal models for better sequence learning
7. **Multi-camera** — Combine multiple viewpoints to reduce blind spots
8. **Firebase integration** — Real-time caregiver mobile notifications
9. **Federated learning** — Train model across devices without sharing data
10. **Wearable integration** — Combine camera with IMU sensor data

---

## 🎓 Viva Preparation Answers

<details>
<summary>Click to expand complete viva Q&A</summary>

**Q1: What is the project?**  
A system that monitors elderly people using a webcam, detects falls without storing raw video, and alerts caregivers. It runs entirely on a local computer using computer vision and machine learning.

**Q2: Why privacy-preserving?**  
Elderly people deserve dignity. Storing continuous video is intrusive and a security risk. Our system extracts only skeletal coordinates and immediately discards the raw camera frame. The database never contains video or images.

**Q3: What is edge computing?**  
Processing data on the local device (the edge) instead of sending it to a remote cloud server. Benefits: lower latency, no internet dependency, lower cost, better privacy.

**Q4: What is skeletal pose estimation?**  
The process of detecting a person in an image and mapping the locations of their major body joints as (x, y) coordinates. We use 17 joints representing the full body.

**Q5: Why MediaPipe?**  
It runs on CPU without GPU, is open-source from Google, works on Raspberry Pi, and is well-documented. It provides reliable 33-point pose detection which we reduce to our 17 key joints.

**Q6: What is optical flow?**  
A computer vision technique that estimates how pixels (or feature points) move between consecutive frames. We use the Lucas-Kanade method which tracks specific corner points.

**Q7: What is Double Background Filtering?**  
Our custom approach: we maintain two background estimates — a short-term (3 frames) and long-term (5 frames) accumulation. We only consider a region as foreground if it differs significantly from BOTH backgrounds. This filters out fans, curtains, and lighting changes.

**Q8: What is the kinetic energy proxy?**  
A normalized measure of movement intensity calculated as the RMS (Root Mean Square) of all joint velocities. It is NOT actual physical kinetic energy (we don't know the person's mass), but it correlates with movement intensity.

**Q9: Why Random Forest?**  
Works well with small datasets, doesn't need a GPU, provides probability estimates, handles imbalanced classes, and is fast enough for edge devices. We can also inspect feature importance to understand what drives decisions.

**Q10: What is IRC?**  
Instantaneous Rate of Change — measures how suddenly movement intensity changes. Formula: IRC = (current - past) / N_frames. A fall causes a sudden spike in IRC. Threshold: 0.25 (prototype value).

**Q11: Why temporal validation?**  
A single "Fallen" frame prediction is unreliable. Sitting quickly also produces a similar posture momentarily. We require the fall signal to persist for N seconds AND the person to not recover before sending an alert.

**Q12: How is accuracy calculated?**  
Accuracy = (correct predictions) / (total predictions). Measured on a 20% held-out test set that was not used during training. We do NOT invent these numbers.

**Q13: What is F1-score?**  
The harmonic mean of precision and recall. F1 = 2 × (P × R) / (P + R). Important when classes are imbalanced (we have fewer "Fallen" samples than "Standing").

**Q14: What is stored in the database?**  
Only timestamps, activity labels (text), confidence scores (numbers), event types (text), duration (number), IRC values (number), and alert status (boolean). No images, no video, no face data.

**Q15: Can this run on Raspberry Pi?**  
The architecture is designed for it. MediaPipe has a Raspberry Pi build. The main bottleneck is FPS — expect ~5-10 FPS on Pi 4. All algorithms use CPU-only processing.

</details>

---

## 📜 License

MIT License — See `LICENSE` file.

---

## 👤 Author

**B.Tech CSE Major Project**  
Computer Vision · Edge AI · Privacy-Preserving Systems

---

*This system is a prototype intended for safety monitoring and does not replace professional medical monitoring.*
