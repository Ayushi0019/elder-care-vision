"""
check_setup.py - Quick Setup Verification Script
Run: python check_setup.py
"""
import sys
import os

print("\n" + "="*60)
print("  ELDER CARE MONITORING - SETUP VERIFICATION")
print("="*60)

print(f"\n[OK] Python: {sys.version.split()[0]}")

packages = {
    "cv2":          "opencv-python",
    "mediapipe":    "mediapipe",
    "numpy":        "numpy",
    "sklearn":      "scikit-learn",
    "streamlit":    "streamlit",
    "pandas":       "pandas",
    "matplotlib":   "matplotlib",
    "seaborn":      "seaborn",
    "joblib":       "joblib",
    "plyer":        "plyer",
    "plotly":       "plotly",
    "requests":     "requests",
    "loguru":       "loguru",
}

all_ok = True
print("\n  Package Check:")
print(f"  {'Package':<20} {'Status'}")
print(f"  {'-'*35}")

for module, pip_name in packages.items():
    try:
        pkg = __import__(module)
        ver = getattr(pkg, "__version__", "?")
        print(f"  [OK] {pip_name:<20} {ver}")
    except ImportError:
        print(f"  [MISSING] {pip_name:<20} NOT FOUND - run: pip install {pip_name}")
        all_ok = False

print("\n  Camera Check:")
try:
    import cv2
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            h, w = frame.shape[:2]
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"  [OK] Camera 0: {w}x{h} @ {fps:.0f}FPS")
        else:
            print("  [WARN] Camera opened but cannot read frame.")
        cap.release()
    else:
        print("  [FAIL] No camera found at index 0.")
except Exception as e:
    print(f"  [FAIL] Camera error: {e}")

print("\n  Config Check:")
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import config
    print(f"  [OK] config.py loaded")
    print(f"       SAVE_RAW_FRAMES = {config.SAVE_RAW_FRAMES}  (must be False for privacy)")
    print(f"       PRIVACY_MODE    = {config.PRIVACY_MODE}  (must be True)")
except Exception as e:
    print(f"  [FAIL] config error: {e}")

print("\n  Model Check:")
if os.path.exists("ml/model.pkl"):
    print("  [OK] Trained model found: ml/model.pkl")
else:
    print("  [INFO] No model yet - using rule-based fallback (train after data collection)")

print("\n" + "="*60)
if all_ok:
    print("  [PASS] ALL PACKAGES INSTALLED!")
    print("\n  Run order:")
    print("  1. python collect_dataset.py  (collect 150+ frames per activity)")
    print("  2. python -m ml.train_model   (train Random Forest)")
    print("  3. python -m ml.evaluate_model (see accuracy)")
    print("  4. python app.py              (live monitoring window)")
    print("  5. streamlit run dashboard/dashboard.py  (browser dashboard)")
else:
    print("  [FAIL] Missing packages. Run: pip install -r requirements.txt")
print("="*60)
