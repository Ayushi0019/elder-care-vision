"""
generate_synthetic_dataset.py — Academic Dataset Synthesizer
============================================================
PURPOSE:
    Bootstraps the elder care monitoring system with realistic,
    physiologically accurate kinetic feature distributions for all 5 classes:
    Standing, Walking, Sitting, Reclining, Fallen.

    Eliminates the empty dataset barrier (features.csv = 0 bytes) and
    ensures proper 36-dimensional temporal feature alignment with
    TemporalWindow and ActivityClassifier.
"""

import os
import csv
import random
import math
from datetime import datetime

# Feature order matching TemporalWindow.get_temporal_feature_names()
BASE_FEATURES = [
    "aspect_ratio",
    "head_vy",
    "head_accel",
    "kinetic_energy",
    "body_vy",
    "movement_intensity",
    "shoulder_vel",
    "hip_vel",
    "knee_vel"
]

STATS = ["mean", "std", "max", "min"]

FEATURE_NAMES = [f"{feat}_{stat}" for stat in STATS for feat in BASE_FEATURES]
CLASSES = ["Standing", "Walking", "Sitting", "Reclining", "Fallen"]


def clamp(val, low, high):
    return max(low, min(high, val))


def sample_normal(mean, std, low=0.0, high=10.0):
    val = random.gauss(mean, std)
    return clamp(val, low, high)


def generate_standing_sample():
    # Standing: tall posture (low AR), minimal movement
    ar_mean = sample_normal(0.40, 0.03, 0.32, 0.50)
    ar_std = sample_normal(0.015, 0.005, 0.005, 0.03)
    ar_max = ar_mean + sample_normal(0.02, 0.008, 0.005, 0.05)
    ar_min = ar_mean - sample_normal(0.02, 0.008, 0.005, 0.05)

    h_vy_mean = sample_normal(0.002, 0.001, 0.0, 0.01)
    h_vy_std = sample_normal(0.002, 0.001, 0.0005, 0.008)
    h_vy_max = h_vy_mean + 0.004
    h_vy_min = max(0.0, h_vy_mean - 0.003)

    h_acc_mean = sample_normal(0.001, 0.0008, 0.0, 0.005)
    h_acc_std = sample_normal(0.001, 0.0005, 0.0002, 0.004)
    h_acc_max = h_acc_mean + 0.003
    h_acc_min = 0.0

    ke_mean = sample_normal(0.008, 0.003, 0.002, 0.018)
    ke_std = sample_normal(0.003, 0.001, 0.001, 0.008)
    ke_max = ke_mean + 0.006
    ke_min = max(0.0, ke_mean - 0.005)

    b_vy_mean = sample_normal(0.002, 0.001, 0.0, 0.008)
    b_vy_std = 0.002
    b_vy_max = 0.006
    b_vy_min = 0.0

    mi_mean = sample_normal(0.012, 0.004, 0.003, 0.025)
    mi_std = sample_normal(0.004, 0.001, 0.001, 0.008)
    mi_max = mi_mean + 0.008
    mi_min = max(0.0, mi_mean - 0.006)

    s_vel_mean = sample_normal(0.005, 0.002, 0.001, 0.012)
    s_vel_std = 0.002
    s_vel_max = s_vel_mean + 0.004
    s_vel_min = 0.001

    hip_vel_mean = sample_normal(0.004, 0.001, 0.001, 0.010)
    hip_vel_std = 0.0015
    hip_vel_max = hip_vel_mean + 0.003
    hip_vel_min = 0.001

    kn_vel_mean = sample_normal(0.004, 0.001, 0.001, 0.010)
    kn_vel_std = 0.0015
    kn_vel_max = kn_vel_mean + 0.003
    kn_vel_min = 0.001

    means = [ar_mean, h_vy_mean, h_acc_mean, ke_mean, b_vy_mean, mi_mean, s_vel_mean, hip_vel_mean, kn_vel_mean]
    stds = [ar_std, h_vy_std, h_acc_std, ke_std, b_vy_std, mi_std, s_vel_std, hip_vel_std, kn_vel_std]
    maxs = [ar_max, h_vy_max, h_acc_max, ke_max, b_vy_max, mi_max, s_vel_max, hip_vel_max, kn_vel_max]
    mins = [ar_min, h_vy_min, h_acc_min, ke_min, b_vy_min, mi_min, s_vel_min, hip_vel_min, kn_vel_min]
    return means + stds + maxs + mins


def generate_walking_sample():
    # Walking: rhythmic oscillation, higher knee/hip velocity
    ar_mean = sample_normal(0.44, 0.04, 0.35, 0.55)
    ar_std = sample_normal(0.04, 0.01, 0.02, 0.07)
    ar_max = ar_mean + sample_normal(0.06, 0.015, 0.03, 0.10)
    ar_min = ar_mean - sample_normal(0.05, 0.015, 0.02, 0.08)

    h_vy_mean = sample_normal(0.012, 0.004, 0.004, 0.025)
    h_vy_std = sample_normal(0.010, 0.003, 0.004, 0.020)
    h_vy_max = h_vy_mean + 0.018
    h_vy_min = max(0.0, h_vy_mean - 0.008)

    h_acc_mean = sample_normal(0.008, 0.003, 0.002, 0.018)
    h_acc_std = sample_normal(0.006, 0.002, 0.002, 0.012)
    h_acc_max = h_acc_mean + 0.012
    h_acc_min = 0.001

    ke_mean = sample_normal(0.055, 0.015, 0.025, 0.095)
    ke_std = sample_normal(0.018, 0.005, 0.008, 0.035)
    ke_max = ke_mean + 0.035
    ke_min = max(0.005, ke_mean - 0.025)

    b_vy_mean = sample_normal(0.010, 0.003, 0.003, 0.020)
    b_vy_std = 0.008
    b_vy_max = b_vy_mean + 0.015
    b_vy_min = 0.001

    mi_mean = sample_normal(0.075, 0.020, 0.035, 0.130)
    mi_std = sample_normal(0.025, 0.008, 0.010, 0.045)
    mi_max = mi_mean + 0.045
    mi_min = max(0.010, mi_mean - 0.030)

    s_vel_mean = sample_normal(0.030, 0.008, 0.012, 0.055)
    s_vel_std = 0.010
    s_vel_max = s_vel_mean + 0.020
    s_vel_min = 0.005

    hip_vel_mean = sample_normal(0.040, 0.010, 0.015, 0.070)
    hip_vel_std = 0.012
    hip_vel_max = hip_vel_mean + 0.025
    hip_vel_min = 0.008

    kn_vel_mean = sample_normal(0.065, 0.018, 0.025, 0.110)
    kn_vel_std = 0.020
    kn_vel_max = kn_vel_mean + 0.040
    kn_vel_min = 0.010

    means = [ar_mean, h_vy_mean, h_acc_mean, ke_mean, b_vy_mean, mi_mean, s_vel_mean, hip_vel_mean, kn_vel_mean]
    stds = [ar_std, h_vy_std, h_acc_std, ke_std, b_vy_std, mi_std, s_vel_std, hip_vel_std, kn_vel_std]
    maxs = [ar_max, h_vy_max, h_acc_max, ke_max, b_vy_max, mi_max, s_vel_max, hip_vel_max, kn_vel_max]
    mins = [ar_min, h_vy_min, h_acc_min, ke_min, b_vy_min, mi_min, s_vel_min, hip_vel_min, kn_vel_min]
    return means + stds + maxs + mins


def generate_sitting_sample():
    # Sitting: moderate aspect ratio (~0.65-0.78), static head/limbs
    ar_mean = sample_normal(0.68, 0.05, 0.58, 0.82)
    ar_std = sample_normal(0.018, 0.006, 0.005, 0.035)
    ar_max = ar_mean + 0.03
    ar_min = ar_mean - 0.03

    h_vy_mean = sample_normal(0.003, 0.001, 0.0, 0.010)
    h_vy_std = 0.002
    h_vy_max = 0.008
    h_vy_min = 0.0

    h_acc_mean = sample_normal(0.002, 0.001, 0.0, 0.006)
    h_acc_std = 0.001
    h_acc_max = 0.005
    h_acc_min = 0.0

    ke_mean = sample_normal(0.009, 0.003, 0.002, 0.020)
    ke_std = 0.003
    ke_max = ke_mean + 0.007
    ke_min = max(0.0, ke_mean - 0.006)

    b_vy_mean = 0.002
    b_vy_std = 0.001
    b_vy_max = 0.005
    b_vy_min = 0.0

    mi_mean = sample_normal(0.014, 0.005, 0.003, 0.030)
    mi_std = 0.004
    mi_max = mi_mean + 0.010
    mi_min = max(0.0, mi_mean - 0.008)

    s_vel_mean = 0.006
    s_vel_std = 0.002
    s_vel_max = 0.012
    s_vel_min = 0.001

    hip_vel_mean = 0.003
    hip_vel_std = 0.001
    hip_vel_max = 0.006
    hip_vel_min = 0.0

    kn_vel_mean = 0.005
    kn_vel_std = 0.002
    kn_vel_max = 0.010
    kn_vel_min = 0.001

    means = [ar_mean, h_vy_mean, h_acc_mean, ke_mean, b_vy_mean, mi_mean, s_vel_mean, hip_vel_mean, kn_vel_mean]
    stds = [ar_std, h_vy_std, h_acc_std, ke_std, b_vy_std, mi_std, s_vel_std, hip_vel_std, kn_vel_std]
    maxs = [ar_max, h_vy_max, h_acc_max, ke_max, b_vy_max, mi_max, s_vel_max, hip_vel_max, kn_vel_max]
    mins = [ar_min, h_vy_min, h_acc_min, ke_min, b_vy_min, mi_min, s_vel_min, hip_vel_min, kn_vel_min]
    return means + stds + maxs + mins


def generate_reclining_sample():
    # Reclining: horizontal aspect ratio (0.85-1.1), low acceleration
    ar_mean = sample_normal(0.96, 0.07, 0.82, 1.15)
    ar_std = sample_normal(0.015, 0.005, 0.005, 0.030)
    ar_max = ar_mean + 0.03
    ar_min = ar_mean - 0.03

    h_vy_mean = 0.002
    h_vy_std = 0.001
    h_vy_max = 0.006
    h_vy_min = 0.0

    h_acc_mean = 0.001
    h_acc_std = 0.001
    h_acc_max = 0.004
    h_acc_min = 0.0

    ke_mean = sample_normal(0.006, 0.002, 0.001, 0.015)
    ke_std = 0.002
    ke_max = 0.012
    ke_min = 0.001

    b_vy_mean = 0.001
    b_vy_std = 0.001
    b_vy_max = 0.004
    b_vy_min = 0.0

    mi_mean = sample_normal(0.010, 0.003, 0.002, 0.020)
    mi_std = 0.003
    mi_max = 0.018
    mi_min = 0.002

    s_vel_mean = 0.004
    s_vel_std = 0.001
    s_vel_max = 0.008
    s_vel_min = 0.001

    hip_vel_mean = 0.002
    hip_vel_std = 0.001
    hip_vel_max = 0.005
    hip_vel_min = 0.0

    kn_vel_mean = 0.003
    kn_vel_std = 0.001
    kn_vel_max = 0.007
    kn_vel_min = 0.001

    means = [ar_mean, h_vy_mean, h_acc_mean, ke_mean, b_vy_mean, mi_mean, s_vel_mean, hip_vel_mean, kn_vel_mean]
    stds = [ar_std, h_vy_std, h_acc_std, ke_std, b_vy_std, mi_std, s_vel_std, hip_vel_std, kn_vel_std]
    maxs = [ar_max, h_vy_max, h_acc_max, ke_max, b_vy_max, mi_max, s_vel_max, hip_vel_max, kn_vel_max]
    mins = [ar_min, h_vy_min, h_acc_min, ke_min, b_vy_min, mi_min, s_vel_min, hip_vel_min, kn_vel_min]
    return means + stds + maxs + mins


def generate_fallen_sample():
    # Fallen: wide aspect ratio (>1.2), high historical max acceleration & kinetic energy spike
    ar_mean = sample_normal(1.38, 0.16, 1.05, 1.85)
    ar_std = sample_normal(0.12, 0.04, 0.04, 0.25)
    ar_max = ar_mean + sample_normal(0.20, 0.06, 0.08, 0.40)
    ar_min = max(0.45, ar_mean - sample_normal(0.40, 0.10, 0.20, 0.70))

    h_vy_mean = sample_normal(0.038, 0.012, 0.015, 0.075)
    h_vy_std = sample_normal(0.045, 0.015, 0.020, 0.090)
    h_vy_max = sample_normal(0.145, 0.035, 0.085, 0.260) # Massive downward plunge
    h_vy_min = 0.0

    h_acc_mean = sample_normal(0.028, 0.010, 0.010, 0.060)
    h_acc_std = sample_normal(0.035, 0.012, 0.015, 0.075)
    h_acc_max = sample_normal(0.115, 0.030, 0.065, 0.210) # Sharp deceleration on impact
    h_acc_min = 0.0

    ke_mean = sample_normal(0.068, 0.020, 0.030, 0.125)
    ke_std = sample_normal(0.060, 0.018, 0.025, 0.110)
    ke_max = sample_normal(0.210, 0.050, 0.120, 0.380) # Kinetic energy explosion
    ke_min = sample_normal(0.002, 0.001, 0.0, 0.005)    # Post-fall immobility

    b_vy_mean = sample_normal(0.032, 0.010, 0.012, 0.065)
    b_vy_std = 0.038
    b_vy_max = sample_normal(0.125, 0.030, 0.070, 0.220)
    b_vy_min = 0.0

    mi_mean = sample_normal(0.095, 0.028, 0.040, 0.180)
    mi_std = sample_normal(0.080, 0.024, 0.030, 0.150)
    mi_max = sample_normal(0.280, 0.065, 0.160, 0.480)
    mi_min = 0.003

    s_vel_mean = sample_normal(0.055, 0.018, 0.020, 0.110)
    s_vel_std = 0.045
    s_vel_max = sample_normal(0.160, 0.040, 0.090, 0.280)
    s_vel_min = 0.002

    hip_vel_mean = sample_normal(0.048, 0.015, 0.018, 0.095)
    hip_vel_std = 0.040
    hip_vel_max = sample_normal(0.140, 0.035, 0.080, 0.250)
    hip_vel_min = 0.002

    kn_vel_mean = sample_normal(0.052, 0.016, 0.020, 0.105)
    kn_vel_std = 0.042
    kn_vel_max = sample_normal(0.155, 0.038, 0.085, 0.270)
    kn_vel_min = 0.002

    means = [ar_mean, h_vy_mean, h_acc_mean, ke_mean, b_vy_mean, mi_mean, s_vel_mean, hip_vel_mean, kn_vel_mean]
    stds = [ar_std, h_vy_std, h_acc_std, ke_std, b_vy_std, mi_std, s_vel_std, hip_vel_std, kn_vel_std]
    maxs = [ar_max, h_vy_max, h_acc_max, ke_max, b_vy_max, mi_max, s_vel_max, hip_vel_max, kn_vel_max]
    mins = [ar_min, h_vy_min, h_acc_min, ke_min, b_vy_min, mi_min, s_vel_min, hip_vel_min, kn_vel_min]
    return means + stds + maxs + mins


GENERATORS = {
    "Standing": generate_standing_sample,
    "Walking": generate_walking_sample,
    "Sitting": generate_sitting_sample,
    "Reclining": generate_reclining_sample,
    "Fallen": generate_fallen_sample,
}


def build_synthetic_dataset(samples_per_class=300, output_path="data/features/features.csv"):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    total_written = 0

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = ["timestamp"] + FEATURE_NAMES + ["label"]
        writer.writerow(header)

        for label in CLASSES:
            gen_fn = GENERATORS[label]
            for _ in range(samples_per_class):
                ts = datetime.now().isoformat()
                feats = gen_fn()
                writer.writerow([ts] + [round(v, 6) for v in feats] + [label])
                total_written += 1

    print(f"Generated {total_written} balanced synthetic kinematic samples in {output_path}")
    print(f"Classes: {CLASSES} ({samples_per_class} per class)")
    print(f"Feature dimensions: {len(FEATURE_NAMES)} (36 temporal features)")


if __name__ == "__main__":
    import sys
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(base_dir, "data", "features", "features.csv")
    build_synthetic_dataset(samples_per_class=300, output_path=out)
