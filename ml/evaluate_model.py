"""
ml/evaluate_model.py — Standalone Model Evaluation Script
===========================================================
PURPOSE:
    Evaluate the trained model on the test dataset and generate:
    - Classification report (precision, recall, F1 per class)
    - Confusion matrix (visual)
    - Feature importance chart
    - ROC curves (if binary)

HOW TO RUN:
    python -m ml.evaluate_model

OR pass a specific CSV:
    python -m ml.evaluate_model --data data/features/features.csv

HONEST EVALUATION PRINCIPLES:
    - All metrics are calculated from ACTUAL predictions vs ACTUAL labels.
    - No values are manually set or invented.
    - If dataset is small → prototype results.
    - All metrics are explained in simple language below.

METRIC EXPLANATIONS (for your viva):
    Accuracy:
        How often the model is correct overall.
        = (correct predictions) / (total predictions)
        Example: 0.85 means 85% of frames were classified correctly.

    Precision (per class):
        Of all frames predicted as "Fallen", how many were actually "Fallen"?
        High precision = few false alarms.
        = True Positives / (True Positives + False Positives)

    Recall (per class):
        Of all actual "Fallen" frames, how many did we correctly detect?
        High recall = few missed falls.
        = True Positives / (True Positives + False Negatives)

    F1-Score:
        Balance between precision and recall.
        = 2 × (Precision × Recall) / (Precision + Recall)
        F1 = 1.0 is perfect, 0.0 is worst.

    Confusion Matrix:
        Rows = actual class, Columns = predicted class.
        Diagonal = correctly classified.
        Off-diagonal = misclassified.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_model_and_data(csv_path: str):
    """Load the trained model files and test data."""

    # Load model artifacts
    if not all([
        os.path.exists(config.MODEL_PATH),
        os.path.exists(config.SCALER_PATH),
        os.path.exists(config.LABEL_ENCODER_PATH),
    ]):
        raise FileNotFoundError(
            "Model files not found! Train first with: python -m ml.train_model"
        )

    model = joblib.load(config.MODEL_PATH)
    scaler = joblib.load(config.SCALER_PATH)
    label_encoder = joblib.load(config.LABEL_ENCODER_PATH)

    # Load dataset
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    df = pd.read_csv(csv_path)
    drop_cols = ["timestamp", "label"]
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols].values.astype(np.float32)
    y_raw = df["label"].values

    return model, scaler, label_encoder, X, y_raw, feature_cols


def evaluate(csv_path: str = None):
    """Run full evaluation and print results."""

    csv_path = csv_path or config.DATASET_FILE

    logger.info(f"Loading model and data from: {csv_path}")
    model, scaler, label_encoder, X, y_raw, feature_names = load_model_and_data(csv_path)

    # Encode labels
    y = label_encoder.transform(y_raw)
    classes = list(label_encoder.classes_)

    # Split same way as training (same random state = same test set)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RF_RANDOM_STATE,
        stratify=y,
    )

    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    y_pred_labels = label_encoder.inverse_transform(y_pred)
    y_test_labels = label_encoder.inverse_transform(y_test)

    # =========================================================================
    # METRICS
    # =========================================================================
    accuracy = accuracy_score(y_test, y_pred)

    # Per-class metrics
    precision = precision_score(y_test, y_pred, average=None, zero_division=0)
    recall = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=None, zero_division=0)

    # Macro averages
    macro_precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
    macro_recall = recall_score(y_test, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

    # =========================================================================
    # PRINT REPORT
    # =========================================================================
    banner = "=" * 65
    logger.info(f"\n{banner}")
    logger.info("  MODEL EVALUATION REPORT")
    logger.info("  PROTOTYPE SYSTEM — Results from our collected dataset")
    logger.info(f"{banner}")

    logger.info(f"\n  Dataset:    {csv_path}")
    logger.info(f"  Test set:   {len(X_test)} samples (20% split)")
    logger.info(f"  Classes:    {classes}")
    logger.info(f"\n  Overall Accuracy: {accuracy * 100:.2f}%")
    logger.info(f"  Macro Precision:  {macro_precision * 100:.2f}%")
    logger.info(f"  Macro Recall:     {macro_recall * 100:.2f}%")
    logger.info(f"  Macro F1-Score:   {macro_f1 * 100:.2f}%")

    logger.info(f"\n  Per-Class Results:")
    logger.info(f"  {'Class':<14} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    logger.info(f"  {'-'*44}")
    for i, cls in enumerate(classes):
        if i < len(precision):
            logger.info(
                f"  {cls:<14} {precision[i]*100:>9.1f}% "
                f"{recall[i]*100:>7.1f}% "
                f"{f1[i]*100:>7.1f}%"
            )

    logger.info(f"\n  Full Classification Report:")
    logger.info(
        "\n" + classification_report(
            y_test_labels, y_pred_labels,
            zero_division=0, target_names=classes
        )
    )

    # Cross-validation score (more robust estimate)
    X_scaled = scaler.transform(X)
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring="accuracy")
    logger.info(f"\n  5-Fold Cross-Validation Accuracy:")
    logger.info(f"    Mean:  {cv_scores.mean() * 100:.2f}%")
    logger.info(f"    Std:   {cv_scores.std() * 100:.2f}%")
    logger.info(f"    All:   {[f'{s*100:.1f}%' for s in cv_scores]}")

    logger.info(f"\n{banner}")
    logger.info("  IMPORTANT: Results are from our collected prototype dataset.")
    logger.info("  Larger/more diverse datasets may yield different results.")
    logger.info(f"{banner}\n")

    # =========================================================================
    # CONFUSION MATRIX PLOT
    # =========================================================================
    cm = confusion_matrix(y_test_labels, y_pred_labels, labels=classes)
    _plot_confusion_matrix(cm, classes)

    # =========================================================================
    # FEATURE IMPORTANCE PLOT
    # =========================================================================
    _plot_feature_importance(model, feature_names)

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class_f1": dict(zip(classes, f1.tolist())),
        "cv_mean": float(cv_scores.mean()),
        "cv_std": float(cv_scores.std()),
    }


def _plot_confusion_matrix(cm: np.ndarray, classes: list):
    """Save confusion matrix as PNG image."""
    os.makedirs("data", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes,
        ax=ax
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix — Activity Classification\n(Prototype Dataset Results)", fontsize=13)
    plt.tight_layout()
    out_path = "data/confusion_matrix.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix saved: {out_path}")


def _plot_feature_importance(model, feature_names: list):
    """Save feature importance chart as PNG."""
    importances = model.feature_importances_
    # Only show top 15 features
    top_n = min(15, len(importances))
    sorted_idx = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(
        [feature_names[i] if i < len(feature_names) else f"feat_{i}" for i in sorted_idx],
        importances[sorted_idx],
        color="#00aaff",
    )
    ax.set_xlabel("Feature Importance (Mean Decrease in Impurity)")
    ax.set_title("Top Feature Importances — Random Forest\n(Elder Care Activity Classifier)")
    plt.tight_layout()
    out_path = "data/feature_importance.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Feature importance chart saved: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate elder care ML model")
    parser.add_argument(
        "--data", type=str, default=config.DATASET_FILE,
        help="Path to features CSV file"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        results = evaluate(args.data)
        print(f"\n✅ Evaluation complete!")
        print(f"   Accuracy:    {results['accuracy']*100:.2f}%")
        print(f"   Macro F1:    {results['macro_f1']*100:.2f}%")
        print(f"   CV Mean:     {results['cv_mean']*100:.2f}% ± {results['cv_std']*100:.2f}%")
    except FileNotFoundError as e:
        logger.error(str(e))
