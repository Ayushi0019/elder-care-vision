"""
ml/train_model.py — Model Training Script
==========================================
PURPOSE:
    Load collected feature data, train a Random Forest classifier,
    evaluate it, and save the model.

HOW TO RUN:
    python -m ml.train_model

PREREQUISITES:
    1. Collect training data using: python collect_dataset.py
    2. This creates: data/features/features.csv
    3. Then run this training script.

WHAT IT DOES:
    1. Loads features.csv
    2. Splits into train (80%) and test (20%)
    3. Scales features using StandardScaler
    4. Trains Random Forest
    5. Evaluates on test set (prints real metrics — not fake numbers!)
    6. Saves model.pkl, scaler.pkl, label_encoder.pkl

HONEST EVALUATION NOTE:
    The accuracy numbers printed here are MEASURED on our actual dataset.
    If the dataset is small (few hundred samples), these numbers may not
    generalize to real-world use. This is a prototype.
    Future work: larger, validated dataset.
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
)

# Add parent directory to path so we can import config
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def load_dataset(csv_path: str) -> tuple:
    """
    Load features from CSV and prepare for training.

    Args:
        csv_path: Path to features.csv

    Returns:
        X (numpy array): Feature matrix
        y (numpy array): Labels
        feature_names (list): Column names
    """
    logger.info(f"Loading dataset from: {csv_path}")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. "
            f"Run collect_dataset.py first!"
        )

    df = pd.read_csv(csv_path)
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")

    # Show class distribution
    logger.info("\nClass distribution:")
    class_counts = df["label"].value_counts()
    for cls, count in class_counts.items():
        logger.info(f"  {cls}: {count} samples")

    # Separate features from label
    # Drop non-feature columns
    drop_cols = ["timestamp", "label"]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    X = df[feature_cols].values.astype(np.float32)
    y = df["label"].values
    feature_names = feature_cols

    logger.info(f"\nFeature matrix: {X.shape} | Labels: {y.shape}")
    return X, y, feature_names


def train(csv_path: str = None) -> dict:
    """
    Full training pipeline.

    Args:
        csv_path: Path to features CSV. Defaults to config.DATASET_FILE.

    Returns:
        Dictionary with evaluation metrics.
    """
    if csv_path is None:
        csv_path = config.DATASET_FILE

    # Step 1: Load data
    X, y_raw, feature_names = load_dataset(csv_path)

    # Step 2: Encode string labels to integers
    # e.g., "Standing"→0, "Walking"→1, "Sitting"→2, "Reclining"→3, "Fallen"→4
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    logger.info(f"\nLabel encoding: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")

    # Step 3: Train/test split
    # 80% for training, 20% for testing
    # stratify=y ensures same class proportions in both splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=config.TEST_SIZE,
        random_state=config.RF_RANDOM_STATE,
        stratify=y,
    )
    logger.info(f"\nTrain: {len(X_train)} samples | Test: {len(X_test)} samples")

    # Step 4: Feature scaling (StandardScaler)
    # Scales each feature to have mean=0, std=1
    # Random Forest doesn't strictly require this, but it helps
    # with consistency and when comparing to SVM later.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fit on train only!
    X_test_scaled = scaler.transform(X_test)        # Apply same scale to test

    # Step 5: Train Random Forest
    logger.info(f"\nTraining Random Forest:")
    logger.info(f"  n_estimators={config.RF_N_ESTIMATORS}")
    logger.info(f"  max_depth={config.RF_MAX_DEPTH}")
    logger.info(f"  class_weight={config.RF_CLASS_WEIGHT}")

    model = RandomForestClassifier(
        n_estimators=config.RF_N_ESTIMATORS,
        max_depth=config.RF_MAX_DEPTH,
        random_state=config.RF_RANDOM_STATE,
        class_weight=config.RF_CLASS_WEIGHT,  # Handles imbalanced classes
        n_jobs=-1,  # Use all CPU cores
    )

    model.fit(X_train_scaled, y_train)
    logger.info("Random Forest training complete!")

    # Train Support Vector Machine (SVM) as specified in Chapter 4 Synopsis
    logger.info("Training Support Vector Machine (SVM) with RBF kernel...")
    svm_model = SVC(kernel="rbf", probability=True, random_state=config.RF_RANDOM_STATE)
    svm_model.fit(X_train_scaled, y_train)
    logger.info("SVM training complete!")

    # Step 6: Evaluate on test set
    y_pred = model.predict(X_test_scaled)
    y_pred_svm = svm_model.predict(X_test_scaled)
    y_pred_labels = label_encoder.inverse_transform(y_pred)
    y_test_labels = label_encoder.inverse_transform(y_test)

    accuracy = accuracy_score(y_test, y_pred)
    svm_accuracy = accuracy_score(y_test, y_pred_svm)

    logger.info("\n" + "="*60)
    logger.info("EVALUATION RESULTS (measured on actual test data)")
    logger.info("="*60)
    logger.info(f"Random Forest Accuracy: {accuracy * 100:.2f}%")
    logger.info(f"Support Vector Machine Accuracy: {svm_accuracy * 100:.2f}%")
    logger.info("\nPer-Class Report:")
    logger.info(
        classification_report(y_test_labels, y_pred_labels, zero_division=0)
    )

    logger.info("Confusion Matrix:")
    cm = confusion_matrix(y_test_labels, y_pred_labels, labels=label_encoder.classes_)
    logger.info(f"Classes: {list(label_encoder.classes_)}")
    logger.info(str(cm))

    # Feature importance (how much each feature contributes)
    importances = model.feature_importances_
    sorted_idx = np.argsort(importances)[::-1]
    logger.info("\nTop Feature Importances:")
    for i, idx in enumerate(sorted_idx[:5]):
        if idx < len(feature_names):
            logger.info(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.4f}")

    # Step 7: Save model files
    os.makedirs(os.path.dirname(config.MODEL_PATH), exist_ok=True)

    joblib.dump(model, config.MODEL_PATH)
    joblib.dump(svm_model, os.path.join(os.path.dirname(config.MODEL_PATH), "svm_model.pkl"))
    joblib.dump(scaler, config.SCALER_PATH)
    joblib.dump(label_encoder, config.LABEL_ENCODER_PATH)

    logger.info(f"\n✅ Random Forest saved: {config.MODEL_PATH}")
    logger.info(f"✅ SVM saved: {os.path.join(os.path.dirname(config.MODEL_PATH), 'svm_model.pkl')}")
    logger.info(f"✅ Scaler saved: {config.SCALER_PATH}")
    logger.info(f"✅ Label encoder saved: {config.LABEL_ENCODER_PATH}")

    return {
        "accuracy": accuracy,
        "svm_accuracy": svm_accuracy,
        "classes": list(label_encoder.classes_),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
    }


if __name__ == "__main__":
    try:
        results = train()
        logger.info(f"\n🎉 Training complete! Accuracy: {results['accuracy']*100:.1f}%")
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.info("\nSteps to fix:")
        logger.info("1. Run: python collect_dataset.py")
        logger.info("2. Collect data for each activity class")
        logger.info("3. Then run: python -m ml.train_model")
