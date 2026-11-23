"""
ml/classifier.py — Activity Classifier
========================================
PURPOSE:
    Load a trained model and classify the current activity from
    temporal feature vectors.

WHY RANDOM FOREST?
    - Works well with small/medium datasets
    - Doesn't need feature scaling (tree-based)
    - Provides probability estimates (model.predict_proba)
    - Resistant to overfitting with balanced class weights
    - Fast inference on CPU (no GPU required)
    - Interpretable — can inspect feature importance
    - Works on Raspberry Pi without problems

    Compared to deep learning:
    - We don't need thousands of samples
    - No GPU required
    - Simpler to train and debug
    - More suitable for a B.Tech prototype

MODULARITY:
    This class can be swapped with an SVM by changing only the
    internal model initialization. The predict() API stays the same.

HOW IT WORKS:
    Input: 36-dimensional temporal feature vector
           (9 features × 4 stats: mean, std, max, min)
    Output: Activity label + confidence

BEFORE USING THIS:
    You must first run: python -m ml.train_model
    This creates ml/model.pkl, ml/scaler.pkl, ml/label_encoder.pkl
"""

import numpy as np
import joblib
import logging
import os
from typing import Optional, Dict, Any, Tuple
import config

logger = logging.getLogger(__name__)


class ActivityClassifier:
    """
    Loads a trained Random Forest model and classifies activities.

    Falls back to a rule-based heuristic if no model is available yet
    (useful during Phase 1 testing before the model is trained).
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.is_ready = False

        self._load_model()

    def _load_model(self) -> None:
        """Load trained model files if they exist."""
        try:
            if (
                os.path.exists(config.MODEL_PATH)
                and os.path.exists(config.SCALER_PATH)
                and os.path.exists(config.LABEL_ENCODER_PATH)
            ):
                self.model = joblib.load(config.MODEL_PATH)
                self.scaler = joblib.load(config.SCALER_PATH)
                self.label_encoder = joblib.load(config.LABEL_ENCODER_PATH)
                self.is_ready = True
                logger.info(f"✅ Model loaded from {config.MODEL_PATH}")
            else:
                logger.warning(
                    "⚠️  No trained model found. Using rule-based fallback. "
                    "Run 'python -m ml.train_model' after collecting data."
                )
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.is_ready = False

    def predict(
        self,
        temporal_features: Optional[np.ndarray],
        aspect_ratio: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Classify the current activity.

        Args:
            temporal_features: 36-dimensional numpy array from TemporalWindow
            aspect_ratio:      Current frame's aspect ratio (for fallback)

        Returns:
            Dictionary:
                activity    (str):   Activity label
                confidence  (float): Model confidence 0-1
                probabilities (dict): Per-class probabilities
                source      (str):   "model" or "rule_based"
        """
        if self.is_ready and temporal_features is not None:
            return self._predict_with_model(temporal_features)
        else:
            return self._predict_rule_based(aspect_ratio, temporal_features)

    def _predict_with_model(self, features: np.ndarray) -> Dict[str, Any]:
        """Use trained Random Forest for prediction."""
        try:
            # Reshape to (1, N_features) for sklearn
            features_2d = features.reshape(1, -1)

            # Scale features using the same scaler used during training
            features_scaled = self.scaler.transform(features_2d)

            # Get probability for each class
            probabilities = self.model.predict_proba(features_scaled)[0]

            # Get predicted class index
            pred_idx = np.argmax(probabilities)
            confidence = float(probabilities[pred_idx])

            # Convert index to class label
            activity = self.label_encoder.inverse_transform([pred_idx])[0]

            # Build per-class probability dictionary
            classes = self.label_encoder.classes_
            prob_dict = {
                cls: float(prob)
                for cls, prob in zip(classes, probabilities)
            }

            return {
                "activity": activity,
                "confidence": confidence,
                "probabilities": prob_dict,
                "source": "model",
            }

        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return self._predict_rule_based(0.5, features)

    def _predict_rule_based(
        self, aspect_ratio: float, temporal_features: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Kinetic heuristic classifier using full temporal spatial-temporal dynamics.
        Ensures robust classification even prior to offline model training.
        """
        if temporal_features is not None and len(temporal_features) >= 9:
            ar_mean = float(temporal_features[0])
            kn_vel = float(temporal_features[8]) if len(temporal_features) > 8 else 0.0
            mi_mean = float(temporal_features[5]) if len(temporal_features) > 5 else 0.0
            h_acc_max = float(temporal_features[20]) if len(temporal_features) > 20 else 0.0
            ke_max = float(temporal_features[21]) if len(temporal_features) > 21 else 0.0

            if ar_mean > 1.0 or (aspect_ratio > 0.85 and (h_acc_max > 0.03 or ke_max > 0.06)):
                activity = "Fallen"
                probs = {"Fallen": 0.89, "Reclining": 0.06, "Sitting": 0.03, "Standing": 0.01, "Walking": 0.01}
                conf = 0.89
            elif kn_vel > 0.03 or mi_mean > 0.04:
                activity = "Walking"
                probs = {"Walking": 0.86, "Standing": 0.08, "Sitting": 0.03, "Reclining": 0.02, "Fallen": 0.01}
                conf = 0.86
            elif ar_mean > 0.82:
                activity = "Reclining"
                probs = {"Reclining": 0.83, "Sitting": 0.10, "Fallen": 0.04, "Standing": 0.02, "Walking": 0.01}
                conf = 0.83
            elif ar_mean > 0.56:
                activity = "Sitting"
                probs = {"Sitting": 0.85, "Reclining": 0.08, "Standing": 0.05, "Walking": 0.01, "Fallen": 0.01}
                conf = 0.85
            else:
                activity = "Standing"
                probs = {"Standing": 0.88, "Sitting": 0.07, "Walking": 0.03, "Reclining": 0.01, "Fallen": 0.01}
                conf = 0.88

            return {
                "activity": activity,
                "confidence": conf,
                "probabilities": probs,
                "source": "heuristic_kinetics",
            }

        # Fallback to single aspect ratio if temporal vector not ready
        if aspect_ratio > 0.95:
            activity = "Fallen"
            conf = 0.75
        elif aspect_ratio > 0.80:
            activity = "Reclining"
            conf = 0.70
        elif aspect_ratio > 0.55:
            activity = "Sitting"
            conf = 0.72
        else:
            activity = "Standing"
            conf = 0.80

        return {
            "activity": activity,
            "confidence": conf,
            "probabilities": {activity: conf},
            "source": "heuristic_geometry",
        }

    def reload(self) -> bool:
        """Reload model from disk (useful if model was retrained)."""
        self._load_model()
        return self.is_ready
