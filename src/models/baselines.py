"""
Baseline anomaly detection models with cost-sensitive class balancing
and temporal train/test splitting protocols.
"""
from typing import Dict, Tuple, Optional
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix
)


class CostSensitiveAnomalyModel:
    """
    Standardized wrapper for anomaly detection baselines.
    Ensures feature scalers and probability calibrations are fitted strictly on train data.
    """
    def __init__(
        self,
        model_family: str = "rf",
        class_weight_ratio: float = 4.32,
        random_state: int = 42
    ):
        self.model_family = model_family.lower()
        self.class_weight_ratio = class_weight_ratio
        self.random_state = random_state
        self.scaler = StandardScaler()
        
        weights = {0: 1.0, 1: float(class_weight_ratio)}
        
        if self.model_family == "lr":
            self.model = LogisticRegression(
                class_weight=weights,
                max_iter=1000,
                random_state=self.random_state
            )
        elif self.model_family == "rf":
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                class_weight=weights,
                n_jobs=-1,
                random_state=self.random_state
            )
        elif self.model_family in ("gbdt", "gbt"):
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown model family '{model_family}'. Choose 'lr', 'rf', or 'gbdt'.")

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "CostSensitiveAnomalyModel":
        """Fit scaler and model strictly on training split."""
        X_scaled = self.scaler.fit_transform(X_train)
        
        # GradientBoostingClassifier does not accept class_weight in constructor, apply sample weights
        if self.model_family in ("gbdt", "gbt"):
            sample_weight = np.where(y_train == 1, self.class_weight_ratio, 1.0)
            self.model.fit(X_scaled, y_train, sample_weight=sample_weight)
        else:
            self.model.fit(X_scaled, y_train)
            
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability scores for anomaly class."""
        X_scaled = self.scaler.transform(X)
        probs = self.model.predict_proba(X_scaled)
        if probs.shape[1] == 2:
            return probs[:, 1]
        return probs[:, 0]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary classification with configurable decision threshold."""
        scores = self.predict_proba(X)
        return (scores >= threshold).astype(int)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
        """Compute comprehensive anomaly detection evaluation metrics."""
        scores = self.predict_proba(X_test)
        preds = (scores >= threshold).astype(int)
        
        # Check if test set contains at least 1 positive
        has_positives = (np.sum(y_test == 1) > 0)
        
        prec = float(precision_score(y_test, preds, zero_division=0))
        rec = float(recall_score(y_test, preds, zero_division=0))
        f1 = float(f1_score(y_test, preds, zero_division=0))
        
        roc_auc = float(roc_auc_score(y_test, scores)) if has_positives and len(np.unique(y_test)) > 1 else 0.0
        ap = float(average_precision_score(y_test, scores)) if has_positives and len(np.unique(y_test)) > 1 else 0.0
        
        cm = confusion_matrix(y_test, preds).tolist()
        
        return {
            "model_family": self.model_family,
            "threshold": threshold,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(roc_auc, 4),
            "average_precision": round(ap, 4),
            "confusion_matrix": cm,
            "test_samples": len(y_test),
            "illicit_count": int(np.sum(y_test == 1)),
            "licit_count": int(np.sum(y_test == 0))
        }
