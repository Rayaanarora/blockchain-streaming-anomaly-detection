import time
from typing import Dict, Tuple, Optional, Any, Union
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier
)
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
    Provides validation-based threshold optimization and comprehensive evaluation.
    """
    def __init__(
        self,
        model_family: str = "rf",
        class_weight_ratio: Optional[Union[float, str]] = None,
        scale_input: bool = True,
        random_state: int = 42
    ):
        self.model_family = model_family.lower()
        self.class_weight_ratio = class_weight_ratio
        self.scale_input = scale_input
        self.random_state = random_state
        self.scaler = StandardScaler() if scale_input else None
        self.fit_time_sec: float = 0.0
        self.best_threshold: float = 0.5
        self.model: Optional[Any] = None

    def _init_estimator(self, effective_ratio: float) -> Any:
        weights = {0: 1.0, 1: float(effective_ratio)}
        
        if self.model_family == "lr":
            return LogisticRegression(
                class_weight=weights,
                max_iter=1000,
                random_state=self.random_state
            )
        elif self.model_family == "rf":
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=12,
                class_weight=weights,
                n_jobs=-1,
                random_state=self.random_state
            )
        elif self.model_family in ("hgbdt", "hist_gbdt", "hgb"):
            return HistGradientBoostingClassifier(
                class_weight=weights,
                max_iter=100,
                learning_rate=0.1,
                random_state=self.random_state
            )
        elif self.model_family in ("gbdt", "gbt"):
            return GradientBoostingClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state
            )
        else:
            raise ValueError(
                f"Unknown model family '{self.model_family}'. Choose 'lr', 'rf', 'gbdt', or 'hgbdt'."
            )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "CostSensitiveAnomalyModel":
        """
        Fit scaler and model strictly on training split.
        If class_weight_ratio is not specified, computes dynamic ratio strictly from y_train.
        """
        t0 = time.time()
        
        # 1. Determine effective class weight ratio strictly from training labels
        if self.class_weight_ratio is None or self.class_weight_ratio == "auto":
            n_licit = int(np.sum(y_train == 0))
            n_illicit = int(np.sum(y_train == 1))
            effective_ratio = float(n_licit / max(n_illicit, 1))
        else:
            effective_ratio = float(self.class_weight_ratio)
            
        self.effective_class_weight_ratio = effective_ratio
        self.model = self._init_estimator(effective_ratio)

        # 2. Fit feature scaler strictly on training split if enabled
        if self.scale_input:
            X_scaled = self.scaler.fit_transform(X_train)
        else:
            X_scaled = X_train

        # 3. Fit estimator
        if self.model_family in ("gbdt", "gbt"):
            sample_weight = np.where(y_train == 1, effective_ratio, 1.0)
            self.model.fit(X_scaled, y_train, sample_weight=sample_weight)
        else:
            self.model.fit(X_scaled, y_train)

        self.fit_time_sec = time.time() - t0
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability scores for anomaly class (illicit = 1)."""
        if self.model is None:
            raise RuntimeError("Model has not been fitted yet. Call fit() first.")
        X_proc = self.scaler.transform(X) if self.scale_input else X
        probs = self.model.predict_proba(X_proc)
        if probs.shape[1] == 2:
            return probs[:, 1]
        return probs[:, 0]

    def predict(self, X: np.ndarray, threshold: Optional[float] = None) -> np.ndarray:
        """Predict binary classification using chosen decision threshold."""
        thresh = self.best_threshold if threshold is None else threshold
        scores = self.predict_proba(X)
        return (scores >= thresh).astype(int)

    def tune_threshold(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        thresholds: Optional[np.ndarray] = None
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        Select decision threshold using validation data only.
        Maximizes illicit F1 score over candidate thresholds.
        """
        if thresholds is None:
            thresholds = np.linspace(0.05, 0.95, 91)

        scores = self.predict_proba(X_val)
        best_f1 = -1.0
        best_thresh = 0.5
        thresh_history = []

        for t in thresholds:
            preds = (scores >= t).astype(int)
            f1 = float(f1_score(y_val, preds, pos_label=1, zero_division=0))
            thresh_history.append({"threshold": round(float(t), 3), "val_f1": round(f1, 4)})
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = float(t)

        self.best_threshold = best_thresh
        val_eval = self.evaluate(X_val, y_val, threshold=best_thresh)
        return best_thresh, best_f1, val_eval

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, threshold: Optional[float] = None) -> Dict[str, Any]:
        """Compute comprehensive anomaly detection evaluation metrics and latency."""
        thresh = self.best_threshold if threshold is None else threshold
        t0 = time.time()
        scores = self.predict_proba(X_test)
        inference_time_sec = time.time() - t0
        
        preds = (scores >= thresh).astype(int)
        n_samples = len(y_test)
        latency_ms_per_tx = (inference_time_sec / max(n_samples, 1)) * 1000.0

        has_positives = (np.sum(y_test == 1) > 0)
        has_multiple_classes = len(np.unique(y_test)) > 1

        prec = float(precision_score(y_test, preds, pos_label=1, zero_division=0))
        rec = float(recall_score(y_test, preds, pos_label=1, zero_division=0))
        f1 = float(f1_score(y_test, preds, pos_label=1, zero_division=0))
        macro_f1 = float(f1_score(y_test, preds, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_test, preds, average="weighted", zero_division=0))

        roc_auc = float(roc_auc_score(y_test, scores)) if has_positives and has_multiple_classes else 0.0
        ap = float(average_precision_score(y_test, scores)) if has_positives and has_multiple_classes else 0.0

        cm = confusion_matrix(y_test, preds).tolist()
        illicit_cnt = int(np.sum(y_test == 1))
        licit_cnt = int(np.sum(y_test == 0))
        prevalence = float(illicit_cnt / max(n_samples, 1))

        return {
            "model_family": self.model_family,
            "threshold": round(float(thresh), 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "macro_f1": round(macro_f1, 4),
            "weighted_f1": round(weighted_f1, 4),
            "roc_auc": round(roc_auc, 4),
            "average_precision": round(ap, 4),
            "confusion_matrix": cm,
            "test_samples": n_samples,
            "illicit_count": illicit_cnt,
            "licit_count": licit_cnt,
            "illicit_prevalence": round(prevalence, 4),
            "fit_time_sec": round(self.fit_time_sec, 4),
            "inference_time_sec": round(inference_time_sec, 4),
            "latency_ms_per_tx": round(latency_ms_per_tx, 6)
        }

