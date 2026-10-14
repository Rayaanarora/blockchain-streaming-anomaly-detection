"""
Unsupervised anomaly detector for Zenodo Bitcoin transactions.

* Score: Isolation Forest anomaly score (= -score_samples, range ~0.3..0.8; higher = more
  unusual), fitted on a deterministic hash sample of an earlier block window.
* Percentile: rank of the score within the fit-sample score distribution.
* Flag: score >= the fit-sample quantile ``flag_quantile`` (default 0.999), i.e. roughly
  0.1% of in-distribution transactions are flagged by construction.
* Reason codes: the three features with the largest robust z-score
  |x - median| / (1.4826 * MAD) on the fit sample, with direction (high/low).

An anomaly score means "statistically unusual relative to the fit window". It is NOT a
probability of illicit activity: no reliable transaction-level labels exist for this data.
"""
import hashlib
import json
import os
import time
from typing import Dict, List

import joblib
import numpy as np
from sklearn.ensemble import IsolationForest

from src.anomaly.zenodo_features import FEATURE_VERSION, MODEL_FEATURES

MODEL_ID = "zenodo-isolation-forest"


class ZenodoAnomalyDetector:
    def __init__(self, n_estimators: int = 200, max_samples: int = 256, flag_quantile: float = 0.999,
                 random_state: int = 42, features: List[str] = None):
        self.features = list(features or MODEL_FEATURES)
        self.params = dict(n_estimators=n_estimators, max_samples=max_samples, random_state=random_state)
        self.flag_quantile = flag_quantile
        self.forest = None
        self.median = self.mad = None
        self.fit_scores_sorted = None
        self.threshold = None
        self.meta: Dict = {}

    # ------------------------------------------------------------------ fit
    def fit(self, X: np.ndarray, meta: Dict = None) -> "ZenodoAnomalyDetector":
        if X.shape[1] != len(self.features):
            raise ValueError(f"expected {len(self.features)} features, got {X.shape[1]}")
        if not np.isfinite(X).all():
            raise ValueError("non-finite values in fit matrix")
        t0 = time.time()
        self.median = np.median(X, axis=0)
        mad = np.median(np.abs(X - self.median), axis=0) * 1.4826
        # features with MAD == 0 (e.g. mostly-constant) fall back to std, then 1
        std = X.std(axis=0)
        self.mad = np.where(mad > 0, mad, np.where(std > 0, std, 1.0))
        self.forest = IsolationForest(n_jobs=-1, **self.params).fit(X)
        scores = -self.forest.score_samples(X)
        self.fit_scores_sorted = np.sort(scores).astype(np.float32)
        self.threshold = float(np.quantile(scores, self.flag_quantile))
        self.meta = {**(meta or {}), "fit_rows": int(len(X)), "fit_seconds": round(time.time() - t0, 3)}
        return self

    # ------------------------------------------------------------------ score
    def robust_z(self, X: np.ndarray) -> np.ndarray:
        return (X - self.median) / self.mad

    def score(self, X: np.ndarray, reasons_min_percentile: float = 0.0) -> Dict[str, np.ndarray]:
        """Score rows. Reason codes are generated for rows whose percentile >= reasons_min_percentile
        (empty string otherwise) - batch scoring uses 0.99 to avoid formatting 16.9M strings."""
        s = -self.forest.score_samples(X)
        pct = np.searchsorted(self.fit_scores_sorted, s, side="right") / len(self.fit_scores_sorted)
        reasons = np.full(len(X), "", dtype=object)
        idx = np.flatnonzero(pct >= reasons_min_percentile)
        if len(idx):
            z = self.robust_z(X[idx])
            top = np.argsort(-np.abs(z), axis=1)[:, :3]
            names = self.features
            for k, i in enumerate(idx):
                reasons[i] = ";".join(f"{names[j]}:{'high' if z[k, j] > 0 else 'low'}({z[k, j]:+.1f})"
                                      for j in top[k])
        return {"score": s, "percentile": pct, "flagged": s >= self.threshold, "reason_codes": reasons}

    # ------------------------------------------------------------------ persistence
    def save(self, directory: str, version: str) -> str:
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, "detector.joblib")
        joblib.dump({"forest": self.forest, "median": self.median, "mad": self.mad,
                     "fit_scores_sorted": self.fit_scores_sorted}, path)
        sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
        manifest = {
            "model_id": MODEL_ID, "model_version": version, "feature_version": FEATURE_VERSION,
            "features": self.features, "params": self.params, "flag_quantile": self.flag_quantile,
            "threshold": self.threshold, "score_type": "isolation_forest_anomaly_score",
            "score_semantics": "Higher = more unusual relative to the fit window (Isolation Forest, -score_samples). "
                               "Not a probability and not evidence of illicit activity.",
            "files": {"detector": "detector.joblib"}, "sha256": {"detector": sha}, "meta": self.meta,
            "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(os.path.join(directory, "manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        return directory

    @classmethod
    def load(cls, directory: str) -> "ZenodoAnomalyDetector":
        manifest = json.load(open(os.path.join(directory, "manifest.json")))
        path = os.path.join(directory, manifest["files"]["detector"])
        if hashlib.sha256(open(path, "rb").read()).hexdigest() != manifest["sha256"]["detector"]:
            raise RuntimeError(f"checksum mismatch for {path}")
        obj = joblib.load(path)
        det = cls(flag_quantile=manifest["flag_quantile"], features=manifest["features"], **manifest["params"])
        det.forest, det.median, det.mad = obj["forest"], obj["median"], obj["mad"]
        det.fit_scores_sorted = obj["fit_scores_sorted"]
        det.threshold = manifest["threshold"]
        det.meta = manifest["meta"]
        det.manifest = manifest
        return det
