"""
Exportable Elliptic model bundle and its inference interface.

A bundle directory contains ``manifest.json`` (model id, version, config, feature
schema, threshold, validation/test metrics, file hashes) plus the fitted scaler and
model files. ``EllipticModelBundle.predict`` accepts a *snapshot*: a table of nodes
(tx_id + f1..f165) and, for graph-dependent configs, the edges among them.

Graph semantics: Elliptic edges are intra-time-step, so the correct graph context for
a transaction is the full transaction graph of its time step. Callers must pass the
whole snapshot; edges that reference transactions outside the snapshot are dropped
and the affected nodes are reported with ``graph_context="partial"``. A tabular-only
row is never silently scored by a graph model: a missing ``edges`` argument raises.
"""
import hashlib
import json
import os
import time
from typing import Dict, Optional

import joblib
import numpy as np
import pandas as pd

FEATURE_COLUMNS = [f"f{i}" for i in range(1, 166)]
MODEL_ID = "elliptic-supervised"

# config -> list of (component name, feature recipe)
RECIPES = {
    "hgb_165_legacy": [("hgb_165_legacy", "x165")],
    "hgb_165_tuned": [("hgb_165_tuned", "x165")],
    "xgb_165_legacy": [("xgb_165_legacy", "x165")],
    "xgb_165_tuned": [("xgb_165_tuned", "x165")],
    "rf_165": [("rf_165", "x165")],
    "hgb_lf93_tuned": [("hgb_lf93_tuned", "x93")],
    "hgb_165_nbr": [("hgb_165_nbr", "x165_nbr")],
    "gat_alone": [("gat", "gat_prob")],
    "xgb_165_gat_cf": [("xgb_165_gat_cf", "x165_emb")],
    "hgb_165_gat_cf": [("hgb_165_gat_cf", "x165_emb")],
    "hgb_165_gat_insample": [("hgb_165_gat_insample", "x165_emb")],
    "ens_hgb_hgbgat": [("hgb_165_tuned", "x165"), ("hgb_165_gat_cf", "x165_emb")],
}
GRAPH_RECIPES = {"x165_nbr", "gat_prob", "x165_emb"}


class InputValidationError(ValueError):
    """Raised when a snapshot does not satisfy the model's input contract."""


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _save_component(obj, directory: str, name: str) -> str:
    import xgboost as xgb
    if isinstance(obj, xgb.XGBClassifier):
        path = os.path.join(directory, f"{name}.ubj")
        obj.save_model(path)
    else:
        path = os.path.join(directory, f"{name}.joblib")
        joblib.dump(obj, path)
    return os.path.basename(path)


def export_bundle(out_root: str, version: str, config: str, fitted: Dict, scaler, gat_direction: str,
                  threshold: float, metrics_seed42: Dict, summary: Dict, meta: Dict) -> str:
    import torch
    directory = os.path.join(out_root, version)
    if os.path.exists(os.path.join(directory, "manifest.json")):
        raise FileExistsError(f"bundle {directory} already exists; bump --model-version instead of overwriting")
    os.makedirs(directory, exist_ok=True)
    files = {"scaler": "scaler.joblib"}
    joblib.dump(scaler, os.path.join(directory, "scaler.joblib"))
    recipe = RECIPES[config]
    needs_gat = any(r in ("gat_prob", "x165_emb") for _, r in recipe)
    components = []
    for comp, feat in recipe:
        if comp == "gat":
            components.append({"name": "gat", "features": feat, "file": None})
            continue
        components.append({"name": comp, "features": feat, "file": _save_component(fitted[comp], directory, comp)})
    if needs_gat:
        torch.save({"config": fitted["gat"].config, "state_dict": fitted["gat"].state_dict()},
                   os.path.join(directory, "gat.pt"))
        files["gat"] = "gat.pt"
    manifest = {
        "model_id": MODEL_ID,
        "model_version": version,
        "config": config,
        "score_type": "illicit_probability",
        "score_semantics": "Probability-like score of the Elliptic 'illicit' class (label 1) from a supervised "
                           "model trained on labelled steps 1-29. Not calibrated across datasets.",
        "decision_threshold": threshold,
        "threshold_source": "validation steps 30-34, F1-maximising, seed 42",
        "requires_graph": any(r in GRAPH_RECIPES for _, r in recipe),
        "gat_edge_direction": gat_direction if needs_gat else None,
        "feature_schema": {"id_column": "tx_id", "time_column": "time_step", "features": FEATURE_COLUMNS,
                           "dtype": "float32", "raw_unscaled": True},
        "components": components,
        "files": {**files, **{c["name"]: c["file"] for c in components if c["file"]}},
        "test_metrics_seed42": metrics_seed42,
        "five_seed_summary": summary,
        "training_meta": meta,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest["sha256"] = {k: _sha256(os.path.join(directory, v)) for k, v in manifest["files"].items()}
    with open(os.path.join(directory, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=float)
    return directory


class EllipticModelBundle:
    def __init__(self, directory: str):
        self.directory = directory
        with open(os.path.join(directory, "manifest.json"), encoding="utf-8") as f:
            self.manifest = json.load(f)
        for key, fname in self.manifest["files"].items():
            if _sha256(os.path.join(directory, fname)) != self.manifest["sha256"][key]:
                raise RuntimeError(f"checksum mismatch for {fname} in {directory}")
        self.scaler = joblib.load(os.path.join(directory, self.manifest["files"]["scaler"]))
        self.components = []
        for c in self.manifest["components"]:
            self.components.append((c["name"], c["features"], self._load(c)))
        self.gat = None
        if "gat" in self.manifest["files"]:
            import torch
            from src.models.graph_encoder import load_encoder
            self.gat = load_encoder(torch.load(os.path.join(directory, "gat.pt"), weights_only=True))

    def _load(self, c):
        if c["file"] is None:
            return None
        path = os.path.join(self.directory, c["file"])
        if path.endswith(".ubj"):
            import xgboost as xgb
            m = xgb.XGBClassifier()
            m.load_model(path)
            return m
        return joblib.load(path)

    # ------------------------------------------------------------------ properties
    @property
    def model_id(self) -> str:
        return self.manifest["model_id"]

    @property
    def version(self) -> str:
        return self.manifest["model_version"]

    @property
    def requires_graph(self) -> bool:
        return bool(self.manifest["requires_graph"])

    @property
    def threshold(self) -> float:
        return float(self.manifest["decision_threshold"])

    # ------------------------------------------------------------------ inference
    def validate_nodes(self, nodes: pd.DataFrame) -> np.ndarray:
        missing = [c for c in ["tx_id"] + FEATURE_COLUMNS if c not in nodes.columns]
        if missing:
            raise InputValidationError(f"missing columns: {missing[:5]}{'...' if len(missing) > 5 else ''}")
        if nodes["tx_id"].duplicated().any():
            raise InputValidationError("duplicate tx_id in snapshot")
        X = nodes[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
        if not np.isfinite(X).all():
            raise InputValidationError("non-finite feature values (NaN/inf) in snapshot")
        return X.astype(np.float32)

    def predict(self, nodes: pd.DataFrame, edges: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Score a snapshot. ``edges`` needs columns source_tx_id, target_tx_id (required if requires_graph)."""
        X = self.scaler.transform(self.validate_nodes(nodes)).astype(np.float32)
        n = len(X)
        context = np.array(["not_required"] * n, dtype=object)
        e_idx = None
        if self.requires_graph:
            if edges is None:
                raise InputValidationError(
                    f"model config '{self.manifest['config']}' needs graph context: pass the snapshot's edges "
                    "(an empty DataFrame explicitly means 'no edges')")
            pos = pd.Series(np.arange(n), index=nodes["tx_id"].to_numpy())
            s = edges["source_tx_id"].map(pos)
            t = edges["target_tx_id"].map(pos)
            ok = s.notna() & t.notna()
            context[:] = "complete"
            partial_ids = set(edges.loc[~ok & s.notna(), "source_tx_id"]) | set(edges.loc[~ok & t.notna(), "target_tx_id"])
            if partial_ids:
                context[nodes["tx_id"].isin(partial_ids).to_numpy()] = "partial"
            e_idx = np.vstack([s[ok].to_numpy(np.int64), t[ok].to_numpy(np.int64)]) if ok.any() \
                else np.zeros((2, 0), dtype=np.int64)
        feats = {}

        def recipe(name):
            if name in feats:
                return feats[name]
            if name == "x165":
                feats[name] = X
            elif name == "x93":
                feats[name] = X[:, :93]
            elif name == "x165_nbr":
                from src.models.elliptic_training import neighbor_aggregate_features
                feats[name] = np.hstack([X, neighbor_aggregate_features(X[:, :93], e_idx, n)])
            elif name in ("gat_prob", "x165_emb"):
                from src.models.graph_encoder import make_edge_index
                p, emb = self.gat.predict(X, make_edge_index(e_idx, self.manifest["gat_edge_direction"]))
                feats["gat_prob"], feats["x165_emb"] = p, np.hstack([X, emb])
            return feats[name]

        probs = []
        for name, feat, model in self.components:
            if feat == "gat_prob":
                probs.append(recipe("gat_prob"))
            else:
                probs.append(model.predict_proba(recipe(feat))[:, 1])
        score = np.mean(probs, axis=0)
        return pd.DataFrame({
            "tx_id": nodes["tx_id"].to_numpy(),
            "score": score.astype(float),
            "flagged": score >= self.threshold,
            "graph_context": context,
        })


def latest_bundle_dir(root: str) -> str:
    versions = [d for d in os.listdir(root) if os.path.exists(os.path.join(root, d, "manifest.json"))]
    if not versions:
        raise FileNotFoundError(f"no exported Elliptic bundle under {root}")
    return os.path.join(root, sorted(versions, key=lambda v: tuple(int(p) for p in v.split(".")))[-1])
