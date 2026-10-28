"""
Elliptic data access and the evaluation protocol shared by notebooks 01-03.

Protocol (fixed before any test number was looked at):
  * Standard temporal split of Weber et al. (2019): train = labelled steps 1-34,
    test = labelled steps 35-49. Illicit (class "1") is the positive class.
  * No validation-based model selection: every model uses hyper-parameters fixed
    a priori (library defaults or the published configuration) and the decision
    threshold 0.5. Test labels are only read to compute metrics.
  * Unknown-label nodes stay in the graph for message passing but never enter a loss
    or a metric.
  * Features are standardised with mean/std of the labelled training rows only.

The loader reads the three raw Kaggle CSVs with pandas, so it works the same in
Colab and locally. Notebook 01 shows that its raw features are bit-identical to the
Spark pipeline output (data/processed/elliptic_spark).
"""
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RAW_FILES = ("elliptic_txs_features.csv", "elliptic_txs_classes.csv", "elliptic_txs_edgelist.csv")
# Mirror already used by scripts/download_elliptic.py (no credentials needed).
HF_MIRROR = "https://huggingface.co/datasets/yhoma/elliptic-bitcoin-dataset/resolve/main"
KAGGLE_DATASET = "ellipticco/elliptic-data-set"

TRAIN_STEPS: Tuple[int, int] = (1, 34)
TEST_STEPS: Tuple[int, int] = (35, 49)
SEEDS = (42, 123, 456, 789, 2024)
N_LOCAL = 93  # f1..f93 local features (time step excluded); f94..f165 aggregated


def _has_raw(d: str) -> bool:
    return all(os.path.exists(os.path.join(d, f)) for f in RAW_FILES)


def find_or_download_elliptic(data_dir: Optional[str] = None, verbose: bool = True) -> str:
    """Return a directory containing the three Elliptic CSVs.

    Search order: ``data_dir``; $ELLIPTIC_DIR; <repo>/data/raw/elliptic; /content/elliptic;
    Google Drive (/content/drive/MyDrive/elliptic). If none has the files, try the Kaggle API
    (needs ~/.kaggle/kaggle.json), then the public Hugging Face mirror.
    """
    candidates = [data_dir, os.environ.get("ELLIPTIC_DIR"), os.path.join(REPO_ROOT, "data", "raw", "elliptic"),
                  "/content/elliptic", "/content/drive/MyDrive/elliptic",
                  "/content/drive/MyDrive/elliptic_bitcoin_dataset"]
    for c in candidates:
        if c and _has_raw(c):
            if verbose:
                print("Elliptic CSVs found in", c)
            return c
    target = data_dir or os.environ.get("ELLIPTIC_DIR") or os.path.join(REPO_ROOT, "data", "raw", "elliptic")
    os.makedirs(target, exist_ok=True)
    try:  # 1) Kaggle API
        import kaggle  # noqa: F401  (authenticates on import; raises without kaggle.json)
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(KAGGLE_DATASET, path=target, unzip=True)
        for root, _, files in os.walk(target):  # Kaggle nests the files one level down
            for f in files:
                if f in RAW_FILES and root != target:
                    os.replace(os.path.join(root, f), os.path.join(target, f))
        if _has_raw(target):
            if verbose:
                print("downloaded from Kaggle to", target)
            return target
    except Exception as e:  # noqa: BLE001
        if verbose:
            print("Kaggle download unavailable:", type(e).__name__, str(e)[:120])
    for f in RAW_FILES:  # 2) public mirror
        dest = os.path.join(target, f)
        if not os.path.exists(dest):
            if verbose:
                print("downloading", f, "from", HF_MIRROR)
            urllib.request.urlretrieve(f"{HF_MIRROR}/{f}", dest + ".part")
            os.replace(dest + ".part", dest)
    if not _has_raw(target):
        raise FileNotFoundError("could not obtain the Elliptic CSVs; see the notebook's data cell")
    return target


@dataclass
class Elliptic:
    """Raw Elliptic graph in a fixed node order (sorted by time step, then tx id)."""
    tx_ids: np.ndarray
    time_steps: np.ndarray
    X_raw: np.ndarray            # (N, 165) float32, the anonymised features (time step excluded)
    y: np.ndarray                # 1 illicit, 0 licit, -1 unknown
    edge_index: np.ndarray       # (2, E) directed source -> target node positions
    feature_names: list = field(default_factory=lambda: [f"f{i}" for i in range(1, 166)])

    @property
    def n(self) -> int:
        return len(self.y)

    def mask(self, steps: Tuple[int, int], labelled: bool = True) -> np.ndarray:
        m = (self.time_steps >= steps[0]) & (self.time_steps <= steps[1])
        return m & (self.y >= 0) if labelled else m

    @property
    def train_mask(self) -> np.ndarray:
        return self.mask(TRAIN_STEPS)

    @property
    def test_mask(self) -> np.ndarray:
        return self.mask(TEST_STEPS)


def load_elliptic(data_dir: Optional[str] = None, cache: bool = True, verbose: bool = True) -> Elliptic:
    d = find_or_download_elliptic(data_dir, verbose=verbose)
    cache_path = os.path.join(d, "elliptic_eval_cache.npz")
    if cache and os.path.exists(cache_path):
        z = np.load(cache_path)
        return Elliptic(z["tx_ids"], z["time_steps"], z["X_raw"], z["y"], z["edge_index"])
    feats = pd.read_csv(os.path.join(d, RAW_FILES[0]), header=None)
    classes = pd.read_csv(os.path.join(d, RAW_FILES[1]))
    edges = pd.read_csv(os.path.join(d, RAW_FILES[2]))
    feats.columns = ["tx_id", "time_step"] + [f"f{i}" for i in range(1, 166)]
    classes.columns = ["tx_id", "class"]
    lab = classes["class"].astype(str).map({"1": 1, "2": 0}).fillna(-1).astype(int)
    feats = feats.merge(pd.DataFrame({"tx_id": classes["tx_id"], "y": lab}), on="tx_id", how="left")
    feats["y"] = feats["y"].fillna(-1).astype(int)
    feats = feats.sort_values(["time_step", "tx_id"]).reset_index(drop=True)
    pos = pd.Series(np.arange(len(feats)), index=feats["tx_id"].to_numpy())
    edges.columns = ["src", "dst"]
    ei = np.vstack([pos.loc[edges["src"]].to_numpy(), pos.loc[edges["dst"]].to_numpy()]).astype(np.int64)
    out = Elliptic(feats["tx_id"].to_numpy(np.int64), feats["time_step"].to_numpy(np.int64),
                   feats[[f"f{i}" for i in range(1, 166)]].to_numpy(np.float32), feats["y"].to_numpy(np.int64), ei)
    if cache:
        np.savez_compressed(cache_path, tx_ids=out.tx_ids, time_steps=out.time_steps, X_raw=out.X_raw,
                            y=out.y, edge_index=out.edge_index)
    return out


def standardise(data: Elliptic, X: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardise with mean/std (population) of labelled training rows only (sklearn convention)."""
    X = data.X_raw if X is None else X
    tr = data.train_mask
    mu = X[tr].astype(np.float64).mean(0)
    sd = X[tr].astype(np.float64).std(0)
    sd[sd == 0] = 1.0
    return ((X - mu) / sd).astype(np.float32), mu, sd


def bidirectional(edge_index: np.ndarray) -> np.ndarray:
    return np.concatenate([edge_index, edge_index[::-1]], axis=1)


def induced(node_mask: np.ndarray, edge_index: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(node positions, re-indexed edges) of the subgraph induced by ``node_mask``.

    Elliptic edges never cross time steps, so the subgraph of any set of whole time steps
    loses no edge and message passing on it is exactly equal to full-graph message passing.
    """
    idx = np.flatnonzero(node_mask)
    remap = -np.ones(len(node_mask), dtype=np.int64)
    remap[idx] = np.arange(len(idx))
    keep = node_mask[edge_index[0]] & node_mask[edge_index[1]]
    return idx, remap[edge_index[:, keep]]


def cross_step_edges(data: Elliptic) -> int:
    s, t = data.edge_index
    return int((data.time_steps[s] != data.time_steps[t]).sum())
