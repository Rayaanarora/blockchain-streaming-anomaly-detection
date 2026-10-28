# %% [markdown]
# ## 0. Setup (Colab or local)
# This cell clones the repository when run in Colab (fresh runtime), installs the dependencies, makes `src/`
# importable, fixes seeds and detects a GPU. Locally it only locates the repository root.
#
# **Data.** The Elliptic CSVs are looked up in `data/raw/elliptic`, `/content/elliptic` and Google Drive
# (`MyDrive/elliptic`). If they are missing, the loader tries the Kaggle API (upload `kaggle.json` first:
# `!mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json`; dataset
# `ellipticco/elliptic-data-set`), then falls back to a public mirror of the same three files.
# %%
import os, sys, subprocess, random, time, json, warnings
warnings.filterwarnings("ignore")
IN_COLAB = "google.colab" in sys.modules
REPO_URL = "https://github.com/Rayaanarora/blockchain-streaming-anomaly-detection.git"
REPO_BRANCH = "eval-notebooks"   # change to "main" once merged
MOUNT_DRIVE = False          # set True to read the CSVs from Google Drive (MyDrive/elliptic)
USE_CACHED_RESULTS = True    # True: reuse results/ shipped with the repo; False: retrain everything
QUICK = os.environ.get("BSAD_QUICK") == "1"   # smoke-test mode (tiny epochs, separate cache)

if IN_COLAB:
    ROOT = "/content/blockchain-streaming-anomaly-detection"
    if not os.path.exists(ROOT):
        subprocess.run(["git", "clone", "--depth", "1", "--branch", REPO_BRANCH, REPO_URL, ROOT], check=True)
    os.chdir(ROOT)
    import torch
    pkgs = ["torch_geometric", "xgboost>=2", "lightgbm", "catboost", "umap-learn", "seaborn", "plotly",
            "pyspark==3.5.0", "psutil", "kaggle"]
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *pkgs], check=True)
    if MOUNT_DRIVE:
        from google.colab import drive
        drive.mount("/content/drive")
else:
    ROOT = os.path.abspath(os.getcwd())
    while not os.path.exists(os.path.join(ROOT, "src", "evaluation")) and os.path.dirname(ROOT) != ROOT:
        ROOT = os.path.dirname(ROOT)
    os.chdir(ROOT)
sys.path.insert(0, ROOT)
if not USE_CACHED_RESULTS:   # recompute into a fresh cache (notebook 03 reuses what notebook 02 computed)
    os.environ["BSAD_RESULTS_DIR"] = os.path.join(ROOT, "results_fresh")
if QUICK:
    os.environ["BSAD_RESULTS_DIR"] = os.path.join(ROOT, "results_quick")
    os.environ["BSAD_FIG_DIR"] = os.path.join(ROOT, "results_quick", "figures")
    os.environ["BSAD_TAB_DIR"] = os.path.join(ROOT, "results_quick", "tables")

import numpy as np, pandas as pd, torch
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
from src.evaluation.tgat_boostnet import get_device
DEVICE = get_device()
print(f"repo={ROOT}  colab={IN_COLAB}  device={DEVICE}  torch={torch.__version__}  quick={QUICK}")
if DEVICE == "cuda":
    print(torch.cuda.get_device_name(0))
from src.evaluation import plotting as PL
PL.set_style()
pd.set_option("display.width", 200, "display.max_columns", 40, "display.precision", 4)
