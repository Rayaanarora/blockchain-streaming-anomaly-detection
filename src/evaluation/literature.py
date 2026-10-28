"""
Published Elliptic results, transcribed from the papers (REPORTED IN PAPER - not produced by this code).

Each row was read from the paper's own table on 2026-10-28 (sources below). All use the same
temporal split as this repository: train time steps 1-34, test 35-49, illicit = positive class.
None of these numbers is used to rank models; notebook 03 shows them in a separate table next
to our reproductions. Empty cells = not reported in that table.

Caveats recorded with the rows:
* Weber et al. (2019) report a single run per model.
* Elliptic++ "EC" rows are on the original Elliptic features (the "TX" rows add 17 features that
  the original dataset does not have, so they are listed but marked as a different feature set).
* Inspection-L is an arXiv preprint (v4, 2022).
* Maganti (2026) is an arXiv preprint; its Table 5 is mean +/- std over 10 seeds.
* EvolveGCN's own paper (Pareja et al., AAAI 2020) shows Elliptic only as a bar chart, with a
  31/5/13 train/val/test split, so no exact number is transcribed from it; the EvolveGCN row
  below is Weber et al.'s Table 2 on the standard split.
"""
import pandas as pd

SOURCES = {
    "Weber2019": "M. Weber et al., 'Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional "
                 "Networks for Financial Forensics', KDD'19 Workshop on Anomaly Detection in Finance, 2019. "
                 "arXiv:1908.02591. Tables 1 and 2.",
    "EllipticPP2023": "Y. Elmougy and L. Liu, 'Demystifying Fraudulent Transactions and Illicit Nodes in the Bitcoin "
                      "Network for Financial Forensics' (Elliptic++), KDD 2023. arXiv:2306.06108. Table 7.",
    "InspectionL2022": "W. W. Lo et al., 'Inspection-L: Self-Supervised GNN Node Embeddings for Money Laundering "
                       "Detection in Bitcoin', arXiv:2203.10465 (v4), 2022. Table 3.",
    "Maganti2026": "S. Maganti, 'When Graph Structure Becomes a Liability: A Critical Re-Evaluation of Graph Neural "
                   "Networks for Bitcoin Fraud Detection under Temporal Distribution Shift', arXiv:2604.19514, 2026. "
                   "Tables 5 and 6.",
}
SPLIT = "train steps 1-34 / test steps 35-49"

# model, features, precision, recall, F1, micro-F1, AUC, source, seeds, maps_to (our reproduced key)
_ROWS = [
    ("Logistic Regression", "AF (all 166)", 0.404, 0.593, 0.481, 0.931, None, "Weber2019", "1", "lr"),
    ("Logistic Regression", "LF (local 94)", 0.348, 0.668, 0.457, 0.920, None, "Weber2019", "1", None),
    ("Logistic Regression", "AF + GCN emb.", 0.537, 0.528, 0.533, 0.945, None, "Weber2019", "1", None),
    ("Random Forest", "AF (all 166)", 0.956, 0.670, 0.788, 0.977, None, "Weber2019", "1", "rf"),
    ("Random Forest", "AF + GCN emb.", 0.971, 0.675, 0.796, 0.978, None, "Weber2019", "1", None),
    ("Random Forest", "LF (local 94)", 0.803, 0.611, 0.694, 0.966, None, "Weber2019", "1", None),
    ("MLP", "AF (all 166)", 0.694, 0.617, 0.653, 0.962, None, "Weber2019", "1", "mlp"),
    ("MLP", "AF + GCN emb.", 0.780, 0.617, 0.689, 0.967, None, "Weber2019", "1", None),
    ("GCN", "AF (all 166)", 0.812, 0.512, 0.628, 0.961, None, "Weber2019", "1", "gcn"),
    ("Skip-GCN", "AF (all 166)", 0.812, 0.623, 0.705, 0.966, None, "Weber2019", "1", "skip_gcn"),
    ("EvolveGCN", "AF (all 166)", 0.850, 0.624, 0.720, 0.968, None, "Weber2019", "1", "evolvegcn"),
    ("Logistic Regression", "Elliptic (EC)", 0.326, 0.707, 0.446, 0.886, None, "EllipticPP2023", "n/r", "lr"),
    ("Random Forest", "Elliptic (EC)", 0.940, 0.724, 0.818, 0.979, None, "EllipticPP2023", "n/r", "rf"),
    ("MLP", "Elliptic (EC)", 0.476, 0.673, 0.558, 0.931, None, "EllipticPP2023", "n/r", "mlp"),
    ("LSTM", "Elliptic (EC)", 0.665, 0.350, 0.459, 0.946, None, "EllipticPP2023", "n/r", None),
    ("XGBoost", "Elliptic (EC)", 0.812, 0.717, 0.761, 0.971, None, "EllipticPP2023", "n/r", "xgb"),
    ("RF + XGB ensemble", "Elliptic (EC)", 0.960, 0.704, 0.812, 0.979, None, "EllipticPP2023", "n/r", None),
    ("RF + MLP + XGB ensemble", "Elliptic (EC)", 0.947, 0.719, 0.817, 0.979, None, "EllipticPP2023", "n/r", None),
    ("Random Forest", "Elliptic++ (TX, +17 feats)", 0.975, 0.719, 0.828, 0.980, None, "EllipticPP2023", "n/r", None),
    ("Inspection-L (DGI emb. + RF)", "AF + DGI emb.", 0.972, 0.721, 0.828, None, 0.916, "InspectionL2022", "n/r", None),
    ("Inspection-L (DGI emb. + RF)", "LF + DGI emb.", 0.906, 0.712, 0.797, None, 0.895, "InspectionL2022", "n/r", None),
    ("3-layer MLP", "raw 165", 0.496, 0.616, 0.549, None, 0.889, "Maganti2026", "10 (mean)", None),
    ("GCN", "raw 165", 0.567, 0.456, 0.503, None, 0.882, "Maganti2026", "10 (mean)", "gcn"),
    ("GraphSAGE", "raw 165", 0.709, 0.670, 0.688, None, 0.928, "Maganti2026", "10 (mean)", "graphsage"),
    ("GAT", "raw 165", 0.588, 0.635, 0.610, None, 0.909, "Maganti2026", "10 (mean)", "gat"),
    ("Logistic Regression", "raw 165", None, None, 0.53, None, 0.88, "Maganti2026", "n/r", "lr"),
    ("XGBoost", "raw 165", None, None, 0.775, None, 0.931, "Maganti2026", "n/r", "xgb"),
    ("Random Forest", "raw 165", None, None, 0.821, None, 0.934, "Maganti2026", "10", "rf"),
    ("RF on [GraphSAGE emb. || raw]", "graph + raw", None, None, 0.699, None, 0.892, "Maganti2026", "n/r", None),
]


def reported_table() -> pd.DataFrame:
    df = pd.DataFrame(_ROWS, columns=["model", "features", "precision", "recall", "f1_illicit", "micro_f1", "auc",
                                      "source", "seeds", "maps_to"])
    df["year"] = df.source.str.extract(r"(\d{4})").astype(int)
    df["split"] = SPLIT
    df["status"] = "reported in paper"
    return df
