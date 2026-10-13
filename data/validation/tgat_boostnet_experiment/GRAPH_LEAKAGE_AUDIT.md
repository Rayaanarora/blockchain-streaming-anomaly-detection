# Graph Construction, Message Passing & Temporal Leakage Audit

**Experiment:** TGAT-BoostNet Evaluation Protocol  
**Artifact:** `data/validation/tgat_boostnet_experiment/GRAPH_LEAKAGE_AUDIT.md`  
**Dataset:** Elliptic Bitcoin Transaction Benchmark  
**Protocol:** Strict Chronological Inductive Temporal Partitioning  

---

## 1. Information-Availability Policy & Evaluation Nature

### Is the Protocol Inductive or Transductive?
This evaluation is **strictly inductive temporal evaluation**:
- **Training Graph (Steps 1–29):** The model is trained exclusively on the historical subgraphs corresponding to time steps 1 through 29.
- **Validation Graph (Steps 30–34):** Model checkpoints and decision thresholds are selected on intermediate subgraphs (steps 30–34) to monitor generalization.
- **Test Graph (Steps 35–49):** The model is evaluated once on unseen future subgraphs (steps 35–49). Neither test node features, test labels, nor test graph topologies are accessible during model fitting or threshold optimization.
- **Transductive vs. Inductive Demarcation:** In a transductive graph setup, test node features and edges are present in the Laplacian during training (with test labels masked). In our **inductive setup**, the training forward pass is strictly isolated to training subgraphs. Future nodes and edges do not exist in the training computation graph.

### Temporal Graph Information-Availability Policy
In the Elliptic benchmark, time steps correspond to discrete 2-week temporal windows (Weber et al., 2019):
- **Within-Time-Step Information Policy:** All transactions and directed payment flows occurring within discrete window $t$ form an intra-step transaction subgraph $\mathcal{G}_t = (\mathcal{V}_t, \mathcal{E}_t)$. When scoring transactions at window $t$, the full intra-step graph structure of window $t$ is observed.
- **Between-Time-Step Information Policy:** No future window information ($\mathcal{G}_{t'}$, $t' > t$) is ever accessible when evaluating window $t$. Message passing is partitioned strictly by time step:
  $$\mathbf{h}_i^{(l+1)} = \text{AGGREGATE}\left(\{ \mathbf{h}_j^{(l)} : (j, i) \in \mathcal{E}_{t(i)} \}\right)$$

---

## 2. Graph Topology & Edge Safety Audit

### Total Graph Scope
- **Total Nodes in Graph:** **203,769 nodes**
- **Total Directed Edges in Graph:** **234,355 edges**
- **Discrete Time Steps:** 49 discrete temporal windows (numbered 1 to 49)

### Intra-Step vs. Cross-Step Edge Analysis
We audited every single edge $(u, v) \in \mathcal{E}$ by comparing the temporal attribute of the source transaction $t(u)$ against the target transaction $t(v)$:
```text
Total Edges Audited:                              234,355
Edges connecting nodes within the same step:      234,355  (100.00%)
Edges from future step to past step (t_dst < t_src):    0  (  0.00%)
Edges from past step to future step (t_dst > t_src):    0  (  0.00%)
```

**Key Proof:** The Elliptic graph naturally decomposes into **49 completely disjoint temporal subgraphs** $\mathcal{G}_1, \mathcal{G}_2, \dots, \mathcal{G}_{49}$. There are exactly **zero cross-timestep edges** in the real dataset. Therefore, when evaluating a node at time step $t$, message passing cannot traverse edges into past or future time steps.

---

## 3. Node Preservation & Unknown-Label Masking

### Why Unknown Nodes (`label = -1`) Must Be Preserved
Of the 203,769 total transactions, **157,205 transactions (77.1%)** have unknown ground truth labels (`label = -1`). These represent background transactions on the Bitcoin blockchain.
- **Message Passing Preservation:** In Bitcoin money-laundering schemes (peel chains, layering), illicit transactions frequently route funds through unlabelled intermediary addresses. Deleting unknown nodes from the graph would artificially destroy the graph topology, severing payment paths and invalidating degree and neighborhood statistics.
- **Loss Masking Contract:** Unknown nodes are **fully retained in the graph tensor** and participate in message-passing aggregations. However, when computing supervised loss or evaluating classification metrics, a strict boolean mask is applied:
  $$\mathcal{L}_{\text{supervised}} = \frac{1}{|\mathcal{V}_{\text{train}}^{\text{labeled}}|} \sum_{i \in \mathcal{V}_{\text{train}}^{\text{labeled}}} \ell(f(\mathbf{x}_i, \mathcal{G}), y_i), \quad \text{where } y_i \in \{0, 1\}$$
  $$\mathcal{V}^{\text{labeled}} = \{ i \in \mathcal{V} : y_i \neq -1 \}$$

---

## 4. Leakage Prevention Controls Summary

| Pipeline Operation | Permitted Population | Prohibited Population | Leakage Safeguard Implementation |
| :--- | :--- | :--- | :--- |
| **Feature Scaling** (`StandardScaler`) | Labeled Train Nodes (Steps 1–29) | Validation & Test Nodes (Steps 30–49) | `scaler.fit(X[train_mask])`; test statistics never enter normalization parameters. |
| **Class Weighting** ($w_{\text{illicit}} = 8.19\times$) | Labeled Train Nodes (Steps 1–29) | Validation & Test Nodes (Steps 30–49) | $w = N_{\text{train}}^{\text{licit}} / N_{\text{train}}^{\text{illicit}}$ calculated solely from training split labels. |
| **GAT Backpropagation** | Labeled Train Nodes (Steps 1–29) | Unknowns, Val & Test Nodes | Loss computed strictly over `train_mask_t`. |
| **Model Checkpoint Selection** | Validation Nodes (Steps 30–34) | Test Nodes (Steps 35–49) | Checkpoints selected on validation illicit F1; zero test metrics observed during training. |
| **Threshold Tuning** | Validation Predictions (Steps 30–34) | Test Predictions (Steps 35–49) | Grid search optimizes threshold on validation split; frozen threshold applied once to test set. |
| **XGBoost Classifier Fitting** | Fused Train Embeddings (Steps 1–29) | Validation & Test Embeddings | XGBoost fit strictly on training period feature vectors $\mathbf{z}_{\text{train}} \in \mathbb{R}^{26381 \times 197}$. |
