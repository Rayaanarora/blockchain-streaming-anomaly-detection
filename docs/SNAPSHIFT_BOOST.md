# SnapShift-Boost

The project's own Elliptic model (bundle **2.0.0**, config `snapshift_P`). This page covers the design, how the final variant was chosen, the test result and what the model does not do. Every number below comes from files under `data/validation/snapshift/`.

## 1. Idea

The previous model (`hgb_165_nbr`, bundle 1.0.1) classifies each transaction from its own features and the *average features* of its graph neighbours. SnapShift-Boost adds a second stage that uses what the model *predicts* about the neighbours:

```
complete time-step snapshot (nodes + edges)
        │
        ▼
 stage 1: HistGBDT on 165 features + neighbour aggregates (degree, mean of f1–f93)
        │   p1 = illicit probability for every node in the snapshot
        ▼
 score propagation: for in-, out- and undirected neighbours, mean and max of p1;
                    undirected 2-hop mean of p1; the node's own p1          (8 features)
        │
        ▼
 stage 2: HistGBDT on stage-1 features + the 8 propagation features  →  illicit probability
```

Illicit Elliptic transactions cluster in the transaction graph. Averaging neighbour *features* only partly captures that. A neighbour that stage 1 already considers suspicious is more direct evidence.

**Leakage guard (cross-fitting).** If stage 2 were trained on stage-1 probabilities that stage 1 fitted on the same labels, those probabilities would be overconfident on training rows, and stage 2 would learn to over-trust them. This is the same defect that inflated the earlier TGAT-BoostNet result. Training rows therefore get **out-of-fold** stage-1 probabilities. The training time steps are split into 4 contiguous groups, and each group is scored by a stage-1 model fitted on the other three. Every other node (validation, test, unknown, and all nodes at inference time) is scored by the stage-1 model fitted on the full training set. `tests/test_snapshift.py::test_stage1_training_scores_are_out_of_fold` checks this.

**No label reads outside training.** The propagation features use predicted probabilities only, never labels. `test_labels_outside_training_rows_are_never_read` scrambles every non-training label and checks that all scores are bit-identical.

**Streaming compatibility.** Every feature is a function of one complete time-step snapshot. Elliptic edges never cross steps, and the streaming consumer already buffers complete snapshots (DECISIONS D-14). Sums are accumulated in float64 and maxima are exact, so scores do not depend on node order (D-18). This is checked by `test_two_stage_bundle_reproduces_fit_per_snapshot_and_is_order_invariant` and by the stream-vs-batch integration test.

## 2. Components tried, and how the final one was chosen

Three components were designed. Every combination was compared against the baseline:

| Code | Component | Rationale |
|---|---|---|
| R | Percentile rank of each raw feature within its own time step | "Unusual for this step" might survive population drift better than raw values |
| I | Isolation Forest anomaly score fitted on each time step (unsupervised; nodes sorted by `tx_id` so it is order invariant) | Links the supervised model with the anomaly-detection approach used for Zenodo |
| P | Stage-2 score propagation (above) | Neighbourhood suspicion |

**Protocol (`scripts/snapshift_dev.py`).** Test steps 35–49 are removed from the data before anything else runs. Three rolling-origin folds within steps 1–34:

| Fold | Train steps | Validate steps |
|---|---|---|
| 1 | 1–19 | 20–24 |
| 2 | 1–24 | 25–29 |
| 3 | 1–29 | 30–34 |

Selection rule, written into the script before the first run:
- Pick the highest mean validation F1 over the three folds.
- A variant must beat the baseline by at least 0.005 to replace it.
- Within 0.005 of the best, fewer components wins, then higher mean AP.

HistGBDT without early stopping is deterministic, so one seed was used. A second seed was later confirmed to give bit-identical scores.

**Results (`data/validation/snapshift/dev/summary.json`)**

| Variant | Mean val F1 | Mean val AP | Fold 1 | Fold 2 | Fold 3 |
|---|---|---|---|---|---|
| Baseline (`hgb_165_nbr`) | 0.9305 | 0.9529 | 0.8503 | 0.9669 | 0.9743 |
| R | 0.9240 | 0.9488 | 0.8420 | 0.9615 | 0.9685 |
| I | 0.9310 | 0.9538 | 0.8511 | 0.9690 | 0.9728 |
| **P** | **0.9364** | **0.9554** | 0.8556 | 0.9740 | 0.9796 |
| RI | 0.9243 | 0.9490 | 0.8401 | 0.9627 | 0.9701 |
| RP | 0.9349 | 0.9526 | 0.8533 | 0.9711 | 0.9804 |
| IP | 0.9364 | 0.9542 | 0.8525 | 0.9753 | 0.9813 |
| RIP | 0.9352 | 0.9525 | 0.8567 | 0.9701 | 0.9787 |

**What the folds showed:**
- **Propagation (P) is the component that helps.** P alone beats the baseline on all three folds, and every variant containing P beats the baseline.
- **Within-step ranks (R) hurt on every fold.** Rank-transforming loses the absolute magnitudes the trees rely on.
- **The per-step anomaly score (I) is neutral.**

**Selected: P.** It is +0.0059 over the baseline. IP ties it on mean F1, and the rule prefers fewer components.

## 3. Test result (single read)

`scripts/snapshift_final.py` → `data/validation/snapshift/final/seed_42.json`. Train on labelled steps 1–29, set the threshold to the F1 optimum on steps 30–34 (it came out at 0.90), then evaluate once on steps 35–49 (16,670 labelled transactions). The baseline is compared on the same transactions with its own validation threshold.

| | Baseline `hgb_165_nbr` (1.0.1) | **SnapShift-Boost (2.0.0)** |
|---|---|---|
| Illicit F1 | 0.7968 | **0.7994** |
| Precision | 0.8831 | **0.8883** |
| Recall | 0.7258 | **0.7267** |
| Average precision | 0.7977 | **0.8004** |
| ROC-AUC | **0.9336** | 0.9315 |
| TN / FP / FN / TP | 15,483 / 104 / 297 / 786 | 15,488 / 99 / 296 / 787 |
| Recall, steps 35–42 | 0.854 | 0.857 |
| Recall, steps 43–49 | 0.030 | 0.024 |

**Paired bootstrap of the F1 difference** (2,000 resamples):

| Resampling unit | Point estimate | 95% interval | Share of resamples > 0 |
|---|---|---|---|
| Transactions | +0.0026 | [−0.0048, +0.0107] | 76% |
| Time steps | +0.0026 | [−0.0050, +0.0104] | 76% |

**How to describe this result:**
- SnapShift-Boost is better than the baseline on F1, precision, recall and AP.
- The improvement is small and **not statistically significant**: zero lies inside the 95% interval.
- The validation gain (+0.006) shrank on test (+0.003), as expected for a selected model.
- It does not address the post-step-43 failure. Recall after step 43 stays near zero, because the illicit transactions there no longer resemble training-period illicit ones (VERIFIED_RESULTS §1.4), so neighbourhood suspicion has nothing to propagate.

Nothing was tuned after the test read. The first run of `snapshift_final.py` crashed on a wrong manifest key *after* saving its predictions. The rerun reused those predictions; the model and threshold were unchanged.

## 4. Export and verification

`scripts/export_snapshift_bundle.py` → `artifacts/models/elliptic/2.0.0` (scaler, stage-1 model, stage-2 model, manifest with `combine: two_stage`, SHA-256 per file). The script checks the following and refuses to export otherwise (`data/validation/snapshift/final/bundle_checks.json`):

1. The refit reproduces the evaluated probabilities exactly (max |diff| 0.0).
2. A different seed gives identical probabilities on all 203,769 nodes (max |diff| 0.0).
3. The bundle, scoring one complete snapshot at a time as the streaming consumer does, reproduces the evaluated test scores exactly (max |diff| 0.0, test F1 0.7994).

The scoring service loads the highest bundle version, so the streaming pipeline, `/api/score` and the dashboard use 2.0.0. Verified end to end on 2026-10-15: unit suite 96/96; Kafka + Spark integration tests 2/2 with 2.0.0 loaded, including streamed = batch scores; inference ≈63 ms per ~5,000-node snapshot. Set `BSAD_ELLIPTIC_BUNDLE` to `artifacts\models\elliptic\1.0.1` to run the baseline instead.

## 5. Novelty and related work

The contribution is a **design for this problem**, not a new learning algorithm:
- Two-stage score propagation over a transaction graph.
- Leakage-free training via time-grouped cross-fitting.
- Per-snapshot computation that is exactly order invariant, so the same model runs unchanged in a Kafka/Spark stream with stream-to-batch score parity.

Its closest relatives are stacked or collective classification on graphs, where a second classifier uses neighbours' predicted labels, and the GCN/EvolveGCN-style use of neighbourhood information on Elliptic. Compared with the project's earlier GAT stacking (TGAT-BoostNet), it removes the stacking leakage and needs no neural network. Without the leakage, the GAT variants scored below the tree baseline (VERIFIED_RESULTS §1.1).

## 6. Limitations

- **Gain not significant:** the test gain over the baseline is +0.003 F1, within noise (§3).
- **No recovery after step 43:** recall after step 43 is ≈0.02–0.03, the same as every other model tested.
- **Threshold sensitivity:** the threshold (0.90) is set on a small, near-saturated validation set, as for every Elliptic model here (VERIFIED_RESULTS §1.3).
- **Higher inference cost than the baseline:** two tree models per snapshot plus sparse neighbour statistics.
- **Not temporal:** each snapshot is scored independently.
