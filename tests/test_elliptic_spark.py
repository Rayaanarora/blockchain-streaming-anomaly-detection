"""
Elliptic Spark pipeline tests.

* A tiny synthetic Elliptic fixture is run through the real Spark pipeline (fast, self-contained):
  ids, ordering, labels, unknown masking, splits, edge mapping, train-only scaling, refusal of bad input.
* If the full Spark output exists, the stored parity report must say PASS.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

FULL_REPORT = os.path.join(ROOT, "data/validation/elliptic_spark/parity_report.json")


def write_fixture(d, steps=(1, 2, 30, 35), per_step=4, bad=None):
    rng = np.random.default_rng(0)
    rows, cls, edges = [], [], []
    tx = 100
    for t in steps:
        ids = []
        for _ in range(per_step):
            tx += 7
            ids.append(tx)
            rows.append([tx, t] + list(rng.normal(size=165)))
            cls.append((tx, ["1", "2", "unknown"][tx % 3]))
        edges += [(ids[i], ids[i + 1]) for i in range(per_step - 1)]
    rng.shuffle(rows)  # input order must not matter
    if bad == "cross_step":
        edges.append((rows[0][0], [r for r in rows if r[1] != rows[0][1]][0][0]))
    if bad == "dangling":
        edges.append((rows[0][0], 999999))
    pd.DataFrame(rows).to_csv(os.path.join(d, "elliptic_txs_features.csv"), header=False, index=False)
    pd.DataFrame(cls, columns=["txId", "class"]).to_csv(os.path.join(d, "elliptic_txs_classes.csv"), index=False)
    pd.DataFrame(edges, columns=["txId1", "txId2"]).to_csv(os.path.join(d, "elliptic_txs_edgelist.csv"), index=False)
    return rows, cls, edges


@pytest.fixture(scope="module")
def spark():
    from src.spark.session import get_spark
    s = get_spark("TestEllipticSpark", master="local[2]",
                  scratch_dir=os.path.join(ROOT, "data/test_spark_scratch"))
    yield s
    s.stop()


def test_pipeline_on_fixture(spark, tmp_path):
    from src.spark.elliptic_spark import run
    raw = tmp_path / "raw"
    raw.mkdir()
    rows, cls, edges = write_fixture(str(raw))
    out = tmp_path / "out"
    res = run(spark, str(raw), str(out), expected_time_steps=None)
    nodes = pd.read_parquet(out / "nodes").sort_values("node_idx")
    nodes["time_step"] = nodes["time_step"].astype(int)
    e = pd.read_parquet(out / "edges").sort_values("edge_id")
    assert len(nodes) == len(rows) and len(e) == len(edges)
    assert list(nodes.node_idx) == list(range(len(rows)))
    assert list(zip(nodes.time_step, nodes.tx_id)) == sorted((r[1], r[0]) for r in rows)
    lab = dict((t, {"1": 1, "2": 0, "unknown": -1}[c]) for t, c in cls)
    assert list(nodes.label) == [lab[t] for t in nodes.tx_id]
    assert (nodes.loc[nodes.label == -1, "split"] == "unlabelled").all()
    exp = np.where(nodes.label == -1, "unlabelled", np.where(nodes.time_step <= 29, "train",
                                                             np.where(nodes.time_step <= 34, "val", "test")))
    assert (nodes.split.to_numpy() == exp).all()
    pos = dict(zip(nodes.tx_id, nodes.node_idx))
    assert list(e.source_tx_id) == [a for a, _ in edges] and list(e.target_tx_id) == [b for _, b in edges]
    assert list(e.source_idx) == [pos[a] for a, _ in edges]
    # train-only scaling: mean of scaled train rows == 0
    sc = pd.read_parquet(out / "nodes_scaled")
    tr = sc[sc.split == "train"]
    assert np.allclose(tr[[f"f{i}" for i in range(1, 166)]].mean().to_numpy(), 0, atol=1e-5)
    assert not np.allclose(sc[sc.split == "test"][["f1", "f2"]].mean().to_numpy(), 0, atol=1e-3)
    assert res["report"]["integrity"]["cross_time_step_edges"] == 0


@pytest.mark.parametrize("bad", ["cross_step", "dangling"])
def test_pipeline_refuses_invalid_graph(spark, tmp_path, bad):
    from src.spark.elliptic_spark import EllipticValidationError, run
    raw = tmp_path / "raw"
    raw.mkdir()
    write_fixture(str(raw), bad=bad)
    with pytest.raises(EllipticValidationError):
        run(spark, str(raw), str(tmp_path / "out"), expected_time_steps=None)
    assert not (tmp_path / "out").exists()


def test_pipeline_does_not_overwrite_without_flag(spark, tmp_path):
    from src.spark.elliptic_spark import run
    raw = tmp_path / "raw"
    raw.mkdir()
    write_fixture(str(raw))
    (tmp_path / "out").mkdir()
    with pytest.raises(FileExistsError):
        run(spark, str(raw), str(tmp_path / "out"), expected_time_steps=None)


def test_pipeline_enforces_real_time_step_range(spark, tmp_path):
    from src.spark.elliptic_spark import EllipticValidationError, run
    raw = tmp_path / "raw"
    raw.mkdir()
    write_fixture(str(raw))
    with pytest.raises(EllipticValidationError, match="time-step range"):
        run(spark, str(raw), str(tmp_path / "out"))


@pytest.mark.skipif(not os.path.exists(FULL_REPORT), reason="full Elliptic Spark output not produced")
def test_full_parity_report_passes():
    r = json.load(open(FULL_REPORT))
    assert r["status"] == "PASS" and r["n_failed"] == 0 and r["n_checks"] >= 30
