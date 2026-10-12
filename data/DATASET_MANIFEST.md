# Dataset Acquisition Manifest
**Project:** Distributed Streaming Anomaly Detection on Blockchain Transaction Graphs  
**Last Updated:** 2026-10-13 00:36:13  
**Zenodo Source:** [Record 7157854 (Part 2 of 4)](https://zenodo.org/records/7157854)

## Verified Archives & Ingestion Status

| Canonical Filename | Local File Path | Range / Part | Expected Size | Actual Size | Expected MD5 | Actual MD5 | Status | Acquisition Date |
|---|---|---|---|---|---|---|---|---|
| `btc-tx-480000-489999-aa.tar.bz2` | `data/raw/a.tar.bz2` | 480000-489999 (aa) | 208,068,011 B | 208,068,011 B | `52cf128ea182d048584140978907eff7` | `52cf128ea182d048584140978907eff7` | **VERIFIED_PASS** | Prior to Phase A |
| `btc-tx-480000-489999-ab.tar.bz2` | `data/raw/btc-tx-480000-489999-ab.tar.bz2` | 480000-489999 (ab) | 210,954,577 B | 210,954,577 B | `fc9995159184b140f0cad2571c02c27d` | `fc9995159184b140f0cad2571c02c27d` | **VERIFIED_PASS** | 2026-10-13 |
| `btc-tx-480000-489999-ac.tar.bz2` | `data/raw/btc-tx-480000-489999-ac.tar.bz2` | 480000-489999 (ac) | 206,665,357 B | 206,665,357 B | `ff240578a3f7ee2114ff771de7a43558` | `ff240578a3f7ee2114ff771de7a43558` | **VERIFIED_PASS** | 2026-10-13 |
| `btc-tx-480000-489999-ad.tar.bz2` | `data/raw/btc-tx-480000-489999-ad.tar.bz2` | 480000-489999 (ad) | 215,577,710 B | 215,577,710 B | `c7a2f1b712d1575910a296bf46b31601` | `c7a2f1b712d1575910a296bf46b31601` | **VERIFIED_PASS** | 2026-10-13 |
| `btc-tx-490000-499999-aa.tar.bz2` | `data/raw/b.tar.bz2` | 490000-499999 (aa) | 212,857,442 B | 212,857,442 B | `fd430a63ea5d2d89b494d9c9d2983073` | `fd430a63ea5d2d89b494d9c9d2983073` | **VERIFIED_PASS** | Prior to Phase A |
| `BitcoinHeistData.csv` | `data/raw/heist_extracted/BitcoinHeistData.csv` | Years 2011-2018 (full) | 235,882,597 B | 235,882,597 B | `N/A (UCI Repository)` | `Verified intact CSV` | **VERIFIED_PASS** | Prior to Phase A |

## Notes & Ingestion Policies
1. **Uncompressed Sample Preservation:** `data/raw/sample_a/btc-tx-480000-489999-aa` (604.44 MB) and `data/raw/a.tar` are preserved intact.
2. **Batch 1 Archives (`ac`, `ad`):** Successfully acquired in compressed form (`.tar.bz2`). Unextracted as per Phase B instructions.
3. **Contiguous Range 480k-490k Progress:** Parts `aa`, `ab`, `ac`, and `ad` are now present and verified on disk.
