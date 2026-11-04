# Reproducing the numbers in the paper

This file maps each table, figure and quoted number in the paper to the code that produces it. There are three kinds of source:

* **Live notebook.** The number was printed by a live run on a GPU. Re-running the notebook reproduces it, and the request log in `data/live/` lets you recompute most of it offline with the frozen library in `lib/frozen/`.
* **Offline script.** Run from its own folder, it reads the logs in `data/live/` or the rehearsal folders in `data/rehearsal/`.
* **Production study.** Scripts in `experiments/07_production/`, run from that folder, read the Chutes windows in `data/`. All of their outputs are committed under `experiments/07_production/data/`.

Scripts are listed in the order they are run. Paths are relative to the repository root.

## Section III. Setting and Table I (populations)

The populations are defined in `experiments/0*/driver*.py`. Pools P1 to P9 and C are in the Baseline, Repeat and Stress drivers, E0 to E11 in the Evasion driver (`05_stress_evasion/driver8.py`), and F1 to F9 in `06_two_replica/driver9.py`.

The content columns of Table VIII come from the companion paper (Khilar, "The limits of query monitoring against distributed model extraction"). Their code is in that paper's repository.

## Section V. Signatures and theory

| Claim | Source |
|---|---|
| R = (1-p)/(1-p/M); R = 1 for an open loop | `experiments/00_foundations/cells.py` (unit tests against simulation) |
| Classifier thresholds fixed in simulation | `simulation/test_fplib.py`, `simulation/gens.py`, `simulation/dispatch.py` |
| Worker count: sampling error ±0.03–0.8 workers, ±1.3 with jitter, bias up to 0.4 | `posthoc/worker_count_intervals.py` |
| Discoverability example: 3.4 h at 10^5 keys, 21 h for the jittered ring | arithmetic from Proposition 4 with κ = 1.9 and τ_c = 1.56 s (see "Not scripted" below) |
| Watermark visibility (AUC 1.00 / 0.55; final gap 29.6 vs 28.1 ms) and cost (+76 ms; 5.69→5.76 s; 28.47→28.57 s) | Stress and Evasion notebook, `experiments/05_stress_evasion/stress_evasion.ipynb` |
| Pinned 0.239 vs rotating 0.30 requests/s | Two-replica notebook |
| Quota margins 26%, 9%, 3.7% | analytic (cost corollary, Appendix A) |

## Section VI. Forensic expansion and Table II

| Column or claim | Source |
|---|---|
| Repeat run, seed share 10/20/30%, and precision | `experiments/04_repeat/analysis/seed_pr2.py` |
| Repeat run, 10 / 20 min | `experiments/04_repeat/analysis/window_seed.py` |
| "A seed that was half impure cost 0.05–0.15 of AUC" | `experiments/04_repeat/analysis/seed_pr.py` |
| Two-replica column (frozen prediction M8) | Two-replica notebook (`lib/replica_an.py: seed_expansion`) |

## Section VII. Site-wide discovery

The method is implemented in `lib/fplib.py` as `spectral_rings*` and `site_handoff_graph*`. Results are listed under Section XI below.

## Sections VIII and IX. Watermark and limits

Proposition 5 is implemented in `lib/fplib.py` as `watermark_*`. The design simulations are `simulation/watermark.py` and `simulation/watermark2.py`.

## Section X. Testbed and Table III (runs)

The testbed is in `experiments/0*/` (`gateway*.py`, `driver*.py`, `simserver.py`). The Stress-run facts (KV usage 0.999, 48,789 preemptions, 65 queued requests) are printed by the Stress and Evasion notebook.

## Section XI. Results

| Claim | Source |
|---|---|
| Pilot: R = 0.412 vs 0.41 (z = 22.7); halves R20 = 2.58 (z = −59); event-loop artefacts | `experiments/02_pilot/pilot.ipynb` |
| Table IV, Repeat columns | `experiments/04_repeat/repeat.ipynb` (frozen library `lib/frozen/fplib_v0.4_repeat.py`) |
| Table IV, Stress columns | `experiments/05_stress_evasion/stress_evasion.ipynb`, phase A (`lib/frozen/fplib_v0.6_stress_evasion.py`, `lib/frozen/stress_evasion_an.py`) |
| Fig. 1 (correlation heatmaps) | `cd figures && python make_baseline_figures.py` |
| Table V, provider-side balancing, unseen dispatchers, failover after 429 | `experiments/06_two_replica/two_replica.ipynb` (`lib/frozen/fplib_v0.8_two_replica.py`, `lib/replica_an.py`) |
| "Ring recovered at 40/40 every time; a human with 2 of 10 seeds" | `cd experiments/06_two_replica/analysis && python surrogate_seeds.py load_balanced.py` (then `load_affinity.py`) |
| Theory on live data: θ1 6.83/6.91, 4.58/4.67, 2.73/2.79 | `experiments/03_baseline/analysis/theory_checks.py` |
| Theory on live data: RMSE 0.015–0.020 | `experiments/03_baseline/analysis/theory_checks2.py` |
| Theory on live data: harmonics (2.88, 2.34), (4.86, 2.30), (7.09, 1.81) | `experiments/04_repeat/analysis/split_ring.py`, then `offline_misc.py` |
| Fig. 2 (ring profile) | `cd figures && python make_baseline_figures.py` |
| Site-wide, Repeat / Stress / Evasion | the respective notebooks |
| Bulk edge 2.6 → 4.6; "v0.6 found only one" | `experiments/05_stress_evasion/analysis/bulk_edge_by_scale.py` |
| Fig. 3 (spectrum) | `cd experiments/04_repeat/analysis && python split_ring.py`, then `cd figures && python make_repeat_figures.py` |
| Table VI (watermark) | Repeat notebook; Stress and Evasion notebook |
| "about 85% of follow-up requests return to the same key" | `experiments/04_repeat/analysis/handoff_prob.py` |
| Random-relay echo diagnosis | `experiments/04_repeat/analysis/relay_diag.py`, `relay_diag2.py` |
| Table VII (evasion sweep) | Stress and Evasion notebook, phase B |
| Server cross-talk (Spearman 0.28 / 0.80; start-count ratios) | `experiments/05_stress_evasion/analysis/crosstalk_stress.py`, `crosstalk_evasion.py` (Spearman −0.08 is from the Pilot notebook) |
| WildChat study: R60 0.982 ± 0.037 etc.; heaviest 3.8%; 1-hour windows ~30% | `experiments/01_wildchat/wildchat_baseline.ipynb` |
| WildChat study: calibrated scan of 5,528 groups; 2.3× nominal; no BH survivor | `experiments/01_wildchat/wildchat_calibrated_scan.ipynb` |

## Section XII. Production study (Chutes trace)

Run these from `experiments/07_production/`.

1. **Extract the windows.** Run `extract_windows.ipynb`, which needs internet access to Hugging Face. Rename its outputs `chutes_W0/1/2.parquet` to `data/W0.parquet`, `data/W1.parquet` and `data/W2.parquet`. Then check them with `python ../../tools/verify_hashes.py`.
2. **Table IX (locked predictions).** Run `./run_window.sh W0`, then `W1`, then `W2`. This writes `data/W*_R*.json` and `.log`.
   * The R6b column is R6 recomputed with κ = 3.2: T*·(3.2/1.9)², checked against the same factor-of-2 rule, using `data/W*_R6.json`.
   * R2b uses `inv_w2.py` (pool-membership table).
3. **Data paragraph** (active accounts, requests, platform accounts). Printed by `extract_windows.ipynb` and by the `real` line of each `data/W*_R3.log`.
4. **Table X (real groups).**
   * The W0 "site" row comes from `data/W0_R5_R8_R9.json` (R9 entry: χ², p, class, ring z) and from `inv46.py` (models).
   * The heavy rows come from `python inv_w2.py W1` and `python inv_w2.py W2`.
   * `data/inv_W*.log` holds the committed outputs.
5. **"In W0 … no pool without a member fired."** `r2check.py`, `r2check2.py`.
6. **Activity floor of 10.** `floor_scan.py`.
7. **Discoverability law on real rings** (T* = 18, 59, 29 and 102 min).
   * `python law_real.py` gives θ1, τ_c and T*.
   * `python law_real2.py` gives the first time each ring is recovered.
8. **Table XI (adaptive attacker).**
   * `python attacker.py data/W2.parquet data/ATT_W2.json` produces the table. Throughput in the table is relative to `counter_J0` within each budget.
   * `python attacker.py data/W1.parquet data/ATT_W1.json` produces the replication paragraph.
9. **Hand-off graph on real traffic.** The `graph_injected_recovery` entry of `data/W*_R5_R8_R9.json`.

The audit trail lives in two places:
* `predictions/LOCKED_PREDICTIONS.md`: the lock, both amendments, and the recorded outcomes.
* `frozen/`: the code exactly as hashed before W1 and W2 were opened.

`tools/verify_hashes.py` checks both.

## Section XIII. Deployment

| Claim | Source |
|---|---|
| "2 human selections in 8,000 trials" | `experiments/06_two_replica/analysis/surrogate_seeds.py` |
| "47 of 800 … 6 of 800" | Two-replica notebook; `experiments/06_two_replica/analysis/graph_human_edges.py` |
| Expansion timings | see Table II |

## Appendix A. Proofs

| Claim | Source |
|---|---|
| Worker-count intervals | `posthoc/worker_count_intervals.py` |
| BBP: predicted outlier locations about 0.4 below observed | `cd experiments/03_baseline/analysis && python spectral.py && python theory_bbp.py` |
| Harmonics (7.09, 1.81, 0.22), etc. | `experiments/04_repeat/analysis/offline_misc.py` |

## Appendix C. Matrix-free spectrum

| Claim | Source |
|---|---|
| Lanczos reproduces the dense top-10 eigenvalues in 0.7 s | `experiments/04_repeat/analysis/offline_misc.py` |
| Chunked/Lanczos computation equals the dense one | `cd posthoc && python lowmem_equivalence.py NB7` (also `NB8B`, `NB9A`, `NB9B`) |

## Appendix D. Post-hoc analyses

| Claim | Source |
|---|---|
| Bulk edge 4.6/5.9/8.3 vs 2.8 | `experiments/05_stress_evasion/analysis/bulk_edge_by_scale.py` |
| v0.7 trimming: edge 2.6, five rings 20/20 | `experiments/05_stress_evasion/analysis/trim_v07_correlation_time.py` |
| v0.8 trims exactly the 20 E5 keys and 2 humans | `experiments/05_stress_evasion/analysis/trim_v08_persistence.py` |
| Rehearsal: ring keys 9.7 s vs 3.3 s; v0.7 trimmed 20 / 103 | `rehearsal/trim_v07_diagnosis.py A` |
| Rehearsal: v0.8 40/40 and 18/40 | `rehearsal/trim_v08_persistence.py` |
| Rehearsal hand-off graph: v0.4 10 humans; v0.8 4 → 0 | `rehearsal/graph_versions.py` (40-minute rehearsal) |
| 8.4% vs 8.7% within 50 ms | `rehearsal/human_lag_test.py` |
| v0.4 sideband mechanism | `rehearsal/sideband_edges.py`, `sideband_edges2.py` |
| Graph versions across Repeat, Stress and Evasion (v0.8 0 humans; jittered ring 78% → 5%; company 70% → 20%) | `cd posthoc && python graph_versions.py NB7` (also `NB8A`, `NB8B`) |
| Live Two-replica graph: v0.8 2 and 4 humans, v0.4 25 and 23; 8–14 starts vs 0.2–0.9 expected | `experiments/06_two_replica/analysis/graph_human_edges.py`, `graph_stints.py` |
| v0.9 removes every human edge; rotate-on-429 recall 2–5% | `experiments/06_two_replica/analysis/graph_v09.py` |
| Failover graph: 100%, purity 0.93; 5%; 2 humans | `experiments/06_two_replica/analysis/failover_graph.py` |
| Start-count ratio | `experiments/05_stress_evasion/analysis/crosstalk_*.py` |

**Running the rehearsal scripts.** Each one runs inside an extracted rehearsal folder:

```bash
mkdir r && tar xzf data/rehearsal/two_replica_rehearsal_2_40min.tar.gz -C r && cd r
python ../experiments/06_two_replica/rehearsal/graph_versions.py
```

* The 10-minute rehearsals (`_1_` and `_3_final`) have both phases.
* The 40-minute rehearsal has the balanced phase only.

## Appendix E. Ledger (Table XII)

| Rows | Source |
|---|---|
| WildChat, Pilot, Baseline, Repeat, Stress, Evasion, Two-replica | Each notebook's markdown states its locked predictions, and its last cell prints a scorecard. The Baseline scorecard is in `data/live/baseline_results.json`. |
| Production | `experiments/07_production/predictions/LOCKED_PREDICTIONS.md` |

## Not scripted

These numbers were computed by hand, or by code that was not kept. The method for each is given so it can be checked:

* **κ = 1.9 and τ_c = 1.56 s.** κ was measured in simulation as the ratio of the observed recovery threshold to σ√N, and τ_c on the Repeat run. To recompute τ_c, apply `fplib.key_corr_time` to the Repeat log.
* **Watermark jitter forecasts and the 6 s check** (0.2, 0.65, 4.4 and 19 h; z = 1.33 vs 1.26). These come from Proposition 5(ii)–(iii) with the measured q, F and λ_bg.
* **σ_L within 5–23% of measured.** Proposition 5(iii) applied to the Repeat run's four closed-loop pools.
* **Holm families.** Holm correction was applied to each run's watermark p-values: cross-key, halves and same-key, non-human pools only.
* **Fig. 4 (cost of evasion).** Plotted from the Evasion run's results for the counter with 0, 2, 6 and 15 s jitter: ring z and z_R as in Table VII, and the spread-watermark z printed by the notebook. The plotting script was not kept.
