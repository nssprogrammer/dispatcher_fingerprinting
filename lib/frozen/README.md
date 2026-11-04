# Frozen library versions

Each live run embedded the detector library as it stood when that run's predictions were locked. These are those copies, byte for byte. Use them to recompute a run's frozen results from its log in `data/live/`. `lib/fplib.py` is the final version, used by the Production study and by all post-hoc scripts.

| file | detector versions in the paper | used by |
|---|---|---|
| `fplib_v0.3_baseline.py` | spectral/graph prototypes | Baseline run |
| `fplib_v0.4_repeat.py` | v0.4 (spectral discovery, sideband hand-off graph, hold-back watermark) | Repeat run |
| `fplib_v0.6_stress_evasion.py` | v0.5 and v0.6 (multi-scale ring statistic, multi-harmonic spectral selection) | Stress and Evasion runs |
| `fplib_v0.8_two_replica.py` | v0.7 classifier, v0.7/v0.8 trimming, v0.8 hand-off graph | Two-replica run |
| `fplib_v0.8_dense_with_v0.9.py` | v0.8 plus the post-hoc v0.9 graph and failover graph, dense eigendecomposition | post-hoc analyses before the low-memory rewrite |
| `../fplib.py` | as above, with chunked hollowed correlation and Lanczos (identical output, see `posthoc/lowmem_equivalence.py`) | Production study (hash-frozen), post-hoc scripts |
| `stress_evasion_an.py` | per-phase analysis of the Stress and Evasion runs | Stress and Evasion notebook |

`lib/replica_an.py` is the per-phase analysis of the Two-replica run. It is unchanged since that run, and the Production study imports its `seed_expansion`.
