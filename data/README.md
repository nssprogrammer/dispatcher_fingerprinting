# Data

## `live/`: request logs of the live runs

One gzipped CSV per run or phase. Each row is one request seen by the logging gateway. Times are Unix seconds.

| column | meaning |
|---|---|
| `key` | API key (account). The pool is the prefix before the last `-`, e.g. `P3_rrshared-017`; `hum-…` keys are simulated humans |
| `rid` | request ID, shared with the client logs |
| `t_arrival` | request arrived at the gateway |
| `t_first` | first byte of the answer left the gateway |
| `t_complete` | model finished generating (absent in the Baseline run, which had no watermark) |
| `t_end` | last byte left the gateway (equals `t_complete` plus any watermark delay) |
| `status` | HTTP status (200, or 429 when a per-key quota refused the request) |
| `nbytes` | bytes streamed |
| `wm_bit` | secret watermark coin for this request |
| `replica` | serving replica (Two-replica run only) |
| `grp`, `pos`, `dur` | pool, position in the client's key list, and duration (Baseline and Repeat logs only; derived columns) |

| file | run | keys | requests |
|---|---|---|---|
| `baseline_requests.csv.gz` | Baseline | 694 | 61,238 |
| `repeat_requests.csv.gz` | Repeat | 703 | 65,143 |
| `stress_requests.csv.gz` | Stress | 704 | 40,260 |
| `evasion_requests.csv.gz` | Evasion | 632 | 80,201 |
| `two_replica_balanced.csv.gz` | Two-replica, balanced phase | 742 | 77,543 |
| `two_replica_affinity.csv.gz` | Two-replica, affinity phase | 742 | 77,066 |

`baseline_results.json` is the Baseline notebook's printed results, including its locked-prediction scorecard.

All clients are synthetic, so the logs contain no user content and no personal data. Released under CC BY 4.0.

## `rehearsal/`: simulated rehearsals of the Two-replica run

Each archive holds the raw run folders (`run_A/`, `run_B/`): the gateway log, per-population client logs, labels, driver configuration, and the printed results (`out.txt`). The scripts in `experiments/06_two_replica/rehearsal/` read them. Extract an archive and run the scripts from inside it (see `REPRODUCE.md`).

| archive | contents |
|---|---|
| `two_replica_rehearsal_1_10min.tar.gz` | first 10-minute rehearsal, both phases (early v0.8 library) |
| `two_replica_rehearsal_2_40min.tar.gz` | 40-minute rehearsal, balanced phase |
| `two_replica_rehearsal_3_10min_final.tar.gz` | final 10-minute rehearsal with the frozen library, both phases |

## Production study

The Chutes windows (`W0`, `W1`, `W2`) are derived from the public Chutes trace and are not stored here. `experiments/07_production/extract_windows.ipynb` regenerates them, and `experiments/07_production/data/data.sha256` lets you check the result. Every analysis output (`W*_R*.json`, `.log`, heavy-group memberships, adaptive-attacker results) is committed under `experiments/07_production/data/`.
