# dispatcher-fingerprinting

Code and data for the paper **"Fingerprinting the Dispatchers behind Distributed Model Extraction"** (S. C. Khilar, 2026).

Campaigns that distil a model through its API spread their requests over many accounts. The client code that decides which key each worker uses next, the *dispatcher*, leaves a recognisable mark in two timestamps per request: when it arrived and when its answer finished streaming. This repository contains the detectors, the live testbed, the analysis scripts behind every number in the paper, the request logs of the live runs, and the locked predictions and frozen-code hashes for the production-traffic study.

`REPRODUCE.md` maps every table, figure and quoted number in the paper to the script that produces it.

## Layout

```
lib/                        detector and analysis libraries (import these)
  occlib.py                 occupancy ratio, nulls, worker counts (Fano, Little's law)
  fplib.py                  fingerprints, ring/successor statistics, spectral discovery, hand-off graphs,
                            watermark, classifier (final version, byte-identical to the frozen production code)
  replica_an.py             per-phase analysis of the Two-replica run, incl. label-free seed expansion
  graph_eval.py             evaluates hand-off graph versions v0.4 / v0.8 / v0.9
  persist_test.py           persistence-time trimming (v0.8 prototype)
  frozen/                   the exact library versions each live run used (v0.3 ... v0.8)
simulation/                 pre-live simulations used to fix the classifier thresholds and the watermark design
experiments/
  00_foundations/           unit tests of the occupancy library against brute force and the closed forms
  01_wildchat/              WildChat study: false-alarm rate among real users; calibrated group scan
  02_pilot/                 Pilot run (first live check)
  03_baseline/              Baseline run + analysis/ (ring-profile theory, BBP check)
  04_repeat/                Repeat run + analysis/ (forensic expansion, spectral split, harmonics, Lanczos)
  05_stress_evasion/        Stress and Evasion runs + analysis/ (cross-talk, bulk-edge inflation, trimming)
  06_two_replica/           Two-replica run + analysis/ (surrogate seeds, hand-off graph) + rehearsal/
  07_production/            Production study on the Chutes trace: extraction, locked predictions,
                            frozen analysis, real-group scans, discoverability law, adaptive attacker
posthoc/                    cross-run checks (worker-count intervals, graph versions, low-memory equivalence)
figures/                    scripts for Figs. 1-3 (output goes to figures/out/)
data/
  live/                     request logs of the live runs (gzipped CSV), Baseline results JSON
  rehearsal/                raw folders of the three simulated rehearsals of the Two-replica run
tools/verify_hashes.py      checks the production study's locks, frozen code and data hashes
```

Each live experiment folder holds the notebook that was run (`*.ipynb`), its cell sources (`cells*.py`), the client populations (`driver*.py`), the logging gateway (`gateway*.py`), a simulated model server for rehearsals (`simserver.py`), and `build.py`, which assembles the notebook from those parts and embeds the library version that run was frozen with.

## Quick start

```bash
pip install -r requirements.txt

# reproduce a theory check from the Baseline run (ring profile vs Proposition 1; RMSE 0.015-0.020)
cd experiments/03_baseline/analysis && python theory_checks2.py

# worker-count intervals (Appendix A)
cd posthoc && python worker_count_intervals.py

# verify the production study's locked predictions and frozen code
python tools/verify_hashes.py
```

Analysis scripts are run from their own folder (several of them `exec` a sibling loader such as `load.py`). They find the repository root automatically and read the logs from `data/live/`.

## Running the live experiments

The live runs used Qwen2.5-0.5B-Instruct on vLLM 0.32.0 on a single NVIDIA T4 GPU. Each notebook installs its dependencies, starts the server, the gateway and every client population, and prints its results and the outcome of its locked predictions. Every notebook can also be rehearsed on a CPU against the simulated server: `python build.py` writes `rehearse.py` next to the notebook.

## Data

* **Live runs** (`data/live/`): one row per request with key, arrival, first byte, model completion, last byte, status, bytes and watermark coin. Pool membership is the key prefix (e.g. `P3_rrshared-017`). All clients are synthetic.
* **WildChat** ([allenai/WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M); see its dataset card for the licence): downloaded by the WildChat notebooks; not redistributed here.
* **Chutes trace** ([harvardMadsys/chutes_metadata](https://huggingface.co/datasets/harvardMadsys/chutes_metadata), CC BY 4.0; Nixon et al., arXiv:2608.13573): the three one-hour windows are extracted by `experiments/07_production/extract_windows.ipynb` and are not stored in the repository. Their SHA-256 hashes are in `experiments/07_production/data/data.sha256`, and all analysis outputs are included.

See `data/README.md` for formats.

## A note on names

The paper names experiments in plain words (Baseline, Repeat, Stress, ...). Some code comments, print labels and command-line arguments still use the short internal labels from development, and the frozen files cannot be edited without breaking their hashes:

| label in code | experiment in the paper |
|---|---|
| NB1 | Foundations (simulation) |
| NB2, NB4 | WildChat study |
| NB5 | Pilot run |
| NB6 | Baseline run |
| NB7 | Repeat run |
| NB8 A / NB8 B | Stress run / Evasion run |
| NB9 A / NB9 B | Two-replica run, balanced / affinity phase |
| NB9 dry run, `NB9dry`, `NB9sim40` | rehearsal of the Two-replica run |
| NB10 | Production study |

Detector versions (v0.3 to v0.9) are named as in the paper; `lib/frozen/README.md` says which run used which.

## Citation

```bibtex
@misc{khilar2026dispatchers,
  author = {Snigdha Chandan Khilar},
  title  = {Fingerprinting the Dispatchers behind Distributed Model Extraction},
  year   = {2026},
  note   = {Code: https://github.com/nssprogrammer/dispatcher_fingerprinting}
}
```

## License

Code: MIT (see `LICENSE`). The live request logs in `data/` are released under CC BY 4.0. Results derived from the Chutes trace inherit its CC BY 4.0 licence.
