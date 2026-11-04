# Production study (Chutes trace)

The frozen detectors run on three one-hour windows of a public LLM serving trace. Predictions were locked before any data was downloaded. W0 was the development window, and W1 and W2 were held out with the code hash-frozen.

| path | what |
|---|---|
| `extract_windows.ipynb`, `extract.py`, `build.py` | download the shards and cut the three windows (needs internet) |
| `predictions/` | the locked predictions (lock, two amendments, recorded outcomes) and the three lock hashes |
| `frozen/` | the analysis code exactly as hashed before W1 and W2 were opened (`FROZEN_CODE.sha256`) |
| `rt.py`, `analyze_window.py`, `run_window.sh` | portable copies of the frozen code. They differ only in path lines; `python ../../tools/verify_hashes.py` prints the differences |
| `attacker.py` | adaptive attacker on real background (Table XI) |
| `inv_w2.py`, `inv46.py`, `r2check*.py`, `floor_scan.py`, `law_real*.py` | post-hoc analyses (Table X, the discoverability law on real rings, the activity floor) |
| `make_fake*.py` | small synthetic windows for testing the pipeline without downloading anything |
| `data/` | every output of the runs above, plus `data.sha256` for the windows |

Run everything from this folder; `REPRODUCE.md` gives the order. The windows are about 12–17 MB each. Rename `chutes_W*.parquet` to `data/W*.parquet` after extraction.
