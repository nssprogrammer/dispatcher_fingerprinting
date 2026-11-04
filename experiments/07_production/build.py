import json
lock = open('predictions/LOCKED_PREDICTIONS.md').read(); sha = open('predictions/LOCKED.sha256').read().split()[0]
lib = open('extract.py').read()
cells = []
md = lambda t: cells.append({"cell_type": "markdown", "metadata": {}, "source": t})
code = lambda t: cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": t})
md("# NB10 — real production traffic: extract three windows from the Chutes trace\n\n"
   "**Settings:** Accelerator **None (CPU)**, Internet **On**. About 15–40 minutes, depending on Hugging Face speed.\n\n"
   "**What it does:** it reads three 1-hour windows from the public Chutes LLM serving trace (Hugging Face `harvardMadsys/chutes_metadata`, CC BY 4.0), "
   "fetching only the row groups it needs. It writes `chutes_W0.parquet`, `chutes_W1.parquet` and `chutes_W2.parquet` (accounts and instances re-coded as integers, "
   "times in seconds from each window's start) and prints a profile.\n\n"
   "**What to send back:** the PASTE block, and the three `.parquet` files from the Output panel.\n\n"
   f"The analysis predictions were locked before this notebook was written (sha256 `{sha}`):\n\n---\n\n" + lock)
code("# Cell 1 — libraries\n!pip -q install -U huggingface_hub pyarrow fsspec 2>/dev/null\nimport os, json, time\nT0 = time.time()")
code("# Cell 2 — extractor\n" + lib)
code("""# Cell 3 — pick shards at 20%, 50% and 80% of the year; W0 (50%) is the development window
shards = list_shards()
print(len(shards), "trace shards; first", shards[0], "last", shards[-1])
n = len(shards)
PICK = {"W0": shards[round(0.5 * (n - 1))], "W1": shards[round(0.2 * (n - 1))], "W2": shards[round(0.8 * (n - 1))]}
print(PICK)""")
code("""# Cell 4 — read the windows (row-group reads; falls back to a full download if remote reads fail)
OUT = {}
for w, path in PICK.items():
    try:
        df, meta = read_window(path)
    except Exception as ex:
        print(f"  remote read failed ({type(ex).__name__}: {ex}); downloading {path}")
        from huggingface_hub import hf_hub_download
        local = hf_hub_download(REPO, path, repo_type="dataset", local_dir="/kaggle/tmp/chutes")
        df, meta = read_window(local); meta["shard"] = path
        os.remove(local)
    fn = f"chutes_{w}.parquet"; df.to_parquet(fn, compression="zstd", index=False)
    meta["file_mb"] = round(os.path.getsize(fn) / 1e6, 1); meta["profile"] = profile(df, meta["window_s"])
    OUT[w] = meta
    print(w, json.dumps({k: v for k, v in meta.items() if k != "profile"}))
    print("   ", json.dumps(meta["profile"])[:1500], flush=True)""")
code("""# Cell 5 — summary to paste
OUT["runtime_min"] = round((time.time() - T0) / 60, 1); OUT["lock_sha256"] = \"""" + sha + """\"
json.dump(OUT, open("nb10_profile.json", "w"), indent=1)
print("===== PASTE FROM HERE =====")
print(json.dumps(OUT, separators=(",", ":")))
print("===== PASTE TO HERE =====")
for w in ("W0", "W1", "W2"):
    if w in OUT: print(f"chutes_{w}.parquet  {OUT[w]['file_mb']} MB")""")
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open("extract_windows.ipynb", "w"), indent=1)
# local test script: same cells without pip / with fake shards
py = "\n\n".join(c["source"] for c in cells if c["cell_type"] == "code")
py = py.replace("!pip -q install -U huggingface_hub pyarrow fsspec 2>/dev/null", "")
py = py.replace("shards = list_shards()", "import glob\nshards = sorted(glob.glob('fake/shard_*.parquet'))")
open("rehearse.py", "w").write(py)
print("cells", len(cells))
