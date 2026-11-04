"""Verify the Production study's audit trail.

1. Each lock hash (predictions/LOCKED*.sha256) matches a prefix of LOCKED_PREDICTIONS.md. The file was locked three
   times (original lock, amendment 1, amendment 2) and outcomes were appended afterwards, so every lock is the hash of
   the file as it stood at that moment.
2. The frozen analysis code (frozen/ and lib/fplib.py) matches FROZEN_CODE.sha256, recorded before the held-out windows
   were opened.
3. The portable copies differ from the frozen files only in path lines.
4. The extracted Chutes windows, if present in experiments/07_production/data/, match data.sha256.
"""
import hashlib, difflib, pathlib, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
P = ROOT / "experiments/07_production"
sha = lambda b: hashlib.sha256(b).hexdigest()
ok = True

text = (P / "predictions/LOCKED_PREDICTIONS.md").read_bytes()
for lock in sorted((P / "predictions").glob("LOCKED*.sha256")):
    h = lock.read_text().split()[0]
    hit = next((n for n in range(1, len(text) + 1) if sha(text[:n]) == h), None)
    print(f"{lock.name:20s} {'OK, first ' + str(hit) + ' bytes' if hit else 'NO MATCH'}")
    ok &= hit is not None

where = {"analyze_window.py": P / "frozen/analyze_window.py", "rt.py": P / "frozen/rt.py",
         "run_window.sh": P / "frozen/run_window.sh", "../fp/fplib.py": ROOT / "lib/fplib.py"}
for line in (P / "frozen/FROZEN_CODE.sha256").read_text().split("\n"):
    if not line.strip(): continue
    h, name = line.split()
    good = sha(where[name].read_bytes()) == h
    print(f"frozen {name:18s} {'OK' if good else 'MISMATCH'}"); ok &= good

for name in ("analyze_window.py", "rt.py", "run_window.sh"):
    a = (P / "frozen" / name).read_text().split("\n"); b = (P / name).read_text().split("\n")
    changed = [l for l in difflib.unified_diff(a, b, lineterm="", n=0) if l[:1] in "+-" and l[:3] not in ("+++", "---")]
    print(f"portable {name:16s} differs in {len(changed) // 2} line(s):")
    for l in changed: print("    " + l[:150])

for line in (P / "data/data.sha256").read_text().split("\n"):
    if not line.strip(): continue
    h, name = line.split(); f = P / "data" / pathlib.Path(name).name
    if f.exists():
        good = sha(f.read_bytes()) == h; print(f"window {f.name:10s} {'OK' if good else 'MISMATCH'}"); ok &= good
    else:
        print(f"window {f.name:10s} not present (extract it with extract_windows.ipynb)")
sys.exit(0 if ok else 1)
