import json, re
nb5 = open('../02_pilot/cells.py').read()
parts5 = re.split(r'^# %% \[(markdown|code)\]\n', nb5, flags=re.M)
mid = []
it = iter(parts5[1:])
for kind, body in zip(it, it):
    if kind == "code" and body.lstrip().startswith(("# Cell 4", "# Cell 5", "# Cell 6")):
        body = body.replace('for f in ("gateway_log.csv", "labels.json"):', 'for f in ("gateway_log.csv",):')
        body = body.replace('[sys.executable, "gateway.py", "--port", "9000", "--backend", "http://127.0.0.1:8000", "--log", "gateway_log.csv"]',
                            '[sys.executable, "gateway7.py", "--port", "9000", "--backend", "http://127.0.0.1:8000", "--log", "gateway_log.csv", "--delta", str(CFG["wm_delta"]), "--seed", "424242"]')
        mid.append("# %% [code]\n" + body)
assert any("gateway7.py" in m for m in mid)
c = open('cells_head.py').read() + "\n" + "\n".join(mid) + "\n" + open('cells_tail.py').read()
open('cells.py', 'w').write(c)
files = {"LIB_PLACEHOLDER": ("occlib.py", open('../../lib/occlib.py').read()),
         "FINGERPRINT_PLACEHOLDER": ("fplib.py", open('../../lib/frozen/fplib_v0.4_repeat.py').read()),
         "SIMSERVER_PLACEHOLDER": ("simserver.py", open('simserver.py').read()),
         "GATEWAY7_PLACEHOLDER": ("gateway7.py", open('gateway7.py').read()),
         "DRIVER7_PLACEHOLDER": ("driver7.py", open('driver7.py').read())}
parts = re.split(r'^# %% \[(markdown|code)\]\n', c, flags=re.M)
cells = []; py = ["import sys, os; sys.path.insert(0, os.getcwd())\n"]
def code(src): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.splitlines(True)}
def md(src): return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(True)}
it = iter(parts[1:])
for kind, body in zip(it, it):
    body = body.rstrip() + "\n"
    if kind == "markdown":
        cells.append(md("\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines()))); continue
    hit = [k for k in files if re.search(r'^' + k + r' = True', body, flags=re.M)]
    if hit:
        fn, src = files[hit[0]]
        cells.append(code(f"%%writefile {fn}\n" + src)); py.append(f"open({fn!r},'w').write({src!r})\n")
    else:
        cells.append(code(body)); py.append(body)
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "accelerator": "GPU", "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open('repeat.ipynb', 'w'), indent=1)
open('rehearse.py', 'w').write("\n".join(py))
print("cells:", len(cells), [ (c['source'][0][:35].strip()) for c in cells])
