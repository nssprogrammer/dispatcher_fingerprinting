import json, re
nb5 = open('../02_pilot/cells.py').read()
parts5 = re.split(r'^# %% \[(markdown|code)\]\n', nb5, flags=re.M)
mid = []
it = iter(parts5[1:])
for kind, body in zip(it, it):
    if kind == "code" and body.lstrip().startswith(("# Cell 4", "# Cell 5", "# Cell 6")):
        body = body.replace('for f in ("gateway_log.csv", "labels.json"):', 'for f in ("gateway_log.csv",):')
        mid.append("# %% [code]\n" + body)
c = open('cells_head.py').read() + "\n" + "\n".join(mid) + "\n" + open('cells_tail.py').read()
open('cells.py', 'w').write(c)
files = {"LIB_PLACEHOLDER": ("occlib.py", open('../../lib/occlib.py').read()),
         "FINGERPRINT_PLACEHOLDER": ("fplib.py", open('../../lib/frozen/fplib_v0.3_baseline.py').read()),
         "SIMSERVER_PLACEHOLDER": ("simserver.py", open('simserver.py').read()),
         "GATEWAY_PLACEHOLDER": ("gateway.py", open('gateway.py').read()),
         "DRIVER_PLACEHOLDER": ("driver6.py", open('driver6.py').read())}
parts = re.split(r'^# %% \[(markdown|code)\]\n', c, flags=re.M)
cells = []; py = ["import sys, os; sys.path.insert(0, os.getcwd())\n"]
def code(src): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.splitlines(True)}
def md(src): return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(True)}
it = iter(parts[1:])
for kind, body in zip(it, it):
    body = body.rstrip() + "\n"
    if kind == "markdown":
        cells.append(md("\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines()))); continue
    hit = [k for k in files if k in body]
    if hit:
        fn, src = files[hit[0]]
        cells.append(code(f"%%writefile {fn}\n" + src)); py.append(f"open({fn!r},'w').write({src!r})\n")
    else:
        cells.append(code(body)); py.append(body)
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "accelerator": "GPU", "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open('baseline.ipynb', 'w'), indent=1)
open('rehearse.py', 'w').write("\n".join(py))
print("cells:", len(cells), "| mid cells:", [m.splitlines()[1][:30] for m in mid])
