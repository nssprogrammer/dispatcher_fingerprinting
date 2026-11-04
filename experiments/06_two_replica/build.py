import json, re
c = open('cells.py').read()
files = {"LIB_PLACEHOLDER": ("occlib.py", open('../../lib/occlib.py').read()),
         "FINGERPRINT_PLACEHOLDER": ("fplib.py", open('../../lib/frozen/fplib_v0.8_two_replica.py').read()),
         "ANALYSIS_PLACEHOLDER": ("nb9an.py", open('../../lib/replica_an.py').read()),
         "SIMSERVER_PLACEHOLDER": ("simserver.py", open('simserver.py').read()),
         "GATEWAY9_PLACEHOLDER": ("gateway9.py", open('gateway9.py').read()),
         "DRIVER9_PLACEHOLDER": ("driver9.py", open('driver9.py').read())}
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
json.dump(nb, open('two_replica.ipynb', 'w'), indent=1)
open('rehearse.py', 'w').write("\n".join(py))
print("cells:", len(cells))
