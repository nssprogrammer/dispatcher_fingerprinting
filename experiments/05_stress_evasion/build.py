import json, re
c = open('cells_main.py').read() + "\n" + open('cells_run.py').read()
open('cells.py', 'w').write(c)
files = {"LIB_PLACEHOLDER": ("occlib.py", open('../../lib/occlib.py').read()),
         "FINGERPRINT_PLACEHOLDER": ("fplib.py", open('../../lib/frozen/fplib_v0.6_stress_evasion.py').read()),
         "ANALYSIS_PLACEHOLDER": ("nb8an.py", open('../../lib/frozen/stress_evasion_an.py').read()),
         "SIMSERVER_PLACEHOLDER": ("simserver.py", open('simserver.py').read()),
         "GATEWAY8_PLACEHOLDER": ("gateway8.py", open('gateway8.py').read()),
         "DRIVER8_PLACEHOLDER": ("driver8.py", open('driver8.py').read())}
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
json.dump(nb, open('stress_evasion.ipynb', 'w'), indent=1)
open('rehearse.py', 'w').write("\n".join(py))
print("cells:", len(cells))
