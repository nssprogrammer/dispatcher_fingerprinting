import json, re, sys
cells_path, lib_path, out_nb, out_py = sys.argv[1:5]
c = open(cells_path).read(); lib = open(lib_path).read()
parts = re.split(r'^# %% \[(markdown|code)\]\n', c, flags=re.M)
cells = []; py = []
def code(src): return {"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":src.splitlines(True)}
def md(src): return {"cell_type":"markdown","metadata":{},"source":src.splitlines(True)}
it = iter(parts[1:])
for kind, body in zip(it, it):
    body = body.rstrip() + "\n"
    if kind == "markdown":
        cells.append(md("\n".join(l[2:] if l.startswith("# ") else l.lstrip("#") for l in body.splitlines())))
    elif "LIB_PLACEHOLDER" in body:
        cells.append(code("%%writefile occlib.py\n" + lib))
        cells.append(code('import importlib, occlib\nimportlib.reload(occlib)\nprint("occlib", occlib.__version__)\n'))
        py.append("open('occlib.py','w').write(" + repr(lib) + ")\nimport importlib, occlib\nimportlib.reload(occlib)\n")
    else:
        cells.append(code(body)); py.append(body)
nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
      "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(out_nb, 'w'), indent=1)
open(out_py, "w").write("import sys, os; sys.path.insert(0, os.getcwd())\n" + "\n".join(py))
print("cells:", len(cells))
