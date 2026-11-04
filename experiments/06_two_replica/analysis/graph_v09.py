exec(open('load_balanced.py').read())
exec(open(str(ROOT / 'lib/graph_eval.py')).read())
evalg(fplib, A, S, E, N, t0, t1, grp, "NB9live-A")
