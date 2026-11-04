import sys
which=sys.argv[1]
if which=='NB7':
    import os; import pathlib; os.chdir(pathlib.Path(__file__).resolve().parents[1] / 'experiments/04_repeat/analysis'); exec(open('load7.py').read())
    S_,E_=S,E
else:
    import os; import pathlib; os.chdir(pathlib.Path(__file__).resolve().parents[1] / 'experiments/05_stress_evasion/analysis'); exec(open({'A': 'load_stress.py', 'B': 'load_evasion.py'}[which[-1]]).read())
    A=d.a.values.astype(np.int64); S_,E_=d.t_arrival.values,d.t_end.values
exec(open(str(ROOT / 'lib/graph_eval.py')).read())
evalg(fplib,A,S_,E_,len(keys),t0,t1,np.asarray(grp),which)
