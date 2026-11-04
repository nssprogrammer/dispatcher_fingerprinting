import sys, numpy as np, importlib.util
which = sys.argv[1]; import pathlib; SCR = str(pathlib.Path(__file__).resolve().parents[1])
def load_mod(name, path):
    sp = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m); return m
import os
if which == 'NB7':
    os.chdir(SCR + '/experiments/04_repeat/analysis'); exec(open('load7.py').read()); S_, E_ = S, E; G = np.asarray(grp)
elif which == 'NB8B':
    os.chdir(SCR + '/experiments/05_stress_evasion/analysis'); exec(open('load_evasion.py').read()); A = d.a.values.astype(np.int64); S_, E_ = d.t_arrival.values, d.t_end.values; G = np.asarray(grp)
else:
    os.chdir(SCR + '/experiments/06_two_replica/analysis'); exec(open({'A': 'load_balanced.py', 'B': 'load_affinity.py'}[which[-1]]).read()); S_, E_ = S, E; G = np.asarray(grp)
old = load_mod('old', SCR + '/fp/fplib_pre_lowmem.py'); new = load_mod('new', SCR + '/fp/fplib.py'); new.EIGH_MAX = 0
N = len(keys)
lo, n1 = old.spectral_rings_trimmed_v08(A, S_, E_, N, t0, t1, np.random.default_rng(5))
ln, n2 = new.spectral_rings_trimmed_v08(A, S_, E_, N, t0, t1, np.random.default_rng(5))
same = all(set(np.nonzero(lo == c)[0]) in [set(np.nonzero(ln == c2)[0]) for c2 in np.unique(ln[ln >= 0])] for c in np.unique(lo[lo >= 0])) and (lo >= 0).sum() == (ln >= 0).sum()
print(which, 'N', N, 'old selected', int((lo >= 0).sum()), 'groups', len(np.unique(lo[lo >= 0])), '| new selected', int((ln >= 0).sum()), 'groups', len(np.unique(ln[ln >= 0])), '| trimmed', n1, n2, '| identical groups:', same)
