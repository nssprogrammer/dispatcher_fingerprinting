import sys, json, numpy as np, pandas as pd
import sys as _s, pathlib as _p; ROOT = next(q for q in _p.Path(__file__).resolve().parents if (q / 'lib' / 'fplib.py').exists()); _s.path.insert(0, str(ROOT / 'lib'))
import fplib, replica_an as nb9an
ph = sys.argv[1]
cfg = json.load(open(f"run_{ph}/driver_cfg.json"))
d, labels, cl, t0, t1, ls, r429 = nb9an.load(f"run_{ph}", cfg)
allk = sorted(d.key.unique()); idx = {k:i for i,k in enumerate(allk)}
A = d.key.map(idx).values.astype(np.int64); S, E = d.t_arrival.values, d.t_end.values; N=len(allk)
grp = np.array([labels[k]["pop"] if labels[k]["pop"] not in ("hum_a","hum_b") else "hum" for k in allk])
tau = fplib.key_corr_time(A,S,E,N,t0,t1)
df = pd.DataFrame(dict(g=grp, tau=tau))
act = np.bincount(A, minlength=N)>0
med = np.median(tau[act]); mad = 1.4826*np.median(np.abs(tau[act]-med))
print("window", t1-t0, "med", med, "mad", mad, "cut", med+5*mad)
print(df.groupby("g").tau.describe()[["count","min","50%","max"]].round(2))
df["trim"] = tau > med+5*mad
print(df[df.trim].g.value_counts())
