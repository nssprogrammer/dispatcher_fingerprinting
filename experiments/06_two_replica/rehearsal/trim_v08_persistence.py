import sys, json, numpy as np, pandas as pd
import sys as _s, pathlib as _p; ROOT = next(q for q in _p.Path(__file__).resolve().parents if (q / 'lib' / 'fplib.py').exists()); _s.path.insert(0, str(ROOT / 'lib'))
import fplib, replica_an as nb9an
exec(open(str(ROOT / 'lib/persist_test.py')).read())
for ph in 'AB':
    cfg = json.load(open(f"run_{ph}/driver_cfg.json")); d, labels, cl, t0, t1, ls, r429 = nb9an.load(f"run_{ph}", cfg)
    allk = sorted(d.key.unique()); idx={k:i for i,k in enumerate(allk)}
    A = d.key.map(idx).values.astype(np.int64); N=len(allk)
    grp = np.array([labels[k]["pop"] if labels[k]["pop"] not in ("hum_a","hum_b") else "hum" for k in allk])
    report(fplib, A, d.t_arrival.values, d.t_end.values, N, t0, t1, grp, np.random.default_rng(1), "NB9dry-"+ph)
