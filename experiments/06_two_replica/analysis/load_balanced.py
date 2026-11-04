import sys, numpy as np, pandas as pd
import sys as _s, pathlib as _p; ROOT = next(q for q in _p.Path(__file__).resolve().parents if (q / 'lib' / 'fplib.py').exists()); _s.path.insert(0, str(ROOT / 'lib'))
import fplib, occlib
raw = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 and sys.argv[1].endswith(('.csv', '.csv.gz')) else str(ROOT / 'data/live/two_replica_balanced.csv.gz'))
raw['grp'] = raw.key.str.rsplit('-', n=1).str[0]
T0 = raw.t_arrival.min(); t0 = T0 + 180; t1 = min(raw.t_end.max(), T0 + 2400) - 30
ok = raw[raw.status == 200]; r429 = raw[raw.status == 429]
d = ok[(ok.t_end > t0) & (ok.t_arrival < t1)].copy()
keys = sorted(d.key.unique()); kid = {k: i for i, k in enumerate(keys)}; d['a'] = d.key.map(kid)
grp = d.groupby('key').grp.first().reindex(keys).values
A, S, E = d.a.values.astype(np.int64), d.t_arrival.values, d.t_end.values; N = len(keys)
