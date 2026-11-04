import numpy as np, pandas as pd, sys
import sys as _s, pathlib as _p; ROOT = next(q for q in _p.Path(__file__).resolve().parents if (q / 'lib' / 'fplib.py').exists()); _s.path.insert(0, str(ROOT / 'lib'))
import fplib, occlib
d = pd.read_csv(ROOT / 'data/live/evasion_requests.csv.gz'); d = d[d.status == 200]
d['grp'] = d.key.str.rsplit('-', n=1).str[0]; d['pos'] = d.key.str.rsplit('-', n=1).str[1].astype(int)
T0 = d.t_arrival.min(); t0 = T0 + 240; t1 = T0 + 2400 - 30
d = d[(d.t_end > t0) & (d.t_arrival < t1)].copy()
keys = sorted(d.key.unique()); kid = {k: i for i, k in enumerate(keys)}; d['a'] = d.key.map(kid)
grp = d.groupby('key').grp.first().reindex(keys).values
