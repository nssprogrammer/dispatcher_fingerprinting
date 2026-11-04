import numpy as np, pandas as pd, sys
import sys as _s, pathlib as _p; ROOT = next(q for q in _p.Path(__file__).resolve().parents if (q / 'lib' / 'fplib.py').exists()); _s.path.insert(0, str(ROOT / 'lib'))
import fplib, occlib
d = pd.read_csv(ROOT / 'data/live/repeat_requests.csv.gz')
T_START = 1792959464.535086; t0 = T_START + 180; t1 = t0 + 2190
keys = sorted(d.key.unique()); kid = {k: i for i, k in enumerate(keys)}
d['a'] = d.key.map(kid); grp = d.groupby('key').grp.first().reindex(keys).values
A, S, E, Cc, Bb = d.a.values.astype(np.int64), d.t_arrival.values, d.t_end.values, d.t_complete.values, d.wm_bit.values.astype(int)
