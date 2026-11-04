import sys, numpy as np
sys.path.insert(0, '.'); import rt
from scipy.stats import hypergeom
W = 3600.0; real, durs, act, strat = rt.load_window('data/W0.parquet', W, floor=10, heavy_max=2000)
site = rt.Site(real.assign(status=200), W); g46 = sorted(set('r' + str(a) for a in np.load('data/W0_g46_accts.npy')))
rng = np.random.default_rng(11); heavy = list(site.by['real_heavy']); rest = [k for k in heavy if k not in set(g46)]
NULL, sn = site.null('real_heavy', rng, n=200)
for m in (3, 4, 5, 6, 8):
    zs = [rt.fingerprint_pool(site, list(rng.choice(g46, m, replace=False)) + list(rng.choice(rest, 40 - m, replace=False)), NULL, sn, rng)['ring_z'] for _ in range(20)]
    print(f"{m} ring members: ring z>6 in {np.mean(np.array(zs) > 6):.2f}; P(random 40-pool has >= {m}) = {hypergeom.sf(m - 1, len(heavy), 46, 40):.3f}", flush=True)
