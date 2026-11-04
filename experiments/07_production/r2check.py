import sys, numpy as np
sys.path.insert(0, '.'); import rt
W = 3600.0; real, durs, act, strat = rt.load_window('data/W0.parquet', W, floor=10, heavy_max=2000)
site = rt.Site(real.assign(status=200), W); g46 = set('r' + str(a) for a in np.load('data/W0_g46_accts.npy'))
rng = np.random.default_rng(10); heavy = list(site.by['real_heavy']); print('ring members in heavy stratum:', len(g46 & set(heavy)), 'of', len(heavy))
NULL, sn = site.null('real_heavy', rng, n=200)
for label, excl in (("pools with no ring member", True), ("pools with >=2 ring members", False)):
    zs = []
    for _ in range(40):
        if excl: ks = list(rng.choice([k for k in heavy if k not in g46], 40, replace=False))
        else: ks = list(rng.choice(sorted(g46 & set(heavy)), 2, replace=False)) + list(rng.choice([k for k in heavy if k not in g46], 38, replace=False))
        zs.append(rt.fingerprint_pool(site, ks, NULL, sn, rng)['ring_z'])
    zs = np.array(zs); print(f"{label}: ring z > 6 in {np.mean(zs > 6):.2f} of 40 pools (median {np.median(zs):.1f})", flush=True)
