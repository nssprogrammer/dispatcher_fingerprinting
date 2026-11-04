import sys, numpy as np, pandas as pd
sys.path.insert(0, '.'); import rt, fplib
fl = int(sys.argv[1]); W = 3600.0
real, durs, act, strat = rt.load_window('data/W0.parquet', W, floor=fl)
allr = rt.inject(real, durs, W, np.random.default_rng(1010)); site = rt.Site(allr, W)
lab, ntr = fplib.spectral_rings_trimmed_v08(site.A, site.S, site.E, len(site.keys), 0, W, np.random.default_rng(3))
groups = [pd.Series(site.grp[lab == c]).value_counts().to_dict() for c in np.unique(lab[lab >= 0])]
rec, pur = rt.recovered(lab, site.grp, "I_rrshared")
print(f"floor {fl}: real accounts {len(act)}, keys {len(site.keys)}, trimmed {ntr}, ring recall {rec:.2f} purity {pur:.2f}, groups {groups}", flush=True)
