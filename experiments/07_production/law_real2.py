import sys, json, numpy as np
sys.path.insert(0, '.'); import rt, fplib
W = 3600.0
cases = [("W2", "heavy", 1727), ("W0", "heavy", 1055), ("W0", "all", 3568), ("W2", "all", 6113)]
for win, scope, Tstar in cases:
    real, durs, act, strat = rt.load_window(f'data/{win}.parquet', W, floor=10, heavy_max=2000)
    if scope == "heavy": real = real[real.grp == "real_heavy"]
    S6 = rt.Site(real.assign(status=200), W); mem = set(json.load(open(f'data/{win}_heavy_groups.json'))["0"])
    idx = np.array([S6.keys.index(k) for k in mem if k in S6.keys]); out = []
    for Tm in (5, 10, 15, 20, 30, 45, 60):
        T = Tm * 60.0; m = S6.S < T
        lab, _ = fplib.spectral_rings_trimmed_v08(S6.A[m], S6.S[m], np.minimum(S6.E[m], T), len(S6.keys), 0, T, np.random.default_rng(3))
        best = (0, 0)
        for g in set(lab[lab >= 0]):
            sel = np.nonzero(lab == g)[0]; hit = len(set(sel) & set(idx)); best = max(best, (hit / len(idx), hit / len(sel)))
        out.append((Tm, round(best[0], 2), round(best[1], 2)))
        print(win, scope, "N", len(S6.keys), "T*", Tstar, "T", Tm, "recall/purity", out[-1][1:], flush=True)
