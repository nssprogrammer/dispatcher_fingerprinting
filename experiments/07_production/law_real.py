import sys, json, numpy as np, pandas as pd
sys.path.insert(0, '.'); import rt, fplib
W = 3600.0
for win, grpfile, gid in (("W0", "W0_heavy_groups.json", "0"), ("W2", "W2_heavy_groups.json", "0")):
    real, durs, act, strat = rt.load_window(f'data/{win}.parquet', W, floor=10, heavy_max=2000)
    site = rt.Site(real.assign(status=200), W)
    ks = json.load(open(f'data/{grpfile}'))[gid]
    a, s, e, kk = site.pool(ks)[1:]; C = fplib.hollow_corr(a, s, e, len(kk), 0, W); w = np.linalg.eigvalsh(C)
    rng = np.random.default_rng(5); bg = [k for k in site.keys if k not in set(ks)]; samp = rng.choice(bg, 300, replace=False)
    _, a2, s2, e2, k2 = site.pool(list(samp)); tc = float(np.mean(fplib.key_corr_time(a2, s2, e2, len(k2), 0, W)))
    th = float(w[-1]); nh = int((strat == 'heavy').sum()); na = len(site.keys)
    for kap in (1.9, 3.2):
        print(win, f"theta1={th:.2f} pair={w[-2]:.2f} tau_c={tc:.2f} kappa={kap}: T*(heavy N={nh})={kap**2*nh*tc/th**2:.0f}s  T*(all N={na})={kap**2*na*tc/th**2:.0f}s")
