import sys, json, collections, numpy as np, pandas as pd
sys.path.insert(0, '.'); import rt, fplib
from scipy import stats
win = sys.argv[1]; W = 3600.0
real, durs, act, strat = rt.load_window(f'data/{win}.parquet', W, floor=10, heavy_max=2000); raw = pd.read_parquet(f'data/{win}.parquet')
d = json.load(open(f'data/{win}_R1_R2.json')); pl = d.get('R1_pools', {}).get('heavy', [])
fire = [p for p in pl if p['ring_z'] > 6]
cf = collections.Counter(k for p in fire for k in p['keys']); ca = collections.Counter(k for p in pl for k in p['keys'])
base = len(fire) / max(len(pl), 1)
enr = sorted(((stats.binom.sf(cf[k] - 1, ca[k], base), k, cf[k], ca[k]) for k in cf), key=lambda x: x[0])
print("most enriched:", [(k, c, n, f"{p:.3g}") for p, k, c, n in enr[:15]])
# exploratory: spectral v0.8 on heavy accounts only
site = rt.Site(real[real.grp == 'real_heavy'].assign(status=200), W)
lab, ntr = fplib.spectral_rings_trimmed_v08(site.A, site.S, site.E, len(site.keys), 0, W, np.random.default_rng(3))
gs = collections.Counter(lab[lab >= 0]); print("heavy-only spectral groups:", dict(gs), "trimmed", ntr)
inw = raw[(raw.started >= 0) & (raw.started < W)]
for g, n in gs.items():
    ks = [site.keys[i] for i in np.nonzero(lab == g)[0]]; acc = [int(k[1:]) for k in ks]
    per = inw[inw.acct.isin(acc)].groupby('acct').size(); chi = ((per - per.mean()) ** 2 / per.mean()).sum()
    print(f" group {g}: n={n} counts p10/50/90={np.percentile(per,[10,50,90])} chi2={chi:.1f} df={len(per)-1} lower-p={stats.chi2.cdf(chi,len(per)-1):.3g}",
          "chutes", inw[inw.acct.isin(acc)].chute.value_counts(normalize=True).head(3).round(2).to_dict(), "fire-overlap", sum(cf[k] for k in ks))
    NULL, sn = site.null('real_heavy', np.random.default_rng(1), n=200)
    f = rt.fingerprint_pool(site, ks, NULL, sn, np.random.default_rng(2)); print("   classifier:", f['cls'], 'ring_z', round(f['ring_z'], 1), 'conc', f['conc_z'], 'R', f['R'])
# per-chute equal-count scan among heavy accounts: groups of >= 20 accounts on one home chute with request-count dispersion far below Poisson
heavy = act[strat == 'heavy'].index; h = inw[inw.acct.isin(heavy)]
home = h.groupby('acct').chute.agg(lambda x: x.value_counts().index[0]); per = h.groupby('acct').size()
for c, accs in home.groupby(home):
    if len(accs) < 20: continue
    x = per[accs.index]; disp = x.var() / x.mean()
    print(f" chute {c}: {len(accs)} heavy accts, counts median {x.median():.0f}, var/mean {disp:.2f}")
json.dump({str(g): [site.keys[i] for i in np.nonzero(lab == g)[0]] for g in gs}, open(f'data/{win}_heavy_groups.json', 'w'))
for g in gs:
    mem = set(site.keys[i] for i in np.nonzero(lab == g)[0])
    m = np.array([len(mem & set(p['keys'])) for p in pl]); z = np.array([p['ring_z'] for p in pl])
    for lo, hi in ((0, 0), (0, 3), (4, 99)):
        sel = (m >= lo) & (m <= hi); print(f" group {g}: pools with {lo}-{hi} members: {sel.sum()}, ring z>6: {(z[sel] > 6).sum()}")
