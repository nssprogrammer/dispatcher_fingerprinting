import sys, numpy as np, pandas as pd
sys.path.insert(0, '.'); import rt, fplib
from occlib import occupancy
W = 3600.0; real, durs, act, strat = rt.load_window('data/W0.parquet', W, floor=10); raw = pd.read_parquet('data/W0.parquet')
site = rt.Site(real.assign(status=200), W)
lab, ntr = fplib.spectral_rings_trimmed_v08(site.A, site.S, site.E, len(site.keys), 0, W, np.random.default_rng(3))
g = np.nonzero(lab >= 0)[0]; ks = [site.keys[i] for i in g]; acc = [int(k[1:]) for k in ks]
np.save('data/W0_g46_accts.npy', np.array(acc))
sub = raw[raw.acct.isin(acc)]; inw = sub[(sub.started >= 0) & (sub.started < W)]
per = inw.groupby('acct').size()
print("group", len(g), "requests/acct p10/50/90", np.percentile(per, [10, 50, 90]), "total", len(inw), "share of site", round(len(inw) / ((raw.started >= 0) & (raw.started < W)).sum(), 4))
print("chutes:", inw.chute.value_counts(normalize=True).head(5).round(3).to_dict(), "distinct", inw.chute.nunique())
print("per-account home chute:", inw.groupby('acct').chute.agg(lambda x: x.value_counts().index[0]).value_counts().head(5).to_dict())
print("instances distinct", inw.inst.nunique(), "duration median", round(float((inw.completed - inw.started).median()), 2), "ot median", float(inw.ot.median()), "ttft NaN", round(float(inw.ttft.isna().mean()), 2))
print("per-account concurrency>1 frac:", end=" ")
cc = []
for a_, x in inw.groupby('acct'):
    ev = np.r_[x.started.values, x.completed.values]; sg = np.r_[np.ones(len(x)), -np.ones(len(x))]; o = np.lexsort((sg, ev)); cc.append(np.cumsum(sg[o]).max())
print(round(float(np.mean(np.array(cc) > 1)), 2), "max conc median", np.median(cc))
NULL, sn = site.null('real_heavy', np.random.default_rng(1), n=200)
f = rt.fingerprint_pool(site, ks, NULL, sn, np.random.default_rng(2)); print("frozen classifier on the group:", f)
a, s, e, kk = site.pool(ks)[1:]; C = fplib.hollow_corr(a, s, e, len(kk), 0, W); iu = np.triu_indices(len(kk), 1)
print("pairwise corr: mean", round(float(C[iu].mean()), 3), "frac>0", round(float((C[iu] > 0).mean()), 2), "p10/p90", np.round(np.percentile(C[iu], [10, 90]), 3))
w = np.linalg.eigvalsh(C); print("top eigenvalues of group's hollowed corr:", np.round(w[-4:], 2), "bottom:", np.round(w[:2], 2))
h = np.histogram(inw.started, bins=12, range=(0, W))[0]; print("group starts per 5 min:", h.tolist())
first = inw.groupby('acct').started.min(); last = inw.groupby('acct').started.max()
print("first start (min) p10/50/90:", np.round(np.percentile(first / 60, [10, 50, 90]), 1), " last start:", np.round(np.percentile(last / 60, [10, 50, 90]), 1))
# successor structure: after an end on account i, which account in the group starts next within 1 s?
o = np.argsort(s); ss, aa = s[o], a[o]; cnt = 0; same = 0
for ei, ai in zip(e, a):
    j = np.searchsorted(ss, ei); 
    if j < len(ss) and ss[j] - ei < 1.0: cnt += 1; same += int(aa[j] == ai)
print("ends followed by a group start within 1 s:", cnt, "of", len(e), " same account:", same)
