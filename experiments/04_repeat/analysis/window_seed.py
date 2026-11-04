exec(open('load7.py').read())
from sklearn.metrics import roc_auc_score
rng = np.random.default_rng(3); N = len(keys)
hum = np.nonzero(grp == 'hum')[0]
def prep(tw):
    m = (S < t0 + tw) & (E > t0)
    a, s, e = A[m], S[m], E[m]; t1w = t0 + tw
    active = np.bincount(a, minlength=N) > 0
    H, Ex = fplib.handoff_excess(a, s, e, N, t0, t1w, 1.0)
    X = fplib.highpass(fplib.occupancy_matrix(a, s, e, N, t0, t1w, 0.1).astype(np.float64), 200)
    X -= X.mean(1, keepdims=True); sd = X.std(1); sd[sd == 0] = 1; X /= sd[:, None]
    return active, H + H.T, Ex + Ex.T, X
def scores(seed, cand, Hs, Es, X):
    agg = X[seed].sum(0); agg /= np.linalg.norm(agg) + 1e-12; proj = X[cand] @ agg
    hz = (Hs[np.ix_(cand, seed)].sum(1) - Es[np.ix_(cand, seed)].sum(1)) / np.sqrt(Es[np.ix_(cand, seed)].sum(1) + 1)
    return {"turn": -proj, "handoff": hz, "co": proj}
def pick(seed, act, Hs, Es, X):
    perc = {}
    for j in range(len(seed)):
        rest = np.delete(seed, j); cand = np.setdiff1d(act, rest); sc = scores(rest, cand, Hs, Es, X)
        for k, v in sc.items(): perc.setdefault(k, []).append((v < v[np.searchsorted(cand, seed[j])]).mean())
    return max(perc, key=lambda k: np.mean(perc[k]))
print("label-free seed expansion, 20% seed, AUC by observation window")
for g in ['P1_shuffle', 'C_company', 'P4_rrworker', 'P2_least']:
    row = []
    for tw in (300, 600, 1200, 2190):
        active, Hs, Es, X = prep(tw); act = np.nonzero(active)[0]
        mem = np.nonzero((grp == g) & active)[0]; res = []
        for rep in range(15):
            seed = np.sort(rng.choice(mem, max(2, int(round(0.2 * len(mem)))), replace=False))
            cand = np.setdiff1d(act, seed); y = np.isin(cand, mem).astype(int)
            k = pick(seed, act, Hs, Es, X); res.append(roc_auc_score(y, scores(seed, cand, Hs, Es, X)[k]))
        row.append(np.mean(res))
    print(f"  {g:12s} 5 min {row[0]:.2f} | 10 min {row[1]:.2f} | 20 min {row[2]:.2f} | 36 min {row[3]:.2f}", flush=True)
# K uncertainty: moving-block bootstrap over 60 s blocks
print("worker count, 95% block-bootstrap interval")
for g in ['P1_shuffle', 'P2_least', 'P3_rrshared', 'P4_rrworker', 'C_company', 'P5_rrjitter']:
    m = (d.grp == g).values; s, e = S[m], E[m]
    K0 = fplib.K_little(s, e, t0, t1)[0]; nb = int((t1 - t0) // 60); bs = []
    for b in range(200):
        blocks = rng.integers(0, nb, nb); ss, ee = [], []
        for i, bl in enumerate(blocks):
            mm = (s >= t0 + bl * 60) & (s < t0 + bl * 60 + 60)
            shift = (i - bl) * 60; ss.append(s[mm] + shift); ee.append(e[mm] + shift)
        bs.append(fplib.K_little(np.concatenate(ss), np.concatenate(ee), t0, t0 + nb * 60)[0])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f"  {g:12s} K = {K0:.1f}  [{lo:.1f}, {hi:.1f}]")
