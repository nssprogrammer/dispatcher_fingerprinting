exec(open('load7.py').read())
from sklearn.metrics import roc_auc_score, average_precision_score
rng = np.random.default_rng(11); N = len(keys)
active = np.bincount(A, minlength=N) > 0; act = np.nonzero(active)[0]
H, Ex = fplib.handoff_excess(A, S, E, N, t0, t1, 1.0); Hs, Es = H + H.T, Ex + Ex.T
X = fplib.highpass(fplib.occupancy_matrix(A, S, E, N, t0, t1, 0.1).astype(np.float64), 200)
X -= X.mean(1, keepdims=True); sd = X.std(1); sd[sd == 0] = 1; X /= sd[:, None]
hum = np.nonzero(grp == 'hum')[0]
def scores(seed, cand):
    agg = X[seed].sum(0); agg /= np.linalg.norm(agg) + 1e-12; proj = X[cand] @ agg
    hz = (Hs[np.ix_(cand, seed)].sum(1) - Es[np.ix_(cand, seed)].sum(1)) / np.sqrt(Es[np.ix_(cand, seed)].sum(1) + 1)
    return {"turn-taking": -proj, "hand-off": hz, "co-movement": proj}
def pick(seed):
    """label-free choice: the score under which held-out seed members rank highest (leave-one-out)"""
    if len(seed) < 2: return "turn-taking"
    perc = {}
    for j in range(len(seed)):
        rest = np.delete(seed, j); cand = np.setdiff1d(act, rest); sc = scores(rest, cand)
        for k, v in sc.items():
            perc.setdefault(k, []).append((v < v[np.searchsorted(cand, seed[j])]).mean())
    return max(perc, key=lambda k: np.mean(perc[k]))
out = {}
for g in ['P1_shuffle', 'P2_least', 'P3_rrshared', 'P4_rrworker', 'P5_rrjitter', 'P6_agents', 'C_company']:
    mem = np.nonzero((grp == g) & active)[0]
    for frac in (0.1, 0.2, 0.3):
        res, chosen = [], []
        for rep in range(20):
            ns = max(2, int(round(frac * len(mem)))); seed = np.sort(rng.choice(mem, ns, replace=False))
            cand = np.setdiff1d(act, seed); y = np.isin(cand, mem).astype(int)
            k = pick(seed); s = scores(seed, cand)[k]; chosen.append(k)
            top = np.argsort(-s)[:y.sum()]
            res.append((roc_auc_score(y, s), average_precision_score(y, s), y[top].mean()))
        r = np.mean(res, 0); c = max(set(chosen), key=chosen.count)
        out[(g, frac)] = r
        print(f"{g:14s} seed {int(frac*100):2d}%: AUC {r[0]:.3f} AP {r[1]:.3f} precision@pool {r[2]:.2f}  (score chosen: {c}, {chosen.count(c)}/20)", flush=True)
