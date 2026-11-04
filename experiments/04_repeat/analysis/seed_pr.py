exec(open('load7.py').read())
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import rankdata
rng = np.random.default_rng(11); N = len(keys)
active = np.bincount(A, minlength=N) > 0
H, Ex = fplib.handoff_excess(A, S, E, N, t0, t1, 1.0); Hs, Es = H + H.T, Ex + Ex.T
X = fplib.highpass(fplib.occupancy_matrix(A, S, E, N, t0, t1, 0.1).astype(np.float64), 200)
X -= X.mean(1, keepdims=True); sd = X.std(1); sd[sd == 0] = 1; X /= sd[:, None]
hum = np.nonzero(grp == 'hum')[0]
rows = []
for g in ['P1_shuffle', 'P2_least', 'P3_rrshared', 'P4_rrworker', 'P5_rrjitter', 'P6_agents', 'C_company']:
    mem = np.nonzero((grp == g) & active)[0]
    for frac in (0.1, 0.2, 0.3):
        for purity in (1.0, 0.5):
            res = []
            for rep in range(20):
                ns = max(1, int(round(frac * len(mem)))); nc = max(1, int(round(ns * purity)))
                seed = np.r_[rng.choice(mem, nc, replace=False), rng.choice(hum, ns - nc, replace=False)].astype(int)
                cand = np.setdiff1d(np.nonzero(active)[0], seed); y = np.isin(cand, mem).astype(int)
                agg = X[seed].sum(0); agg /= np.linalg.norm(agg) + 1e-12
                occ = -(X[cand] @ agg)                                   # turn-taking with the seed
                hz = (Hs[np.ix_(cand, seed)].sum(1) - Es[np.ix_(cand, seed)].sum(1)) / np.sqrt(Es[np.ix_(cand, seed)].sum(1) + 1)
                pos = np.abs(X[cand] @ agg)                              # any coupling (rings correlate positively with neighbours)
                comb = rankdata(occ) + rankdata(hz) + rankdata(pos)
                k = y.sum(); top = np.argsort(-comb)[:k]
                res.append((roc_auc_score(y, comb), average_precision_score(y, comb), y[top].mean()))
            r = np.mean(res, 0)
            rows.append((g, frac, purity, *r))
            print(f"{g:14s} seed {int(frac*100):2d}% purity {purity:.1f}: AUC {r[0]:.3f}  AP {r[1]:.3f}  precision@pool-size {r[2]:.2f}", flush=True)
import json; json.dump(rows, open('seed_pr.json', 'w'))
