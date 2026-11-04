import numpy as np, sys
import sys as _s, pathlib as _p; ROOT = next(q for q in _p.Path(__file__).resolve().parents if (q / 'lib' / 'fplib.py').exists()); _s.path.insert(0, str(ROOT / 'lib'))
import fplib; from dispatch import simulate
from gens import per_worker_rr
rng = np.random.default_rng(3); T = 2400.0; t0, t1 = 60.0, T
def excess_lag_hist(a, s, e, edges):
    """pool-level end->start lag histogram minus expectation (starts uniform in time), any key"""
    o = np.sort(s); lam = len(s) / (t1 - t0); out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        n = (np.searchsorted(o, e + hi, "right") - np.searchsorted(o, e + lo, "right")).sum()
        out.append(n - len(e) * lam * (hi - lo))
    return np.array(out)
edges = np.r_[0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16]
for name, gen, K in [("rr K12", lambda: simulate(rng, T, 12, 40, "rr"), 12), ("rr K8 gap3", lambda: simulate(rng, T, 8, 40, "rr", gap_med=3.0, gap_sig=0.6), 8),
                     ("rr K24", lambda: simulate(rng, T, 24, 40, "rr"), 24), ("random K6", lambda: simulate(rng, T, 6, 40, "random"), 6),
                     ("least K10", lambda: simulate(rng, T, 10, 40, "least"), 10), ("perworker K8", lambda: per_worker_rr(rng, T, 8, 40), 8)]:
    a, s, e = gen(); m = (s > t0) & (e < t1); a, s, e = a[m], s[m], e[m]
    M = 40; lam = len(s) / (t1 - t0); ED = (e - s).mean()
    h = excess_lag_hist(a, s, e, edges); mid = (edges[:-1] + edges[1:]) / 2
    hp = np.clip(h, 0, None); EG = (hp * mid).sum() / hp.sum()
    C = fplib.corr_matrix(fplib.highpass(fplib.occupancy_matrix(a, s, e, M, t0, t1), 400)); order = fplib.circular_seriation(C)
    res = []
    for w in (0.5, 2.0, 8.0):
        X = fplib.successor_concentration(a, s, e, M, t0, t1, w)[1]; off = fplib.ring_offsets(X, order)[1:20]
        sm = np.convolve(off, np.ones(3) / 3, "same")
        res.append((w, int(np.argmax(sm)) + 1, round(float(np.sum(np.arange(1, 20) * off) / off.sum()), 1)))
    print(f"{name:13s} K={K:2d}  Little K=lam*(ED+EG)={lam * (ED + EG):5.1f} (EG={EG:.2f}, true gap mean {np.exp(np.log(0.3)+0.125) if 'gap3' not in name else np.exp(np.log(3)+0.18):.2f})  ring (win, mode, mean): {res}")
