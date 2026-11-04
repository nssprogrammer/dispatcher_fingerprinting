import numpy as np, heapq, sys
import sys as _s, pathlib as _p; ROOT = next(q for q in _p.Path(__file__).resolve().parents if (q / 'lib' / 'fplib.py').exists()); _s.path.insert(0, str(ROOT / 'lib'))
import fplib
from occlib import occupancy
from gens import humans
rng = np.random.default_rng(21); T = 2400.0; t0, t1 = 60.0, T; M = 20
def closed(K, M, policy, J=0.0, H=1):
    heap = [(rng.uniform(0, 2), w) for w in range(K)]; heapq.heapify(heap); ctr = 0; held = {}; A, S, E = [], [], []
    while heap:
        t, w = heapq.heappop(heap)
        if t >= T: continue
        if policy == 'rr': k = ctr % M; ctr += 1
        elif policy == 'hold':
            n, k = held.get(w, (0, None))
            if k is None or n >= H: k = ctr % M; ctr += 1; n = 0
            held[w] = (n + 1, k)
        elif policy == 'pinned': k = w % M
        else: k = int(rng.integers(M))
        d = rng.lognormal(np.log(2.1), 0.5); A.append(k); S.append(t); E.append(t + d)
        heapq.heappush(heap, (t + d + rng.lognormal(np.log(0.25), 0.5) + (rng.uniform(0, J) if J else 0), w))
    return np.array(A), np.array(S), np.array(E)
def bursty(M, on=30, off=60, lam=6.0):
    A, S, E = [], [], []; t = 0.0
    while t < T:
        end = t + rng.exponential(on)
        while t < min(end, T):
            t += rng.exponential(1 / lam)
            if t < end: d = rng.lognormal(np.log(2.1), 0.5); A.append(int(rng.integers(M))); S.append(t); E.append(t + d)
        t = end + rng.exponential(off)
    return np.array(A), np.array(S), np.array(E)
null = {k: [] for k in ('R', 'R5', 'R20', 'R120')}
for _ in range(40):
    a, s, e = humans(rng, M, T, 6.0); f, _ = fplib.fingerprint(a, s, e, M, t0, t1, rng, n_sur=0, occupancy_fn=occupancy)
    for k in null: null[k].append(f[k])
NR = {k: (np.mean(v), np.std(v)) for k, v in null.items()}
pools = {'rr J=0': lambda: closed(8, M, 'rr'), 'rr J=2': lambda: closed(8, M, 'rr', 2), 'rr J=6': lambda: closed(8, M, 'rr', 6), 'rr J=15': lambda: closed(8, M, 'rr', 15),
         'rr J=15 K=24': lambda: closed(24, M, 'rr', 15), 'hold H=5': lambda: closed(8, M, 'hold', H=5), 'hold H=20': lambda: closed(8, M, 'hold', H=20),
         'shuffle K=8': lambda: closed(8, M, 'random'), 'shuffle K=18': lambda: closed(18, M, 'random'), 'pinned K=8': lambda: closed(8, 8, 'pinned'),
         'bursty open loop': lambda: bursty(M)}
for name, f in pools.items():
    a, s, e = f(); Mk = int(a.max()) + 1
    fp, _ = fplib.fingerprint(a, s, e, Mk, t0, t1, rng, n_sur=20, occupancy_fn=occupancy)
    zR = {k: (NR[k][0] - fp[k]) / NR[k][1] for k in NR}
    print(f"{name:16s} req/s {len(a)/T:5.2f}  R {fp['R']:.2f} R5 {fp['R5']:.2f}  zR {zR['R']:5.1f}  ring z {fp['ring_z']:5.1f}  conc z {fp['conc_z']:5.1f}  -> {fplib.classify(fp, zR)}")
