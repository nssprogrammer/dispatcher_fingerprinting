import numpy as np, heapq, sys
import sys as _s, pathlib as _p; ROOT = next(q for q in _p.Path(__file__).resolve().parents if (q / 'lib' / 'fplib.py').exists()); _s.path.insert(0, str(ROOT / 'lib'))
from dispatch import simulate
from occlib import open_loop_relay, benign_humans
def per_worker_rr(rng, T, K, M):
    heap = [(rng.uniform(0, 1), w) for w in range(K)]; heapq.heapify(heap); pos = rng.integers(M, size=K); A, S, E = [], [], []
    while heap:
        t, w = heapq.heappop(heap)
        if t >= T: continue
        k = pos[w] % M; pos[w] += 1; d = rng.lognormal(0, 0.5); A.append(k); S.append(t); E.append(t + d)
        heapq.heappush(heap, (t + d + rng.lognormal(np.log(0.3), 0.5), w))
    return np.array(A), np.array(S), np.array(E)
def agents(rng, T, K, M, steps=8, tool=2.0):
    rows = []
    for w in range(K):
        t = rng.uniform(0, 5)
        while t < T:
            k = rng.integers(M)
            for _ in range(rng.geometric(1 / steps)):
                d = rng.lognormal(0, 0.5); rows.append((k, t, t + d)); t += d + rng.lognormal(np.log(tool), 0.7)
            t += rng.lognormal(0, 0.5)
    a, s, e = map(np.array, zip(*rows)); return a, s, e
def humans(rng, M, T, think):
    a, s, e = benign_humans(rng, M, T, session_rate=1 / 300, think_med=think)
    return a, s, s + (e - s) / 8.0
def ol(rng, T, pol, M, lam):
    x = open_loop_relay(rng, T, M, lam, pol); return x[0], x[1], x[1] + (x[2] - x[1]) / 8
def site(rng, T, jitter=0.0):
    spec = [("hum", lambda: humans(rng, 400, T, 6.0), 400, None),
        ("shuffle_K6", lambda: simulate(rng, T, 6, 40, "random"), 40, 6),
        ("least_K10", lambda: simulate(rng, T, 10, 40, "least"), 40, 10),
        ("rrshared_K12", lambda: simulate(rng, T, 12, 40, "rr"), 40, 12),
        ("rrworker_K8", lambda: per_worker_rr(rng, T, 8, 40), 40, 8),
        ("rrjit_K8", lambda: simulate(rng, T, 8, 40, "rr", gap_med=3.0, gap_sig=0.6), 40, 8),
        ("agents_K6", lambda: agents(rng, T, 6, 32), 32, 6),
        ("ol_random", lambda: ol(rng, T, "random", 40, 2.0), 40, None),
        ("ol_rr", lambda: ol(rng, T, "rr", 40, 2.0), 40, None),
        ("company_K4", lambda: simulate(rng, T, 4, 10, "random"), 10, 4)]
    A, S, E, lab = [], [], [], []; off = 0; truth = {}
    for name, f, M, K in spec:
        a, s, e = f(); A.append(a.astype(int) + off); S.append(s); E.append(e); truth[name] = (off, off + M, K); lab += [name] * M; off += M
    return np.concatenate(A), np.concatenate(S), np.concatenate(E), np.array(lab), truth
