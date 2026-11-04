"""Active timing watermark: provider delays the end of each request by b*delta (b = secret random bit).
A closed-loop worker's next request (on ANY key) shifts by delta. Randomisation test, exact p-value."""
import numpy as np, heapq
rng = np.random.default_rng(11)
T, delta = 1800.0, 0.05

def run(kind, K=6, M=40, jitter=0.0, think=6.0):
    """returns per request: key, start, true completion c, bit b (sent end = c + b*delta), worker ignored"""
    heap = [(rng.uniform(0, 5 if kind != "humans_via_relay" else 600), w) for w in range(K)]; heapq.heapify(heap)
    A, S, C, B = [], [], [], []
    if kind == "open_loop":
        t = 0.0
        while t < T:
            t += rng.exponential(1 / 2.0); A.append(rng.integers(M)); S.append(t); C.append(t + rng.lognormal(0, .5)); B.append(rng.integers(2))
        return map(np.array, (A, S, C, B))
    while heap:
        t, w = heapq.heappop(heap)
        if t >= T: continue
        k = (w % M) if kind == "pinned" else rng.integers(M)
        d = rng.lognormal(0, .5); b = rng.integers(2)
        A.append(k); S.append(t); C.append(t + d); B.append(b)
        g = (rng.lognormal(np.log(think), 0.9) if kind == "humans_via_relay" else rng.lognormal(np.log(0.2), .5) + rng.uniform(0, jitter))
        heapq.heappush(heap, (t + d + b * delta + g, w))
    return map(np.array, (A, S, C, B))

def wm_test(a, s, c, b, w, n_perm=2000):
    """D_k = (#other-key starts in [c+delta, c+delta+w)) - (#other-key starts in [c, c+w)); T = mean D|b=1 - mean D|b=0"""
    o = np.argsort(s); ss, sa = s[o], a[o]
    def cnt(lo, hi):
        i0, i1 = np.searchsorted(ss, lo), np.searchsorted(ss, hi)
        same = np.array([np.sum(sa[x:y] == k) for x, y, k in zip(i0, i1, a)])
        return (i1 - i0) - same
    D = cnt(c + delta, c + delta + w) - cnt(c, c + w)
    Tobs = D[b == 1].mean() - D[b == 0].mean()
    perm = np.array([(lambda bb: D[bb == 1].mean() - D[bb == 0].mean())(rng.permutation(b)) for _ in range(n_perm)])
    return Tobs, (Tobs - perm.mean()) / perm.std(), (1 + (perm >= Tobs).sum()) / (n_perm + 1)

print(f"{'pool':34s} {'req':>6s} {'T':>7s} {'z':>6s} {'p':>7s}")
for name, kw, w in [("closed loop, random keys", dict(kind="closed"), 1.0),
                    ("closed loop, random keys, 0-6 s jitter", dict(kind="closed", jitter=6.0), 8.0),
                    ("closed loop, PINNED keys (least-busy)", dict(kind="pinned", K=10), 1.0),
                    ("open-loop relay, random keys", dict(kind="open_loop"), 1.0),
                    ("relay of 150 humans (think 6 s)", dict(kind="humans_via_relay", K=150, think=6.0), 30.0)]:
    a, s, c, b = run(**kw)
    m = s > 60; a, s, c, b = a[m], s[m], c[m], b[m]
    Tobs, z, p = wm_test(a, s, c, b, w)
    print(f"{name:40s} {len(a):6d} {Tobs:7.3f} {z:6.1f} {p:7.4f}")
