"""Event-driven simulator of K closed-loop slots sharing one dispatcher over M keys, plus covariance/seriation tools."""
import numpy as np, heapq

def simulate(rng, T, K, M, policy="random", dur_med=1.0, dur_sig=0.5, gap_med=0.3, gap_sig=0.5, stint=None):
    """policy: random | rr (shared round-robin cycle) | least (fewest in-flight, ties random) | sticky (slot keeps a key for `stint` s)
    returns acct, s, e arrays."""
    heap = [(rng.uniform(0, 1), w) for w in range(K)]; heapq.heapify(heap)
    inflight = np.zeros(M, int); ends = []           # (end_time, key)
    rr = 0; held = {}
    A, S, E = [], [], []
    while heap:
        t, w = heapq.heappop(heap)
        if t >= T: continue
        while ends and ends[0][0] <= t:
            _, k = heapq.heappop(ends); inflight[k] -= 1
        if policy == "rr":
            k = rr % M; rr += 1
        elif policy == "least":
            m = inflight.min(); k = int(rng.choice(np.flatnonzero(inflight == m)))
        elif policy == "sticky":
            key = (w, int(t // stint))
            if key not in held: held[key] = int(rng.integers(M))
            k = held[key]
        else:
            k = int(rng.integers(M))
        d = rng.lognormal(np.log(dur_med), dur_sig)
        A.append(k); S.append(t); E.append(t + d)
        inflight[k] += 1; heapq.heappush(ends, (t + d, k))
        heapq.heappush(heap, (t + d + rng.lognormal(np.log(gap_med), gap_sig), w))
    return np.array(A), np.array(S), np.array(E)

def occupancy_matrix(acct, s, e, M, t0, t1, dt=0.05):
    """X[i, t] on a fine grid (exploration only)."""
    n = int((t1 - t0) / dt)
    X = np.zeros((M, n), np.float32)
    i0 = np.clip(((s - t0) / dt).astype(int), 0, n); i1 = np.clip(((e - t0) / dt).astype(int), 0, n)
    for k, a, b in zip(acct, i0, i1):
        X[k, a:b] += 1
    return X

def highpass(X, block):
    """subtract each block's mean (block in grid steps)"""
    M, n = X.shape; nb = n // block
    Y = X[:, : nb * block].reshape(M, nb, block)
    return (Y - Y.mean(2, keepdims=True)).reshape(M, nb * block)

def corr_matrix(X):
    C = np.cov(X); d = np.sqrt(np.diag(C)); return C / np.outer(d, d)

def circular_seriation(Cm):
    """Recover a circular order from a similarity matrix: angle of the two leading non-trivial eigenvectors."""
    S = np.clip(Cm, 0, None); np.fill_diagonal(S, 0)
    w, V = np.linalg.eigh(S)
    v1, v2 = V[:, -1], V[:, -2]
    # for circulant-like matrices the top eigenpair is often constant; use the two leading eigenvectors after removing the constant
    cand = [(V[:, -i], V[:, -j]) for i, j in [(1, 2), (2, 3)]]
    best = None
    for a, b in cand:
        ang = np.arctan2(b, a)
        r = np.hypot(a, b).std() / (np.hypot(a, b).mean() + 1e-12)   # a ring embeds on a circle: low radial spread
        if best is None or r < best[0]: best = (r, ang)
    return np.argsort(best[1]), best[1]

def circular_order_score(order, M):
    """fraction of predicted neighbours that are true ring neighbours (either direction)."""
    nb = 0
    for i in range(M):
        a, b = order[i], order[(i + 1) % M]
        if (a - b) % M in (1, M - 1): nb += 1
    return nb / M
