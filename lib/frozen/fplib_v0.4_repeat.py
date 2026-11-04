"""fplib — dispatcher fingerprinting from per-key request logs (start, end) only.

Signals (all computed without labels):
  ring     zero-lag high-pass occupancy correlation arranged on a recovered circle (shared round-robin counter)
  conc     concentration of each key's short-lag end->start successors (per-worker round-robin counters)
  coll     same-key overlap relative to uniform spreading of concurrent requests (least-busy / round-robin avoid collisions)
  R(tau)   occupancy variance ratio at several timescales (closed-loop turn-taking, sticky/episodic key holding)
  K_little worker count from the interactive response-time law, think time read off the successor-lag excess
"""
import numpy as np

__version__ = "fp-0.4"


# ---------------------------------------------------------------- occupancy on a grid
def occupancy_matrix(a, s, e, M, t0, t1, dt=0.05):
    n = int((t1 - t0) / dt)
    i0 = np.clip(np.floor((s - t0) / dt).astype(np.int64), 0, n)
    i1 = np.clip(np.floor((e - t0) / dt).astype(np.int64), 0, n)
    D = np.zeros((M, n + 1), np.float32)
    np.add.at(D, (a, i0), 1.0); np.add.at(D, (a, i1), -1.0)
    return np.cumsum(D, 1)[:, :n]


def highpass(X, block):
    M, n = X.shape; nb = n // block
    Y = X[:, : nb * block].reshape(M, nb, block)
    return (Y - Y.mean(2, keepdims=True)).reshape(M, nb * block)


def corr_matrix(X):
    C = np.cov(X.astype(np.float64)); d = np.sqrt(np.clip(np.diag(C), 0, None))
    with np.errstate(invalid="ignore", divide="ignore"):
        R = C / np.outer(d, d)
    R[~np.isfinite(R)] = 0.0
    return R


# ---------------------------------------------------------------- circular seriation
def circular_seriation(C):
    """Order keys around a circle from a similarity matrix (angle in the plane of two leading eigenvectors)."""
    S = np.clip(C, 0, None).copy(); np.fill_diagonal(S, 0)
    w, V = np.linalg.eigh(S)
    best = None
    for i, j in [(1, 2), (2, 3), (1, 3)]:
        x, y = V[:, -i], V[:, -j]
        r = np.hypot(x, y)
        score = r.std() / (r.mean() + 1e-12)          # points of a ring embed at constant radius
        if best is None or score < best[0]:
            best = (score, np.arctan2(y, x))
    return np.argsort(best[1])


def ring_score(C, order):
    """mean correlation between circular neighbours in the recovered order minus mean off-diagonal correlation"""
    M = len(order)
    nb = np.mean([C[order[i], order[(i + 1) % M]] for i in range(M)])
    off = C[~np.eye(M, dtype=bool)].mean()
    return float(nb - off)


def neighbour_accuracy(order, true_pos):
    """fraction of recovered neighbours that are true ring neighbours; true_pos[k] = key k's true ring position"""
    M = len(order); p = np.asarray(true_pos)[order]
    return float(np.mean([((p[i] - p[(i + 1) % M]) % M) in (1, M - 1) for i in range(M)]))


# ---------------------------------------------------------------- end -> start successors
def handoff_counts(a, s, e, M, lag_lo=0.0, lag_hi=1.0):
    """H[i, j] = number of (end on key i, start on key j within (lag_lo, lag_hi] s after that end)"""
    o = np.argsort(s, kind="stable"); ss, sa = s[o], a[o]
    lo = np.searchsorted(ss, e + lag_lo, "right"); hi = np.searchsorted(ss, e + lag_hi, "right")
    cnt = hi - lo
    H = np.zeros((M, M))
    if cnt.sum() == 0:
        return H
    src = np.repeat(a, cnt)
    idx = np.repeat(lo - np.r_[0, np.cumsum(cnt)[:-1]], cnt) + np.arange(cnt.sum())
    np.add.at(H, (src, sa[idx]), 1)
    return H


def handoff_excess(a, s, e, M, t0, t1, lag_hi=1.0):
    """observed minus expected successor counts (expected: key j's starts spread uniformly over time)"""
    H = handoff_counts(a, s, e, M, 0.0, lag_hi)
    ends = np.bincount(a, minlength=M).astype(float); rate = ends / (t1 - t0)
    Ex = np.outer(ends, rate) * lag_hi
    return H, Ex


def successor_concentration(a, s, e, M, t0, t1, lag_hi=1.0, min_rows=5):
    H, Ex = handoff_excess(a, s, e, M, t0, t1, lag_hi)
    X = np.clip(H - Ex, 0, None); np.fill_diagonal(X, 0)
    rows = X.sum(1); ok = rows >= min_rows
    if ok.sum() < 3:
        return float("nan"), X
    return float(np.mean(X[ok].max(1) / rows[ok])), X


def ring_offsets(X, order):
    """distribution of successor offsets measured along the recovered circle (direction chosen by mass)"""
    M = len(order); pos = np.empty(M, int); pos[order] = np.arange(M)
    off = np.zeros(M)
    I, J = np.nonzero(X)
    np.add.at(off, (pos[J] - pos[I]) % M, X[I, J])
    h = M // 2
    if off[h + 1:].sum() > off[1:h].sum():
        off = np.r_[off[0], off[1:][::-1]]
    return off


def K_ring(X, order):
    off = ring_offsets(X, order); M = len(order); h = M // 2
    w = off[1:h]
    return float((np.arange(1, h) * w).sum() / w.sum()) if w.sum() > 0 else float("nan")


# ---------------------------------------------------------------- collisions
def collision_ratio(a, s, e, M, t0, t1):
    """time-integrated same-key concurrent pairs / expected if concurrent requests were spread uniformly over M keys"""
    ev = np.r_[s, e]; dlt = np.r_[np.ones(len(s), int), -np.ones(len(e), int)]; kk = np.r_[a, a]
    o = np.lexsort((dlt, ev)); ev, dlt, kk = ev[o], dlt[o], kk[o]
    X = np.zeros(M, int); N = 0; pairs = 0; obs = exp_ = 0.0; last = t0
    for t, d, k in zip(ev.tolist(), dlt.tolist(), kk.tolist()):
        span = min(t, t1) - max(last, t0)
        if span > 0:
            obs += pairs * span; exp_ += N * (N - 1) / 2.0 / M * span
        last = t
        if d > 0: pairs += X[k]
        else: pairs -= X[k] - 1
        X[k] += d; N += d
    return float(obs / exp_) if exp_ > 0 else float("nan")


# ---------------------------------------------------------------- surrogates
def shift_keys(a, s, e, t0, t1, rng, M):
    """independent circular time shift per key (keeps each key's own pattern, destroys cross-key timing)"""
    L = t1 - t0; off = rng.uniform(0, L, M)[a]
    s2 = t0 + np.mod(s - t0 + off, L)
    return a, s2, s2 + (e - s)


# ---------------------------------------------------------------- worker count from the interactive response-time law
LAG_EDGES = np.r_[0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16]


def excess_lag_hist(s, e, t0, t1, edges=LAG_EDGES):
    """pool-level histogram of (end -> any later start in the pool) lags minus its expectation if starts were uniform in time"""
    o = np.sort(s); lam = len(s) / (t1 - t0)
    return np.array([(np.searchsorted(o, e + hi, "right") - np.searchsorted(o, e + lo, "right")).sum() - len(e) * lam * (hi - lo)
                     for lo, hi in zip(edges[:-1], edges[1:])])


def K_little(s, e, t0, t1, edges=LAG_EDGES):
    """K = X * (E[D] + E[G]) (Denning-Buzen interactive response time law, N = X (R + Z)) with throughput X, mean response
    E[D] and the client gap ('think time') E[G] read off the excess successor-lag histogram. Valid for always-busy closed loops."""
    h = np.clip(excess_lag_hist(s, e, t0, t1, edges), 0, None); mid = (edges[:-1] + edges[1:]) / 2
    EG = float((h * mid).sum() / h.sum()) if h.sum() > 0 else float("nan")
    lam = len(s) / (t1 - t0)
    return lam * (float(np.mean(e - s)) + EG), EG


# ---------------------------------------------------------------- full fingerprint of one pool
def best_lag_window(a, s, e, M, t0, t1, windows=(0.5, 1.0, 2.0, 4.0, 8.0)):
    """successor window with the largest cross-key excess z (sharpest view of the closed-loop hand-off)"""
    off = ~np.eye(M, dtype=bool); best = (windows[0], -np.inf)
    for w in windows:
        H, Ex = handoff_excess(a, s, e, M, t0, t1, w)
        z = (H[off].sum() - Ex[off].sum()) / np.sqrt(Ex[off].sum() + 1)
        if z > best[1]: best = (w, float(z))
    return best


def fingerprint(a, s, e, M, t0, t1, rng, n_sur=40, dt=0.05, hp_block_s=20.0, lag_hi=None, occupancy_fn=None):
    """a: keys 0..M-1 (requests overlapping [t0, t1)). Returns (features, recovered circular order)."""
    m = (e > t0) & (s < t1); a, s, e = a[m], s[m], e[m]
    out = dict(n_req=int(len(a)), n_keys=int(M))
    if lag_hi is None:
        lag_hi, out["handoff_z"] = best_lag_window(a, s, e, M, t0, t1)
    out["lag_win"] = lag_hi
    if occupancy_fn is not None:
        for tau in (None, 5.0, 20.0, 120.0):
            if tau is None or (t1 - t0) >= 2 * tau:
                out["R" if tau is None else f"R{int(tau)}"] = float(occupancy_fn(a, s, e, t0, t1, tau, n_accounts=M)["R"])
    out["coll"] = collision_ratio(a, s, e, M, t0, t1)
    out["K_little"], out["think_s"] = K_little(s, e, t0, t1)
    def ring_of(a_, s_, e_):
        C = corr_matrix(highpass(occupancy_matrix(a_, s_, e_, M, t0, t1, dt), int(hp_block_s / dt)))
        o = circular_seriation(C); return ring_score(C, o), o
    out["ring"], order = ring_of(a, s, e)
    out["conc"], X = successor_concentration(a, s, e, M, t0, t1, lag_hi)
    if n_sur == 0:
        return out, order
    sr, sc = [], []
    for _ in range(n_sur):
        aa, ss, ee = shift_keys(a, s, e, t0, t1, rng, M)
        sr.append(ring_of(aa, ss, ee)[0]); sc.append(successor_concentration(aa, ss, ee, M, t0, t1, lag_hi)[0])
    sr, sc = np.array(sr), np.array(sc)
    out["ring_p"] = float((1 + (sr >= out["ring"]).sum()) / (n_sur + 1))
    out["ring_z"] = float((out["ring"] - sr.mean()) / (sr.std() + 1e-12))
    sc = sc[np.isfinite(sc)]
    ok = len(sc) > 2 and np.isfinite(out["conc"])
    out["conc_z"] = float((out["conc"] - sc.mean()) / (sc.std() + 1e-12)) if ok else float("nan")
    out["conc_p"] = float((1 + (sc >= out["conc"]).sum()) / (len(sc) + 1)) if ok else float("nan")
    return out, order


# ---------------------------------------------------------------- pre-registered classifier
CLASSES = ["none", "shared round-robin (closed loop)", "shared round-robin (open-loop relay)", "per-worker round-robin",
           "random / shuffle", "least-busy", "sticky / episodic", "bunching"]


def classify(f, zR, z_thr=3.0, ring_z_thr=6.0, conc_z_thr=6.0):
    """f: fingerprint dict; zR: dict of z-scores of R-type statistics against the human null (positive = turn-taking).
    Thresholds are fixed in advance from simulation (fp/explore*.py), not tuned on the testbed."""
    z_full, z_short = zR.get("R", 0.0), zR.get("R5", 0.0)
    turn = max(z_full, zR.get("R20", 0.0)) > z_thr
    if f["ring_z"] > ring_z_thr:
        return CLASSES[1] if turn else CLASSES[2]
    if np.isfinite(f.get("conc_z", np.nan)) and f["conc_z"] > conc_z_thr:
        return CLASSES[3]
    if turn:
        r_full, r5 = f.get("R", 1.0), f.get("R5", 1.0)
        if (1 - r5) < 0.5 * (1 - r_full):
            return CLASSES[6]
        return CLASSES[5] if f["coll"] < 0.5 else CLASSES[4]
    if min(z_full, zR.get("R20", 0.0)) < -z_thr:
        return CLASSES[7]
    return CLASSES[0]


# ---------------------------------------------------------------- site-wide blind discovery
def site_handoff_graph(a, s, e, N, t0, t1, lag_hi=1.0, q=1e-3, min_count=4):
    """Symmetric key-pair successor excess with exact Poisson tails, Benjamini-Hochberg over all pairs.
    Returns component labels (-1 = not in any component) and the edge list."""
    from scipy.stats import poisson
    H, Ex = handoff_excess(a, s, e, N, t0, t1, lag_hi)
    np.fill_diagonal(H, 0); np.fill_diagonal(Ex, 0)
    Hs, Es = H + H.T, Ex + Ex.T
    iu = np.triu_indices(N, 1)
    obs, exp_ = Hs[iu], Es[iu]
    p = poisson.sf(obs - 1, np.maximum(exp_, 1e-12))
    p[obs < min_count] = 1.0
    m = len(p); o = np.argsort(p); thr = q * np.arange(1, m + 1) / m
    passed = np.zeros(m, bool)
    k = np.nonzero(p[o] <= thr)[0]
    if len(k): passed[o[: k.max() + 1]] = True
    I, J = iu[0][passed], iu[1][passed]
    parent = np.arange(N)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i, j in zip(I, J):
        ri, rj = find(i), find(j)
        if ri != rj: parent[ri] = rj
    roots = np.array([find(i) for i in range(N)])
    deg = np.bincount(np.r_[I, J], minlength=N)
    lab = np.where(deg > 0, roots, -1)
    return lab, list(zip(I.tolist(), J.tolist()))


# ================================================================ v0.4 additions (fixed after NB6, frozen for NB7)
def sideband_expect(starts_j, ends_i, w=1.0, B=60.0):
    """observed and local-rate expected count of key-j starts within (e, e+w] after key-i ends.
    Expectation uses j's own starts in [e-B, e+B] outside (e-w, e+w] (an off-source sideband), so bursty sessions cancel."""
    obs = (np.searchsorted(starts_j, ends_i + w, "right") - np.searchsorted(starts_j, ends_i, "right")).sum()
    side = (np.searchsorted(starts_j, ends_i + B, "right") - np.searchsorted(starts_j, ends_i - B, "left")).sum() - \
           (np.searchsorted(starts_j, ends_i + w, "right") - np.searchsorted(starts_j, ends_i - w, "left")).sum()
    return float(obs), float(side) * w / (2 * B - 2 * w)


def site_handoff_graph_sideband(a, s, e, N, t0, t1, lag_hi=1.0, q=1e-3, min_count=4, B=60.0, alpha=1e-3):
    """Stage 1: Poisson screen with Benjamini-Hochberg (site_handoff_graph). Stage 2: keep an edge only if it also beats the
    local-rate sideband null (Bonferroni over screened edges). Returns component labels (-1 = none) and kept edges."""
    from scipy.stats import poisson
    _, edges = site_handoff_graph(a, s, e, N, t0, t1, lag_hi, q, min_count)
    st = {k: np.sort(s[a == k]) for k in set(np.ravel(edges).tolist())} if edges else {}
    en = {k: e[a == k] for k in st}
    kept = []
    for i, j in edges:
        o1, e1 = sideband_expect(st[j], en[i], lag_hi, B); o2, e2 = sideband_expect(st[i], en[j], lag_hi, B)
        if poisson.sf(o1 + o2 - 1, max(e1 + e2, 1e-9)) < alpha / len(edges):
            kept.append((i, j))
    parent = np.arange(N)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i, j in kept:
        parent[find(i)] = find(j)
    deg = np.bincount(np.ravel(kept).astype(int), minlength=N) if kept else np.zeros(N, int)
    return np.where(deg > 0, [find(i) for i in range(N)], -1), kept


def hollow_corr(a, s, e, N, t0, t1, dt=0.1, hp_block_s=20.0):
    """zero-lag high-pass occupancy correlation over all keys, site-wide common mode removed, diagonal set to 0"""
    X = highpass(occupancy_matrix(a, s, e, N, t0, t1, dt).astype(np.float64), int(hp_block_s / dt))
    sd = X.std(1); ok = sd > 0
    Z = np.zeros_like(X); Z[ok] = (X[ok] - X[ok].mean(1, keepdims=True)) / sd[ok, None]
    g = Z[ok].mean(0); g /= np.linalg.norm(g) + 1e-12
    Z[ok] -= np.outer(Z[ok] @ g, g)
    C = Z @ Z.T / Z.shape[1]; np.fill_diagonal(C, 0.0)
    return C


def spectral_rings(a, s, e, N, t0, t1, rng, n_null=3, edge_factor=1.05, pair_tol=0.9, load_k=2.5):
    """Rings as outlier eigen-pairs of the hollowed correlation matrix above a time-shifted null bulk.
    Returns labels (-1 = none), list of eigen-groups with eigenvalues, the bulk edge, and per-group circular order."""
    C = hollow_corr(a, s, e, N, t0, t1)
    w, V = np.linalg.eigh(C)
    edge = edge_factor * max(np.linalg.eigvalsh(hollow_corr(*shift_keys(a, s, e, t0, t1, rng, N), N, t0, t1))[-1] for _ in range(n_null))
    out = np.nonzero(w > edge)[0][::-1]
    groups, i = [], 0
    while i < len(out):
        if i + 1 < len(out) and w[out[i + 1]] > pair_tol * w[out[i]]:
            groups.append([int(out[i]), int(out[i + 1])]); i += 2
        else:
            groups.append([int(out[i])]); i += 1
    lab = -np.ones(N, int); orders = []
    if groups:
        L = np.stack([np.sqrt((V[:, g] ** 2).sum(1)) for g in groups], 1)
        best, strength = L.argmax(1), L.max(1)
        lab = np.where(strength > load_k / np.sqrt(N), best, -1)
        for gi, g in enumerate(groups):
            mem = np.nonzero(lab == gi)[0]
            orders.append(mem[np.argsort(np.arctan2(V[mem, g[1]], V[mem, g[0]]))] if len(g) == 2 else mem)
    return lab, [[float(w[j]) for j in g] for g in groups], float(edge), orders


# ---------------------------------------------------------------- active timing watermark (randomisation test)
def watermark_echo(a, s, c, b, src_keys, dst_keys, w=1.0, n_perm=2000, rng=None, same_key=False):
    """Provider delayed the end of each response by b*delta (b = secret fair coin). For every request on a source key,
    take the mean lag, from its true completion c, of starts on destination keys within [c, c+w) (other keys only, or the
    same key if same_key). Statistic T = mean lag | b=1 minus mean lag | b=0. Exact p-value by re-drawing the coins."""
    rng = rng or np.random.default_rng(0)
    src = np.isin(a, src_keys); dst = np.isin(a, dst_keys)
    o = np.argsort(s[dst]); ss, sa = s[dst][o], a[dst][o]
    idx = np.nonzero(src)[0]
    i0 = np.searchsorted(ss, c[idx], "left"); i1 = np.searchsorted(ss, c[idx] + w, "left")
    L = np.full(len(idx), np.nan)
    for n, (k, x, y) in enumerate(zip(idx, i0, i1)):
        if y > x:
            m = (sa[x:y] == a[k]) if same_key else (sa[x:y] != a[k])
            if m.any():
                L[n] = (ss[x:y][m] - c[k]).mean()
    ok = np.isfinite(L); L, bb = L[ok], b[idx][ok]
    if (bb == 1).sum() < 10 or (bb == 0).sum() < 10:
        return dict(n=int(ok.sum()), shift_ms=float("nan"), z=float("nan"), p=float("nan"))
    T = L[bb == 1].mean() - L[bb == 0].mean()
    perm = np.empty(n_perm)
    for r in range(n_perm):
        q = rng.permutation(bb); perm[r] = L[q == 1].mean() - L[q == 0].mean()
    return dict(n=int(ok.sum()), shift_ms=float(T * 1000), z=float((T - perm.mean()) / (perm.std() + 1e-12)),
                p=float((1 + (perm >= T).sum()) / (n_perm + 1)))
