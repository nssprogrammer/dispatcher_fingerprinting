"""fplib — dispatcher fingerprinting from per-key request logs (start, end) only.

Signals (all computed without labels):
  ring     zero-lag high-pass occupancy correlation arranged on a recovered circle (shared round-robin counter)
  conc     concentration of each key's short-lag end->start successors (per-worker round-robin counters)
  coll     same-key overlap relative to uniform spreading of concurrent requests (least-busy / round-robin avoid collisions)
  R(tau)   occupancy variance ratio at several timescales (closed-loop turn-taking, sticky/episodic key holding)
  K_little worker count from the interactive response-time law, think time read off the successor-lag excess
"""
import numpy as np

__version__ = "fp-0.8"


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


def coverage_expect(starts_j, ends_i, w=1.0, B=60.0):
    """v0.8 null for one direction of a hand-off edge. obs = number of DISTINCT key-j starts that fall within (e, e+w] after
    some key-i end. exp = sum over j's starts of the fraction of the local sideband [t-B, t+B] minus (t-w, t+w] that is covered
    by the union of i's windows (e, e+w]. Conditioning on where i's ends actually are makes bursts of ends (a key held until it
    is refused) count once, which the end-to-start pair count of sideband_expect did not."""
    ends = np.sort(ends_i); st = np.sort(starts_j)
    if len(st) == 0 or len(ends) == 0: return 0.0, 0.0
    lo, hi = ends, ends + w                                     # union of windows
    merged = []
    for a_, b_ in zip(lo, hi):
        if merged and a_ <= merged[-1][1]: merged[-1][1] = max(merged[-1][1], b_)
        else: merged.append([a_, b_])
    M_ = np.array(merged); L, R = M_[:, 0], M_[:, 1]; cum = np.r_[0, np.cumsum(R - L)]
    def covered(x0, x1):                                        # covered length of [x0, x1]
        def F(x):
            k = np.searchsorted(L, x, "right")
            return cum[k] - np.where(k > 0, np.maximum(R[np.maximum(k - 1, 0)] - x, 0), 0)
        return np.maximum(F(x1) - F(x0), 0)
    k = np.searchsorted(L, st, "left") - 1                      # is each start inside a window (L, R]?
    inside = (k >= 0) & (st > L[np.maximum(k, 0)]) & (st <= R[np.maximum(k, 0)])
    cov = (covered(st - B, st + B) - covered(st - w, st + w)) / (2 * B - 2 * w)
    return float(inside.sum()), float(np.clip(cov, 0, 1).sum())


def site_handoff_graph_coverage(a, s, e, N, t0, t1, lag_hi=1.0, q=1e-3, min_count=4, B=60.0, alpha=1e-3, family="all"):
    """v0.8 hand-off graph: same Poisson/BH screen as v0.4, then each edge must beat the coverage null in both directions
    combined. Bonferroni is over ALL N(N-1)/2 pairs (family="all"): the screen used the same counts, so correcting only over
    screened edges (as v0.4 did) double-dips."""
    from scipy.stats import poisson
    _, edges = site_handoff_graph(a, s, e, N, t0, t1, lag_hi, q, min_count)
    st = {k: np.sort(s[a == k]) for k in set(np.ravel(edges).tolist())} if edges else {}
    en = {k: e[a == k] for k in st}
    kept = []
    for i, j in edges:
        o1, e1 = coverage_expect(st[j], en[i], lag_hi, B); o2, e2 = coverage_expect(st[i], en[j], lag_hi, B)
        fam = N * (N - 1) / 2 if family == "all" else len(edges)
        if poisson.sf(o1 + o2 - 1, max(e1 + e2, 1e-9)) < alpha / fam:
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


def site_handoff_graph_split(a, s, e, N, t0, t1, lag_hi=1.0, q=1e-3, min_count=4, B=60.0, alpha=1e-3):
    """v0.8 split-sample hand-off graph: screen edges on the first half of the window, test them with the coverage null on
    the second half only (Bonferroni over screened edges is then valid, because the test data were not used to select)."""
    from scipy.stats import poisson
    tm = 0.5 * (t0 + t1)
    m1 = s < tm
    _, edges = site_handoff_graph(a[m1], s[m1], np.minimum(e[m1], tm), N, t0, tm, lag_hi, q, min_count)
    m2 = s >= tm
    a2, s2, e2 = a[m2], s[m2], e[m2]
    st = {k: np.sort(s2[a2 == k]) for k in set(np.ravel(edges).tolist())} if edges else {}
    en = {k: e2[a2 == k] for k in st}
    kept = []
    for i, j in edges:
        o1, x1 = coverage_expect(st[j], en[i], lag_hi, B); o2, x2 = coverage_expect(st[i], en[j], lag_hi, B)
        if poisson.sf(o1 + o2 - 1, max(x1 + x2, 1e-9)) < alpha / len(edges):
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


def collapse_bursts(t, w=1.0):
    """keep the first of each run of events closer than w to the previous one (a burst counts once)"""
    t = np.sort(t)
    if len(t) == 0: return t
    return t[np.r_[True, np.diff(t) > w]]


def site_handoff_graph_v09(a, s, e, N, t0, t1, lag_hi=1.0, q=1e-3, min_count=4, B=60.0, alpha=1e-3):
    """v0.9 (post hoc, after live NB9): as v0.8, but j's starts are collapsed to bursts before the coverage test, so a stint of
    simultaneous failover starts on one key counts once. (v0.8 already merges i's ends into union windows.)"""
    from scipy.stats import poisson
    _, edges = site_handoff_graph(a, s, e, N, t0, t1, lag_hi, q, min_count)
    st = {k: collapse_bursts(s[a == k], lag_hi) for k in set(np.ravel(edges).tolist())} if edges else {}
    en = {k: e[a == k] for k in st}
    kept = []; fam = N * (N - 1) / 2
    for i, j in edges:
        o1, e1 = coverage_expect(st[j], en[i], lag_hi, B); o2, e2 = coverage_expect(st[i], en[j], lag_hi, B)
        if poisson.sf(o1 + o2 - 1, max(e1 + e2, 1e-9)) < alpha / fam:
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


def failover_graph(a, s, a429, t429, N, w=0.3, B=30.0, alpha=1e-3, min_count=3):
    """Site-wide failover graph (post hoc, after NB9): link key i -> j when j's starts follow i's REFUSALS (HTTP 429, issued by
    the provider) within w seconds more often than j's local start rate predicts (+-B s sideband). Bonferroni over all
    ordered pairs (i with a refusal, any j). Only the first start of j per refusal counts."""
    from scipy.stats import poisson
    order = np.argsort(s); ss, aa = s[order], a[order]
    st = {k: np.sort(s[a == k]) for k in range(N)}
    keys429 = np.unique(a429); fam = max(len(keys429) * (N - 1), 1); kept = []
    for i in keys429:
        ti = np.sort(t429[a429 == i]); cnt = {}
        for t in ti:
            x, y = np.searchsorted(ss, t, "right"), np.searchsorted(ss, t + w, "right")
            for j in set(aa[x:y].tolist()) - {i}: cnt[j] = cnt.get(j, 0) + 1
        for j, o in cnt.items():
            if o < min_count: continue
            sj = st[j]
            side = (np.searchsorted(sj, ti + B) - np.searchsorted(sj, ti - B)).sum() - (np.searchsorted(sj, ti + w) - np.searchsorted(sj, ti - w)).sum()
            exp_ = max(side * w / (2 * B - 2 * w), 1e-9)
            if poisson.sf(o - 1, exp_) < alpha / fam: kept.append((int(i), int(j)))
    parent = np.arange(N)
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i, j in kept: parent[find(i)] = find(j)
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


def spectral_rings(a, s, e, N, t0, t1, rng, n_null=3, edge_factor=1.05, pair_tol=0.9, load_k=2.5, hp_block_s=20.0):
    """Rings as outlier eigen-pairs of the hollowed correlation matrix above a time-shifted null bulk.
    Returns labels (-1 = none), list of eigen-groups with eigenvalues, the bulk edge, and per-group circular order."""
    C = hollow_corr(a, s, e, N, t0, t1, hp_block_s=hp_block_s)
    w, V = np.linalg.eigh(C)
    tops = []
    for _ in range(n_null):
        C_null = hollow_corr(*shift_keys(a, s, e, t0, t1, rng, N), N, t0, t1, hp_block_s=hp_block_s)
        tops.append(np.linalg.eigvalsh(C_null)[-1])
    edge = edge_factor * max(tops)
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
        # v0.5: one ring can give several outlier pairs (its Fourier harmonics), so keys are not assigned to eigen-groups.
        # Select keys that load on any outlier group, then split them into rings by strong pairwise correlation
        # (neighbours on a ring correlate at ~0.5; unrelated keys at ~0 +- sigma), threshold max(0.2, 6 sigma_null).
        import scipy.sparse as _sp, scipy.sparse.csgraph as _cg
        sel = np.nonzero(L.max(1) > load_k / np.sqrt(N))[0]
        iu = np.triu_indices(N, 1)
        thr = max(0.2, 6.0 * float(np.sqrt(np.mean(C_null[iu] ** 2))))
        if len(sel):
            Cs = C[np.ix_(sel, sel)]
            ncomp, cl = _cg.connected_components(_sp.csr_matrix(Cs > thr), directed=False)
            nxt = 0
            for c in range(ncomp):
                mem = sel[cl == c]
                if len(mem) >= 5:
                    lab[mem] = nxt; nxt += 1
                    orders.append(mem[circular_seriation(C[np.ix_(mem, mem)])])
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


# ================================================================ v0.6 (defined after the NB8 dry run, before any NB8 data on vLLM)
# A ring whose period T_c = M / lambda exceeds the high-pass block is removed with the slow trend. Long answers make T_c long,
# so ring statistics are computed at several block lengths; v0.5 (20 s only) stays as the frozen reference.
SCALES = (20.0, 60.0, 180.0)


def ring_multiscale(a, s, e, M, t0, t1, rng, n_sur=40, scales=SCALES):
    """max over scales of the ring z (surrogates per scale); returns (z_max, best scale, order at that scale)"""
    best = (-np.inf, None, None)
    for sc in scales:
        if (t1 - t0) < 4 * sc: continue
        f, order = fingerprint(a, s, e, M, t0, t1, rng, n_sur=n_sur, hp_block_s=sc)
        if f["ring_z"] > best[0]: best = (f["ring_z"], sc, order)
    return best


def spectral_rings_multiscale(a, s, e, N, t0, t1, rng, scales=SCALES, **kw):
    """union of spectral_rings over block lengths; rings found at several scales are merged when they share keys"""
    labs = []
    for sc in scales:
        if (t1 - t0) < 4 * sc: continue
        lab, groups, edge, orders = spectral_rings(a, s, e, N, t0, t1, rng, hp_block_s=sc, **kw)
        labs.append(lab)
    parent = {}
    def find(x):
        while parent.setdefault(x, x) != x: x = parent[x]
        return x
    for li, lab in enumerate(labs):
        for c in np.unique(lab[lab >= 0]):
            mem = np.nonzero(lab == c)[0]
            for k in mem: parent[find(("k", int(k)))] = find(("r", li, int(c)))
    out = -np.ones(N, int); roots = {}
    for k in range(N):
        if ("k", k) in parent:
            r = find(("k", k)); out[k] = roots.setdefault(r, len(roots))
    return out


# ================================================================ v0.7 (defined after NB8, frozen for NB9)
def key_corr_time(a, s, e, N, t0, t1, dt=0.5, hp_block_s=20.0, max_lag_s=120.0):
    """integrated squared autocorrelation time of each key's high-pass occupancy (seconds)"""
    X = highpass(occupancy_matrix(a, s, e, N, t0, t1, dt).astype(np.float64), int(hp_block_s / dt))
    X = X - X.mean(1, keepdims=True); sd = X.std(1); sd[sd == 0] = 1; Z = X / sd[:, None]
    n = Z.shape[1]; F = np.fft.rfft(Z, 2 * n); ac = np.fft.irfft(F * np.conj(F), 2 * n)[:, : int(max_lag_s / dt)] / n
    return dt * (2 * (ac ** 2).sum(1) - ac[:, 0] ** 2)


def spectral_rings_trimmed(a, s, e, N, t0, t1, rng, trim_k=5.0, scales=SCALES, **kw):
    """v0.7: drop keys whose correlation time is a robust outlier (median + trim_k * MAD), then the v0.6 multi-scale scan.
    Long-memory keys (e.g. a key held by one worker for many requests) inflate the null bulk edge and hide rings."""
    tau = key_corr_time(a, s, e, N, t0, t1)
    active = np.bincount(a, minlength=N) > 0
    med = np.median(tau[active]); mad = 1.4826 * np.median(np.abs(tau[active] - med))
    keep = active & (tau <= med + trim_k * mad)
    sel = np.nonzero(keep)[0]; mp = -np.ones(N, int); mp[sel] = np.arange(len(sel))
    m = keep[a]
    lab_sub = spectral_rings_multiscale(mp[a[m]], s[m], e[m], len(sel), t0, t1, rng, scales=scales, **kw)
    lab = -np.ones(N, int); lab[sel] = lab_sub
    return lab, int((~keep & active).sum())


def key_persist_time(a, s, e, N, t0, t1, dt=0.5, hp_block_s=20.0, max_lag_s=120.0):
    """v0.8: persistence time of each key's high-pass occupancy = autocorrelation integrated up to its FIRST zero crossing (s).
    A key held by one worker (long positive memory) scores high; a key visited periodically by a ring oscillates and
    scores low, whereas the integrated SQUARED autocorrelation of v0.7 counted every periodic peak and trimmed rings."""
    X = highpass(occupancy_matrix(a, s, e, N, t0, t1, dt).astype(np.float64), int(hp_block_s / dt))
    X = X - X.mean(1, keepdims=True); sd = X.std(1); sd[sd == 0] = 1; Z = X / sd[:, None]
    n = Z.shape[1]; F = np.fft.rfft(Z, 2 * n); ac = np.fft.irfft(F * np.conj(F), 2 * n)[:, : int(max_lag_s / dt)] / n
    neg = ac <= 0; first = np.where(neg.any(1), neg.argmax(1), ac.shape[1])
    mask = np.arange(ac.shape[1])[None, :] < first[:, None]
    return dt * (2 * (ac * mask).sum(1) - ac[:, 0])


def spectral_rings_trimmed_v08(a, s, e, N, t0, t1, rng, trim_k=5.0, scales=SCALES, **kw):
    """v0.8: as v0.7, but trims on key_persist_time (robust outliers, median + trim_k * MAD)."""
    tau = key_persist_time(a, s, e, N, t0, t1)
    active = np.bincount(a, minlength=N) > 0
    med = np.median(tau[active]); mad = 1.4826 * np.median(np.abs(tau[active] - med))
    keep = active & (tau <= med + trim_k * mad)
    sel = np.nonzero(keep)[0]; mp = -np.ones(N, int); mp[sel] = np.arange(len(sel))
    m = keep[a]
    lab_sub = spectral_rings_multiscale(mp[a[m]], s[m], e[m], len(sel), t0, t1, rng, scales=scales, **kw)
    lab = -np.ones(N, int); lab[sel] = lab_sub
    return lab, int((~keep & active).sum())


def start_count_ratio(a, s, M, t0, t1, bin_s=2.0, block_s=60.0):
    """variance ratio of request START counts (durations never used, so server cross-talk cannot enter)"""
    nb = int((t1 - t0) / bin_s); C = np.zeros((M, nb))
    b = ((s - t0) / bin_s).astype(int); ok = (b >= 0) & (b < nb)
    np.add.at(C, (a[ok], b[ok]), 1)
    k = int(block_s / bin_s); nb2 = (nb // k) * k
    C = C[:, :nb2].reshape(M, -1, k); C = (C - C.mean(2, keepdims=True)).reshape(M, -1)
    den = C.var(1).sum()
    return float(C.sum(0).var() / den) if den > 0 else float("nan")


def classify_v07(f, zR, z_start, z_thr=3.0, **kw):
    """v0.6 rule, except that 'bunching' must be confirmed on start counts (z_start < -z_thr); otherwise 'none'"""
    c = classify(f, zR, z_thr=z_thr, **kw)
    if c == CLASSES[7] and not (np.isfinite(z_start) and z_start < -z_thr):
        return CLASSES[0]
    return c
