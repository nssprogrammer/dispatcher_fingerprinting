"""
occlib — occupancy statistics for detecting hidden closed-loop worker pools behind many accounts.

Data model: a request log as three equal-length arrays
    acct : int   account index 0..A-1
    s    : float request arrival time (seconds)
    e    : float time the streamed response finished (seconds), e >= s
X_i(t) = number of account i's requests in flight; N(t) = sum_i X_i(t).

All statistics are exact time integrals of step functions (no time grid).
"""
import numpy as np

__version__ = "0.1.0"

# --------------------------------------------------------------------------------------------
# core: exact per-group integrals of X and X^2
# --------------------------------------------------------------------------------------------
def _group_integrals(gid, s, e, ngroups):
    """For intervals carrying a group id, return I1[g] = int X_g dt and I2[g] = int X_g^2 dt,
    where X_g(t) = number of the group's intervals covering t (half-open [s, e))."""
    I1 = np.zeros(ngroups); I2 = np.zeros(ngroups)
    n = len(s)
    if n == 0:
        return I1, I2
    t = np.concatenate([s, e])
    d = np.concatenate([np.ones(n), -np.ones(n)])
    g = np.concatenate([gid, gid])
    o = np.lexsort((d, t, g))          # by group, then time, then ends before starts
    t, d, g = t[o], d[o], g[o]
    x = np.cumsum(d)                    # each group nets to zero, so the level resets between groups
    dt = np.empty_like(t)
    dt[:-1] = t[1:] - t[:-1]
    dt[-1] = 0.0
    dt[:-1][g[1:] != g[:-1]] = 0.0
    I1 += np.bincount(g, weights=x * dt, minlength=ngroups)
    I2 += np.bincount(g, weights=x * x * dt, minlength=ngroups)
    return I1, I2


def _clip(acct, s, e, t0, t1):
    s = np.clip(np.asarray(s, float), t0, t1); e = np.clip(np.asarray(e, float), t0, t1)
    m = e > s
    return np.asarray(acct)[m], s[m], e[m]


def _split_blocks(acct, s, e, t0, tau, nb):
    """Split intervals at block boundaries t0 + k*tau. Returns (acct, block, s, e) pieces."""
    b0 = np.floor((s - t0) / tau).astype(np.int64)
    b1 = np.floor((np.nextafter(e, -np.inf) - t0) / tau).astype(np.int64)
    b0 = np.clip(b0, 0, nb - 1); b1 = np.clip(b1, 0, nb - 1)
    cnt = b1 - b0 + 1
    idx = np.repeat(np.arange(len(s)), cnt)
    off = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt)
    blk = b0[idx] + off
    ps = np.maximum(s[idx], t0 + blk * tau)
    pe = np.minimum(e[idx], t0 + (blk + 1) * tau)
    m = pe > ps
    return acct[idx][m], blk[m], ps[m], pe[m]


def occupancy(acct, s, e, t0, t1, tau=None, n_accounts=None):
    """Occupancy moments over [t0, t1].
    tau=None -> full-window variances; tau=float -> mean over blocks of within-block variances
    (the 'high-pass' statistic). Returns dict with R, V_N, V_i (per account), mean_N, n_blocks."""
    acct = np.asarray(acct, np.int64)
    A = int(n_accounts if n_accounts is not None else (acct.max() + 1 if len(acct) else 0))
    if tau is None:
        tau = t1 - t0
    nb = int(np.floor((t1 - t0) / tau + 1e-9))
    if nb < 1:
        raise ValueError("window shorter than one block")
    t1 = t0 + nb * tau
    a, s, e = _clip(acct, s, e, t0, t1)
    a, b, s, e = _split_blocks(a, s, e, t0, tau, nb)
    # aggregate N, per block
    I1, I2 = _group_integrals(b, s, e, nb)
    VN_b = I2 / tau - (I1 / tau) ** 2
    # per account per block (compress group ids)
    key = a * nb + b
    uk, inv = np.unique(key, return_inverse=True)
    J1, J2 = _group_integrals(inv, s, e, len(uk))
    Vg = J2 / tau - (J1 / tau) ** 2
    V_i = np.bincount((uk // nb).astype(np.int64), weights=Vg, minlength=A) / nb
    V_N = VN_b.mean()
    sV = V_i.sum()
    return dict(R=V_N / sV if sV > 0 else np.nan, V_N=V_N, V_i=V_i, sumV=sV,
                mean_N=I1.sum() / (nb * tau), n_blocks=nb, tau=tau)


def R_full(acct, s, e, t0, t1):
    return occupancy(acct, s, e, t0, t1, None)["R"]


def R_hp(acct, s, e, t0, t1, tau=60.0):
    return occupancy(acct, s, e, t0, t1, tau)["R"]


def fano_khat(acct, s, e, t0, t1):
    """Fano factor of N and the implied effective number of hidden slots K = <N>^2/(<N>-Var N)."""
    o = occupancy(acct, s, e, t0, t1, None)
    F = o["V_N"] / o["mean_N"] if o["mean_N"] > 0 else np.nan
    p = 1 - F
    K = o["mean_N"] / p if p > 0 else np.inf
    return dict(F=F, p_hat=p, K_hat=K, mean_N=o["mean_N"], V_N=o["V_N"])


def baselines(acct, s, e, t0, t1):
    """Simple baselines: requests/hour, peak concurrency, Fano factor of N."""
    a, sc, ec = _clip(acct, s, e, t0, t1)
    n_req = int(((np.asarray(s) >= t0) & (np.asarray(s) < t1)).sum())
    t = np.concatenate([sc, ec]); d = np.concatenate([np.ones(len(sc)), -np.ones(len(ec))])
    o = np.lexsort((d, t)); peak = int(np.cumsum(d[o]).max()) if len(t) else 0
    f = fano_khat(acct, s, e, t0, t1)
    return dict(req_per_h=n_req / ((t1 - t0) / 3600), peak_N=peak, fano=f["F"])


# --------------------------------------------------------------------------------------------
# nulls
# --------------------------------------------------------------------------------------------
def circular_shift(acct, s, e, t0, t1, rng):
    """Shift each account's timeline by an independent uniform offset, wrapping around [t0, t1).
    Keeps each account's marginal behaviour and autocorrelation; destroys cross-account dependence."""
    acct = np.asarray(acct, np.int64); L = t1 - t0
    a, s, e = _clip(acct, s, e, t0, t1)
    A = int(acct.max() + 1) if len(acct) else 0
    off = rng.uniform(0, L, A)[a]
    s2 = t0 + np.mod(s - t0 + off, L); e2 = s2 + (e - s)
    wrap = e2 > t1
    a_out = np.concatenate([a, a[wrap]])
    s_out = np.concatenate([s2, np.full(wrap.sum(), t0)])
    e_out = np.concatenate([np.minimum(e2, t1), t0 + (e2[wrap] - t1)])
    return a_out, s_out, e_out


def surrogate_null(acct, s, e, t0, t1, tau, n_sur, rng):
    """Distribution of R (tau=None) or R_tau under independent circular shifts."""
    return np.array([occupancy(*circular_shift(acct, s, e, t0, t1, rng), t0, t1, tau)["R"] for _ in range(n_sur)])


# --------------------------------------------------------------------------------------------
# pool growth: correlation of each candidate's occupancy with a pool's aggregate occupancy
# --------------------------------------------------------------------------------------------
def _cum_integral(s, e, t0, t1):
    """Knots and running integral F(t) = int_t0^t N(u) du for the superposition of intervals."""
    t = np.concatenate([[t0], s, e, [t1]])
    d = np.concatenate([[0.0], np.ones(len(s)), -np.ones(len(e)), [0.0]])
    o = np.lexsort((d, t)); t, d = t[o], d[o]
    x = np.cumsum(d)
    F = np.concatenate([[0.0], np.cumsum(x[:-1] * np.diff(t))])
    return t, F


def corr_with_pool(pool, cand, t0, t1, tau=None, n_cand=None):
    """pool=(acct,s,e) of pool members; cand=(acct,s,e) of candidates (acct = candidate index 0..C-1).
    Returns per-candidate correlation of X_j with N_pool (within-block if tau is set). Linear in traffic."""
    pa, ps, pe = _clip(*pool, t0, t1)
    ca, cs, ce = _clip(*cand, t0, t1)
    C = int(n_cand if n_cand is not None else np.asarray(cand[0]).max() + 1)
    if tau is None:
        tau = t1 - t0
    nb = int(np.floor((t1 - t0) / tau + 1e-9)); t1 = t0 + nb * tau
    # pool per-block mean and within-block variance
    _, pb, pps, ppe = _split_blocks(pa, np.clip(ps, t0, t1), np.clip(pe, t0, t1), t0, tau, nb)
    I1, I2 = _group_integrals(pb, pps, ppe, nb)
    Nbar = I1 / tau; VN = (I2 / tau - Nbar ** 2).mean()
    kn, F = _cum_integral(np.clip(ps, t0, t1), np.clip(pe, t0, t1), t0, t1)
    # candidate pieces
    ca2, cb, cs2, ce2 = _split_blocks(ca, np.clip(cs, t0, t1), np.clip(ce, t0, t1), t0, tau, nb)
    XN = np.interp(ce2, kn, F) - np.interp(cs2, kn, F)       # int over piece of N_pool
    key = ca2 * nb + cb
    uk, inv = np.unique(key, return_inverse=True)
    A = np.bincount(inv, weights=XN)                          # int X_j N over block
    B = np.bincount(inv, weights=ce2 - cs2)                   # int X_j over block
    jj = uk // nb; bb = uk % nb
    cov = np.bincount(jj, weights=A / tau - (B / tau) * Nbar[bb], minlength=C) / nb
    Vj = occupancy(ca, cs, ce, t0, t1, None if nb == 1 else tau, n_accounts=C)["V_i"]
    with np.errstate(invalid="ignore", divide="ignore"):
        return cov / np.sqrt(Vj * VN)


# --------------------------------------------------------------------------------------------
# generators (simulation)
# --------------------------------------------------------------------------------------------
DUR_MED, DUR_SIG = 8.0, 0.6       # streamed response duration, log-normal
GAP_MED, GAP_SIG = 0.4, 0.5       # client processing gap, log-normal
E_DUR = DUR_MED * np.exp(DUR_SIG ** 2 / 2)
E_GAP = GAP_MED * np.exp(GAP_SIG ** 2 / 2)


def duty(J):
    return E_DUR / (E_DUR + E_GAP + J / 2.0)


def theory_R(p, M):
    return (1 - p) / (1 - p / M)


def _pack(rows):
    if not rows:
        return np.zeros(0, np.int64), np.zeros(0), np.zeros(0)
    a, s, e = map(np.asarray, zip(*rows))
    return a.astype(np.int64), s.astype(float), e.astype(float)


def benign_humans(rng, n_accounts, T, session_rate=1 / 7200, mean_req=6, think_med=25.0, think_sig=0.9,
                  diurnal=0.0, offset=0):
    """Human-like accounts: Poisson session starts, geometric #requests, log-normal think time.
    diurnal in [0,1): amplitude of a 24 h sinusoidal rate modulation (common to all accounts)."""
    rows = []
    for i in range(n_accounts):
        t = rng.exponential(1 / session_rate)
        while t < T:
            if diurnal > 0 and rng.random() > (1 + diurnal * np.sin(2 * np.pi * t / 86400)) / (1 + diurnal):
                t += rng.exponential(1 / session_rate); continue
            tt = t
            for _ in range(rng.geometric(1 / mean_req)):
                if tt > T: break
                d = rng.lognormal(np.log(DUR_MED), DUR_SIG); rows.append((offset + i, tt, tt + d))
                tt += d + rng.lognormal(np.log(think_med), think_sig)
            t = tt + rng.exponential(1 / session_rate)
    return _pack(rows)


def closed_loop_pool(rng, T, K, M, J=0.0, assign="random", stint=None, K_bar=None, epoch=None, offset=0):
    """K closed-loop slots over M accounts. Each slot: send, wait for stream end, gap, uniform(0,J) delay.
    assign: 'random' (per request), 'pinned' (slot w -> account w % M), 'stint' (hold an account for `stint` s).
    K_bar/epoch: if set, the number of active slots is redrawn ~ Poisson(K_bar) (capped at K) every `epoch` s."""
    rows = []
    Kact = None
    if K_bar is not None:
        Kact = np.minimum(rng.poisson(K_bar, int(T // epoch) + 2), K)
    held = {}
    for w in range(K):
        t = rng.uniform(0, 5)
        while t < T:
            if Kact is not None:
                ei = int(t // epoch)
                if w >= Kact[ei]:
                    t = (ei + 1) * epoch + rng.uniform(0, 1); continue
            if assign == "pinned":
                a = w % M
            elif assign == "stint":
                k = (w, int(t // stint))
                if k not in held: held[k] = rng.integers(M)
                a = held[k]
            else:
                a = rng.integers(M)
            d = rng.lognormal(np.log(DUR_MED), DUR_SIG)
            rows.append((offset + a, t, t + d))
            t += d + rng.lognormal(np.log(GAP_MED), GAP_SIG) + (rng.uniform(0, J) if J > 0 else 0.0)
    return _pack(rows)


def open_loop_relay(rng, T, M, lam, policy="random", offset=0):
    """Open-loop Poisson arrivals (rate lam/s) dispatched over M accounts: random | rr | least."""
    import heapq
    rows = []; inflight = np.zeros(M, int); heap = []; t = 0.0; rr = 0
    while True:
        t += rng.exponential(1 / lam)
        if t > T: break
        while heap and heap[0][0] <= t:
            _, a = heapq.heappop(heap); inflight[a] -= 1
        if policy == "random": a = int(rng.integers(M))
        elif policy == "rr": a = rr % M; rr += 1
        else:
            m = inflight.min(); a = int(rng.choice(np.flatnonzero(inflight == m)))
        d = rng.lognormal(np.log(DUR_MED), DUR_SIG)
        rows.append((offset + a, t, t + d)); inflight[a] += 1; heapq.heappush(heap, (t + d, a))
    return _pack(rows)


def concat(*logs):
    return tuple(np.concatenate(x) for x in zip(*logs))


def relabel(acct):
    """Map arbitrary account ids to 0..A-1."""
    u, inv = np.unique(acct, return_inverse=True)
    return inv.astype(np.int64), u
