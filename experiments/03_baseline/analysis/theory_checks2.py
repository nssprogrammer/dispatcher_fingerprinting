exec(open('load.py').read())
rng = np.random.default_rng(1)
def overlap(tau, D1, D2):
    """overlap length of [0, D1] and [tau, tau + D2] (tau may be negative)"""
    return np.clip(np.minimum(D1, tau + D2) - np.maximum(0, tau), 0, None)
for g in ['P3_rrshared', 'P8_relay_rr', 'P5_rrjitter']:
    m = (d.grp == g).values; a = d.pos.values[m].astype(int); s, e = S[m], E[m]; M = 40; dur = e - s
    lam = len(s) / (t1 - t0); Tc = M / lam                 # one trip round the ring
    p = dur.mean() / Tc                                     # duty per key (expected on-time per cycle / cycle)
    X = fplib.highpass(fplib.occupancy_matrix(a, s, e, M, t0, t1, 0.1).astype(float), 200)
    C = fplib.corr_matrix(X); Dm = np.abs(np.subtract.outer(np.arange(M), np.arange(M))); Dm = np.minimum(Dm, M - Dm)
    prof = np.array([C[Dm == k].mean() for k in range(1, 21)])
    o = np.argsort(s); ss, aa = s[o], a[o]; pred = []
    for k in range(1, 21):
        lags = []
        for i in range(0, len(ss), 5):
            j = np.searchsorted(ss, ss[i], 'right'); nxt = np.nonzero(aa[j:j + 400] == (aa[i] + k) % M)[0]
            if len(nxt): lags.append(ss[j + nxt[0]] - ss[i])
        lags = np.array(lags); lags = np.where(lags > Tc / 2, lags - Tc, lags)
        D1 = rng.choice(dur, len(lags)); D2 = rng.choice(dur, len(lags))
        both = overlap(lags, D1, D2).mean() / Tc
        pred.append((both - p * p) / (p - p * p))
    pred = np.array(pred)
    print(f"{g}: duty p={p:.3f}  observed {np.round(prof[:7], 3)}\n{'':14s} predicted {np.round(pred[:7], 3)}  rmse={np.sqrt(np.mean((prof - pred) ** 2)):.3f}")
