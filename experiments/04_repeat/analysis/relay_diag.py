exec(open('load7.py').read())
rng = np.random.default_rng(1)
for g in ['P7_relay_random', 'P8_relay_rr', 'P9_pinned']:
    m = (d.grp == g).values; a = A[m]; s, c, e, b = S[m], Cc[m], E[m], Bb[m]
    o = np.argsort(s); ss, sa = s[o], a[o]
    # lags of other-key starts relative to the SEND time e (when the client gets the final chunk), fine bins
    edges = np.array([0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
    res = {}
    for bit in (0, 1):
        idx = np.nonzero(b == bit)[0]; H = np.zeros(len(edges) - 1)
        for k in idx:
            x, y = np.searchsorted(ss, e[k]), np.searchsorted(ss, e[k] + 1.0)
            l = ss[x:y][sa[x:y] != a[k]] - e[k]
            H += np.histogram(l, edges)[0]
        lam = len(s) / (t1 - t0); expv = lam * np.diff(edges) * len(idx) * (1 - 1 / 40)
        res[bit] = np.round(H / expv, 2)
    print(g, "ratio observed/expected of other-key starts after the end-of-stream, bins (s):", edges[1:].tolist())
    print("   b=0:", res[0].tolist()); print("   b=1:", res[1].tolist())
    # stability: the echo statistic in the first and second half of the run
    ks = np.unique(a)
    for lo, hi in ((t0, (t0 + t1) / 2), ((t0 + t1) / 2, t1)):
        mm = (c >= lo) & (c < hi)
        r = fplib.watermark_echo(a[mm], s[mm], c[mm], b[mm], ks, ks, 1.0, 2000, rng)
        print(f"   half [{lo - t0:.0f},{hi - t0:.0f}) shift {r['shift_ms']:.1f} ms z {r['z']:.2f} p {r['p']:.4f}")
