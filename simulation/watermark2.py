import numpy as np, sys
sys.argv = ['x']; exec(open('watermark.py').read().split("print(f\"{'pool'")[0])
def wm_lag(a, s, c, b, w, d, n_perm=2000):
    """mean lag (from true completion c) of other-key starts within [c, c+w): shifts by ~d * (signal fraction) when b=1"""
    o = np.argsort(s); ss, sa = s[o], a[o]
    i0, i1 = np.searchsorted(ss, c), np.searchsorted(ss, c + w)
    L = np.full(len(c), np.nan)
    for k, (x, y) in enumerate(zip(i0, i1)):
        m = sa[x:y] != a[k]
        if m.any(): L[k] = (ss[x:y][m] - c[k]).mean()
    ok = np.isfinite(L); L, bb = L[ok], b[ok]
    T = L[bb == 1].mean() - L[bb == 0].mean()
    perm = np.array([(lambda q: L[q == 1].mean() - L[q == 0].mean())(rng.permutation(bb)) for _ in range(n_perm)])
    return T, (T - perm.mean()) / perm.std(), (1 + (perm >= T).sum()) / (n_perm + 1)
for d_ in (0.05, 0.2):
    globals()['delta'] = d_
    print(f"--- delay {int(d_*1000)} ms on half the responses (secret random bit)")
    for name, kw, w in [("closed loop, random keys", dict(kind="closed"), 0.8),
                        ("closed loop, random keys, 0-6 s jitter", dict(kind="closed", jitter=6.0), 8.0),
                        ("closed loop, PINNED keys", dict(kind="pinned", K=10), 0.8),
                        ("open-loop relay, random keys", dict(kind="open_loop"), 0.8),
                        ("relay of 150 humans (think 6 s)", dict(kind="humans_via_relay", K=150, think=6.0), 30.0)]:
        a, s, c, b = run(**kw); m = s > 60; a, s, c, b = a[m], s[m], c[m], b[m]
        T_, z, p = wm_lag(a, s, c, b, w, d_)
        print(f"   {name:40s} n={len(a):6d}  shift={T_*1000:6.1f} ms  z={z:5.1f}  p={p:.4f}")
