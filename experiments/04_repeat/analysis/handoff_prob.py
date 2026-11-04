exec(open('load7.py').read())
delta, W = 0.155, 3.0
rng = np.random.default_rng(0)
def est(g):
    m = (d.grp == g).values; a = A[m]; s, c, b = S[m], Cc[m], Bb[m]
    o = np.argsort(s); ss, sa = s[o], a[o]
    same = np.zeros(len(c)); cross = np.zeros(len(c))
    for k in range(len(c)):
        x, y = np.searchsorted(ss, c[k], 'right'), np.searchsorted(ss, c[k] + W)
        l = ss[x:y] - c[k]; sk = sa[x:y] == a[k]
        same[k] = np.clip(W - l[sk], 0, None).sum(); cross[k] = np.clip(W - l[~sk], 0, None).sum()
    out = {}
    for name, v in (("same", same), ("cross", cross)):
        q = (v[b == 0].mean() - v[b == 1].mean()) / delta
        boot = [(lambda i: (v[i][b[i] == 0].mean() - v[i][b[i] == 1].mean()) / delta)(rng.integers(0, len(v), len(v))) for _ in range(300)]
        out[name] = f"{q:.2f} ± {np.std(boot):.2f}"
    return out
for g in ['P9_pinned', 'bat', 'P1_shuffle', 'P3_rrshared', 'P4_rrworker', 'C_company', 'P2_least', 'P7_relay_random']:
    print(f"{g:16s}", est(g))
