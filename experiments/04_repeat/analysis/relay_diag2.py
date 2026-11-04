exec(open('load7.py').read())
rng = np.random.default_rng(2)
edges = np.arange(0, 1.0001, 0.05)
for g in ['P7_relay_random', 'P8_relay_rr']:
    m = (d.grp == g).values; a = A[m]; s, c, b = S[m], Cc[m], Bb[m]
    o = np.argsort(s); ss, sa = s[o], a[o]; lam = len(s) / (t1 - t0)
    def hist(idx):
        H = np.zeros(len(edges) - 1)
        for k in idx:
            x, y = np.searchsorted(ss, c[k]), np.searchsorted(ss, c[k] + 1.0)
            H += np.histogram(ss[x:y][sa[x:y] != a[k]] - c[k], edges)[0]
        return H / (lam * 0.05 * len(idx) * (1 - 1 / 40))
    h1, h0 = hist(np.nonzero(b == 1)[0]), hist(np.nonzero(b == 0)[0])
    print(g); print("  b=1 - b=0 (ratio units, 50 ms bins):", np.round(h1 - h0, 2).tolist())
    # how many ends have zero other-key starts in the window (they drop out of the statistic), by bit
    cnt = np.array([np.sum(sa[np.searchsorted(ss, c[k]):np.searchsorted(ss, c[k] + 1)] != a[k]) for k in range(len(c))])
    print("  mean # starts in window b=1 %.3f  b=0 %.3f ; share with none b=1 %.3f b=0 %.3f" % (cnt[b == 1].mean(), cnt[b == 0].mean(), (cnt[b == 1] == 0).mean(), (cnt[b == 0] == 0).mean()))
    # is bit associated with the request's own properties? (duration, arrival-time position) -> should not be
    dur = c - s
    print("  mean duration b=1 %.3f b=0 %.3f ; n b=1 %d b=0 %d" % (dur[b == 1].mean(), dur[b == 0].mean(), (b == 1).sum(), (b == 0).sum()))
# multiplicity: p-values of all 13 pools x 4 tests under a global null would be uniform; how extreme is the min among the null-expected ones?
print("Bonferroni over ~52 watermark tests: 0.0017*52 =", round(0.0017 * 52, 3))
