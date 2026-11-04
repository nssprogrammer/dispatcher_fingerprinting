import sys
exec(open(sys.argv[1]).read())
res = []
for sd in range(10):
    labs, ntr = fplib.spectral_rings_trimmed_v08(A, S, E, N, t0, t1, np.random.default_rng(100 + sd))
    m = grp == 'F6_rrshared'; sel = labs >= 0
    others = sorted(set(grp[sel & ~m]) - {'hum'})
    res.append((int((sel & m).sum()), int((sel & (grp == 'hum')).sum()), others))
print(sys.argv[1], res)
