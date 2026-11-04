exec(open('load7.py').read())
rng = np.random.default_rng(0); N = len(keys)
C = fplib.hollow_corr(A, S, E, N, t0, t1); np.save('C7.npy', C)
w, V = np.linalg.eigh(C)
lab, groups, edge, orders = fplib.spectral_rings(A, S, E, N, t0, t1, rng)
print("edge", round(edge, 2), "groups", [[round(x, 2) for x in g] for g in groups])
for gi in range(len(groups)):
    mem = grp[lab == gi]; u, n = np.unique(mem, return_counts=True); print("  group", gi, dict(zip(u.tolist(), n.tolist())))
# loadings of the P5 keys on each group
L = np.stack([np.sqrt((V[:, [len(w) - 1 - i for i in range(10)][j]] ** 2)) for j in range(10)], 1)
p5 = np.nonzero(grp == 'P5_rrjitter')[0]
print("P5 keys: top-10 eigenvector participation (sum of squared loadings):", np.round((V[p5][:, -10:] ** 2).sum(0)[::-1], 2))
