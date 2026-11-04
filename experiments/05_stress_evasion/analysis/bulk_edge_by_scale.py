exec(open('load_evasion.py').read())
import importlib; importlib.reload(fplib)
rng = np.random.default_rng(1); N = len(keys); A, S, E = d.a.values.astype(np.int64), d.t_arrival.values, d.t_end.values
print("window", round(t1 - t0), "N", N)
for sc in (20.0, 60.0, 180.0):
    lab, groups, edge, orders = fplib.spectral_rings(A, S, E, N, t0, t1, np.random.default_rng(1), hp_block_s=sc)
    comp = {}
    for c in range(lab.max() + 1):
        u, n = np.unique(grp[lab == c], return_counts=True); comp[c] = dict(zip(u.tolist(), n.tolist()))
    print(f"scale {sc:5.0f}s edge {edge:.2f} groups {[[round(x,2) for x in g] for g in groups]}\n   rings: {list(comp.values())}")
lab6 = fplib.spectral_rings_multiscale(A, S, E, N, t0, t1, np.random.default_rng(1))
print("union:", [dict(zip(*[x.tolist() for x in np.unique(grp[lab6 == c], return_counts=True)])) for c in range(lab6.max() + 1)])
