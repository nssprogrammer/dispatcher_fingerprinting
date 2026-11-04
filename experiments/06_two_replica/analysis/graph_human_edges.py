exec(open('load_balanced.py').read())
lab4, k4 = fplib.site_handoff_graph_sideband(A, S, E, N, t0, t1)
lab8, k8 = fplib.site_handoff_graph_coverage(A, S, E, N, t0, t1)
print("v0.4 humans", int(((lab4 >= 0) & (grp == 'hum')).sum()), "edges", len(k4))
print("v0.8 humans", int(((lab8 >= 0) & (grp == 'hum')).sum()), "edges", len(k8))
from scipy.stats import poisson
st = {k: np.sort(S[A == k]) for k in range(N)}; en = {k: E[A == k] for k in range(N)}
for i, j in k8:
    if 'hum' in (grp[i], grp[j]):
        h, o = (i, j) if grp[i] == 'hum' else (j, i)
        o1, e1 = fplib.coverage_expect(st[h], en[o]); o2, e2 = fplib.coverage_expect(st[o], en[h])
        p = poisson.sf(o1 + o2 - 1, e1 + e2)
        print(f"  {keys[h]} <-> {keys[o]} ({grp[o]}): {grp[o]}->hum obs {o1:.0f} exp {e1:.2f} | hum->{grp[o]} obs {o2:.0f} exp {e2:.2f} | p {p:.1e} | human requests {int((A==h).sum())}")
