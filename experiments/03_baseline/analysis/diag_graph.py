exec(open('load.py').read())
from collections import Counter
N = len(keys)
comp, edges = fplib.site_handoff_graph(A, S, E, N, t0, t1)
H, Ex = fplib.handoff_excess(A, S, E, N, t0, t1, 1.0)
kinds = Counter()
hum_edges = []
for i, j in edges:
    gi, gj = sorted([grp[i], grp[j]]); kinds[(gi, gj) if gi != gj else (gi, 'same')] += 1
    if 'hum' in (grp[i], grp[j]): hum_edges.append((i, j))
print("edge types:", dict(kinds))
print("edges touching humans:", len(hum_edges))
for i, j in hum_edges[:12]:
    print(f"  {keys[i]:16s} {keys[j]:16s} H_ij={H[i,j]:.0f} Ex={Ex[i,j]:.2f}  H_ji={H[j,i]:.0f} Ex={Ex[j,i]:.2f}  n_i={np.sum(A==i)} n_j={np.sum(A==j)}")
