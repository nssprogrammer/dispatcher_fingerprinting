exec(open('load_balanced.py').read())
r = r429[(r429.t_arrival > t0) & (r429.t_arrival < t1) & r429.key.isin(kid)]
lab, kept = fplib.failover_graph(A, S, r.key.map(kid).values.astype(np.int64), r.t_end.values, N)
print("edges", len(kept), "humans", int(((lab >= 0) & (grp == 'hum')).sum()), "bat", int(((lab >= 0) & (grp == 'bat')).sum()))
for g in sorted(set(grp)):
    if g == 'hum': continue
    mem = grp == g; best = 0; pur = None
    for c in np.unique(lab[lab >= 0]):
        n = int(((lab == c) & mem).sum())
        if n > best: best, pur = n, round(n / int((lab == c).sum()), 2)
    if best: print(f"  {g:18s} recall {best/mem.sum():.2f} purity {pur}")
print("429s by group in window:", r.grp.value_counts().to_dict())
