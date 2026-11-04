exec(open('load_affinity.py').read())
print('window', round(t1-t0), 'keys', N, 'ok rows', len(raw[raw.status==200]), '429', int((raw.status==429).sum()))
rng = np.random.default_rng(1)
lab8, k8 = fplib.site_handoff_graph_coverage(A, S, E, N, t0, t1)
print("v0.8 graph humans", int(((lab8 >= 0) & (grp == 'hum')).sum()))
st = {k: np.sort(S[A == k]) for k in range(N)}; en = {k: E[A == k] for k in range(N)}
for i, j in k8:
    if 'hum' in (grp[i], grp[j]) and grp[i] != grp[j]:
        h, o = (i, j) if grp[i] == 'hum' else (j, i)
        o1, e1 = fplib.coverage_expect(st[h], en[o]); o2, e2 = fplib.coverage_expect(st[o], en[h])
        print(f"  {keys[h]} <-> {keys[o]}: other->hum {o1:.0f}/{e1:.2f}  hum->other {o2:.0f}/{e2:.2f}")
    elif grp[i] == grp[j] == 'hum':
        print("  human-human edge", keys[i], keys[j])
lab9, _ = fplib.site_handoff_graph_v09(A, S, E, N, t0, t1); print("v0.9 graph humans", int(((lab9 >= 0) & (grp == 'hum')).sum()))
labs, _ = fplib.spectral_rings_trimmed_v08(A, S, E, N, t0, t1, rng)
m = grp == 'F6_rrshared'; c = np.bincount(labs[m & (labs >= 0)]).argmax() if (labs[m] >= 0).any() else -1
print("spectral v0.8: F6 recall", (labs[m] == c).mean() if c >= 0 else 0, "purity", round(((labs == c) & m).sum() / (labs == c).sum(), 2) if c >= 0 else None, "humans", int(((labs >= 0) & (grp == 'hum')).sum()))
for g in ['F6_rrshared', 'F1_rot429_shared', 'F7_shuffle', 'F8_pinned']:
    x = d[d.grp == g]; ks = sorted(x.key.unique()); km = {k: i for i, k in enumerate(ks)}
    R = occlib.occupancy(x.key.map(km).values, x.t_arrival.values, x.t_end.values, t0, t1, None, n_accounts=len(ks))["R"]
    print(f"  {g:18s} R = {R:.3f}  K = {fplib.K_little(x.t_arrival.values, x.t_end.values, t0, t1)[0]:.1f}")
