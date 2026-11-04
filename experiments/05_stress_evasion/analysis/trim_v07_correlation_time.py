exec(open('load_evasion.py').read())
rng = np.random.default_rng(1); N = len(keys); A, S, E = d.a.values.astype(np.int64), d.t_arrival.values, d.t_end.values
# per-key effective correlation time of high-pass occupancy (label-free)
X = fplib.highpass(fplib.occupancy_matrix(A, S, E, N, t0, t1, 0.5).astype(float), 40)
X = X - X.mean(1, keepdims=True); sd = X.std(1); sd[sd == 0] = 1; Z = X / sd[:, None]
F = np.fft.rfft(Z, 2 * Z.shape[1]); ac = np.fft.irfft(F * np.conj(F), 2 * Z.shape[1])[:, :240] / Z.shape[1]
tau = 0.5 * (ac ** 2).sum(1) * 2                     # integrated squared autocorrelation (s)
for g in ['E11_open_bursty', 'E10_open_steady', 'E0_rr_J0', 'E8_shuffle_K18', 'hum', 'bat']:
    print(f"{g:16s} median tau_c {np.median(tau[grp == g]):6.2f}s")
# label-free trimming: drop keys whose tau_c is far above the site median (robust z > 5)
med = np.median(tau); mad = np.median(np.abs(tau - med)) * 1.4826
keep = tau < med + 5 * mad
print("trimmed keys:", dict(pd.Series(grp[~keep]).value_counts()))
sel = np.nonzero(keep)[0]; mp = -np.ones(N, int); mp[sel] = np.arange(len(sel))
m = keep[A]; A2 = mp[A[m]]; lab, groups, edge, orders = fplib.spectral_rings(A2, S[m], E[m], len(sel), t0, t1, rng)
print("after trimming: edge", round(edge, 2), "groups", [[round(x, 2) for x in g] for g in groups])
for c in range(lab.max() + 1):
    u, n = np.unique(grp[sel][lab == c], return_counts=True); print("   ring:", dict(zip(u.tolist(), n.tolist())))
