exec(open('load.py').read())
dt = 0.1
out = {}
# ---------------- Check A: ring correlation profile = single-key autocorrelation averaged over the d-step dispatch delay
for g in ['P3_rrshared', 'P8_relay_rr', 'P5_rrjitter']:
    m = (d.grp == g).values; a = d.pos.values[m].astype(int); s, e = S[m], E[m]; M = 40
    X = fplib.highpass(fplib.occupancy_matrix(a, s, e, M, t0, t1, dt).astype(float), int(20 / dt))
    C = fplib.corr_matrix(X)
    D = np.abs(np.subtract.outer(np.arange(M), np.arange(M))); D = np.minimum(D, M - D)
    prof = np.array([C[D == k].mean() for k in range(1, 21)])
    # single-key autocorrelation (average over keys)
    Y = X - X.mean(1, keepdims=True); n = Y.shape[1]; F = np.fft.rfft(Y, 2 * n)
    ac = np.fft.irfft(F * np.conj(F), 2 * n)[:, :n].mean(0); ac /= ac[0]
    # empirical distribution of the delay between a dispatch to key i and the next dispatch to key i+d (ring order known only for scoring theory)
    o = np.argsort(s); ss, aa = s[o], a[o]
    pred = []
    for k in range(1, 21):
        lags = []
        for i in range(0, len(ss), 7):
            j = np.searchsorted(ss, ss[i], 'right'); tgt = (aa[i] + k) % M
            nxt = np.nonzero(aa[j:j + 400] == tgt)[0]
            if len(nxt): lags.append(ss[j + nxt[0]] - ss[i])
        lags = np.array(lags); half_cycle = M / (len(s) / (t1 - t0)) / 2
        lags = np.where(lags > half_cycle, lags - 2 * half_cycle, lags)     # nearest dispatch in either direction (circular)
        pred.append(np.mean(ac[np.clip(np.round(np.abs(lags) / dt).astype(int), 0, n - 1)]))
    pred = np.array(pred)
    lam = np.linalg.eigvalsh(np.where(np.eye(M, dtype=bool), 0, C))[-1]
    theta = 2 * sum(prof[k - 1] * np.cos(2 * np.pi * k / M) for k in range(1, 20)) + prof[19] * np.cos(np.pi)
    out[g] = dict(profile=np.round(prof[:8], 3).tolist(), predicted=np.round(pred[:8], 3).tolist(),
                  rmse=round(float(np.sqrt(np.mean((prof - pred) ** 2))), 3), block_top_eig=round(float(lam), 2), circulant_theta=round(float(theta), 2))
    print(g, out[g], flush=True)
# ---------------- Check B: successor offset along the ring has mean K for a shared counter
m = (d.grp == 'P3_rrshared').values; a = d.pos.values[m].astype(int); s, e = S[m], E[m]
H, Ex = fplib.handoff_excess(a, s, e, 40, t0, t1, 1.0); Xh = np.clip(H - Ex, 0, None); np.fill_diagonal(Xh, 0)
offs = np.zeros(40)
for i in range(40):
    for j in range(40): offs[(j - i) % 40] += Xh[i, j]
mean_off = (np.arange(1, 20) * offs[1:20]).sum() / offs[1:20].sum()
print("Check B: P3 mean forward successor offset", round(mean_off, 2), "(K = 12); mode", int(np.argmax(offs[1:20]) + 1))
