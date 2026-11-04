exec(open('load.py').read())
rng = np.random.default_rng(3); N = len(keys)
C = np.load('C_hollow.npy'); w = np.linalg.eigvalsh(C)
# null entries from one time-shifted surrogate: entry variance sigma^2
Cn = fplib.hollow_corr(*fplib.shift_keys(A, S, E, t0, t1, rng, N), N, t0, t1)
iu = np.triu_indices(N, 1); sig2 = float(np.mean(Cn[iu] ** 2)); wn = np.linalg.eigvalsh(Cn)
print(f"null entry sd {np.sqrt(sig2):.4f}; semicircle edge 2*sigma*sqrt(N) = {2 * np.sqrt(sig2 * N):.2f}; observed null top eig {wn[-1]:.2f}")
# BBP: a spike of strength theta (in operator norm) gives an outlier at theta + sigma^2 N / theta iff theta > sigma sqrt(N)
thr = np.sqrt(sig2 * N)
for g, theta in [('P3_rrshared', 6.83), ('P8_relay_rr', 4.58), ('P5_rrjitter', 2.73)]:
    loc = theta + sig2 * N / theta if theta > thr else float('nan')
    print(f"  {g}: ring strength theta={theta:.2f} (threshold {thr:.2f}) -> predicted outlier {loc:.2f}")
print("observed outliers:", np.round(w[-8:][::-1], 2))
# minimum ring strength for site-wide discovery as a function of N keys and window T (sigma^2 ~ tau_c / T)
tau_c = sig2 * (t1 - t0)
print(f"effective correlation time tau_c = sigma^2 * T = {tau_c:.2f} s  -> detectable if theta > sqrt(N * tau_c / T)")
for Nk in (700, 1e4, 1e5):
    for T in (2190, 3600 * 6, 86400):
        print(f"   N={int(Nk):6d}  T={T/3600:5.1f} h  threshold theta* = {np.sqrt(Nk * tau_c / T):.2f}")
