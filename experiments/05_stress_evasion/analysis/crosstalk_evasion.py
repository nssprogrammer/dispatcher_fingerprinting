exec(open('load_evasion.py').read())
from scipy.stats import spearmanr
rng = np.random.default_rng(0)
print("window", round(t1 - t0), "s; requests", len(d))
# (1) do durations rise together with site load? (the cross-talk mechanism)
allv = d.sort_values('t_arrival'); tt = np.r_[allv.t_arrival.values, allv.t_end.values]; dd = np.r_[np.ones(len(allv)), -np.ones(len(allv))]
o = np.lexsort((dd, tt)); lvl = pd.Series(np.cumsum(dd[o]), index=tt[o]).groupby(level=0).last()
n_at = lvl.reindex(allv.t_arrival.values, method='ffill').values
per_tok = (allv.t_end - allv.t_first).values / np.maximum(allv.nbytes.values, 1)
print("Spearman(site in-flight at start, decode time per byte) =", round(spearmanr(n_at, per_tok)[0], 3))
# (2) arrival-count ratio: same variance ratio on request START counts (durations never used)
def R_arr(sub_keys, bin_s=2.0, block_s=60.0):
    m = d.key.isin(sub_keys).values; a = d.a.values[m]; s = d.t_arrival.values[m]
    idx = {k: i for i, k in enumerate(np.unique(a))}; M = len(idx)
    nb = int((t1 - t0) / bin_s); C = np.zeros((M, nb))
    b = ((s - t0) / bin_s).astype(int); ok = (b >= 0) & (b < nb)
    np.add.at(C, (np.array([idx[x] for x in a[ok]]), b[ok]), 1)
    k = int(block_s / bin_s); nb2 = (nb // k) * k; C = C[:, :nb2].reshape(M, -1, k); C = (C - C.mean(2, keepdims=True)).reshape(M, -1)
    return C.sum(0).var() / C.var(1).sum()
def R_occ(sub_keys):
    m = d.key.isin(sub_keys).values; a, _ = occlib.relabel(d.a.values[m])
    return occlib.occupancy(a, d.t_arrival.values[m], d.t_end.values[m], t0, t1, None)["R"]
hum = [k for k in keys if k.startswith('hum')]
null_arr = np.array([R_arr(list(rng.choice(hum, 40, replace=False))) for _ in range(100)])
null_occ = np.array([R_occ(list(rng.choice(hum, 40, replace=False))) for _ in range(100)])
print(f"human null: R_occ {null_occ.mean():.3f}±{null_occ.std():.3f}   R_arrivals {null_arr.mean():.3f}±{null_arr.std():.3f}")
for g in ['E0_rr_J0', 'E2_rr_J6', 'E3_rr_J15', 'E7_shuffle_K8', 'E9_pinned', 'E10_open_steady', 'E11_open_bursty']:
    ks = [k for k in keys if k.startswith(g)]
    ro, ra = R_occ(ks), R_arr(ks)
    print(f"{g:16s} R_occ {ro:.3f} (z {(null_occ.mean()-ro)/null_occ.std():6.1f})   R_arrivals {ra:.3f} (z {(null_arr.mean()-ra)/null_arr.std():6.1f})")
