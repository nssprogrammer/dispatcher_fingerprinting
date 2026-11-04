exec(open('load.py').read())
rng = np.random.default_rng(0); N = len(keys); dt = 0.1
def hollow_corr(a, s, e):
    X = fplib.occupancy_matrix(a, s, e, N, t0, t1, dt).astype(np.float64)
    X = fplib.highpass(X, int(20 / dt))
    X -= X.mean(0, keepdims=True) * 0      # placeholder: common drift handled below
    sd = X.std(1); ok = sd > 0
    Z = np.zeros_like(X); Z[ok] = (X[ok] - X[ok].mean(1, keepdims=True)) / sd[ok, None]
    # remove the site-wide common mode (first principal direction of the time axis shared by all keys)
    g = Z[ok].mean(0); g /= np.linalg.norm(g) + 1e-12
    Z[ok] -= np.outer(Z[ok] @ g, g)
    C = Z @ Z.T / Z.shape[1]
    np.fill_diagonal(C, 0.0)
    return C
C = hollow_corr(A, S, E)
w, V = np.linalg.eigh(C)
null_top = []
for _ in range(3):
    a2, s2, e2 = fplib.shift_keys(A, S, E, t0, t1, rng, N)
    wn = np.linalg.eigvalsh(hollow_corr(a2, s2, e2)); null_top.append((wn[-1], wn[0]))
edge_hi = max(x[0] for x in null_top) * 1.05; edge_lo = min(x[1] for x in null_top) * 1.05
print("null bulk edges (x1.05):", round(edge_lo, 3), round(edge_hi, 3))
out_hi = np.nonzero(w > edge_hi)[0]; out_lo = np.nonzero(w < edge_lo)[0]
print("outlier eigenvalues above:", np.round(w[out_hi][::-1], 2))
print("outlier eigenvalues below:", np.round(w[out_lo], 2))
# composition of each top outlier eigenvector: keys carrying half its weight
for idx in list(out_hi[::-1][:12]) + list(out_lo[:6]):
    v = V[:, idx] ** 2; o = np.argsort(-v); c = np.cumsum(v[o]); top = o[: np.searchsorted(c, 0.5) + 1]
    u, n = np.unique(grp[top], return_counts=True)
    print(f"  lambda={w[idx]:+.2f}  half-weight keys={len(top):3d}  {dict(zip(u.tolist(), n.tolist()))}")
# group selection: embed keys by outlier eigenvectors, link pairs with strong positive hollowed correlation among selected keys
sel = np.unique(np.concatenate([np.nonzero(V[:, i] ** 2 > 1.0 / N * 4)[0] for i in out_hi])) if len(out_hi) else np.array([], int)
print("selected keys:", len(sel), dict(pd.Series(grp[sel]).value_counts()))
import scipy.sparse.csgraph as cg, scipy.sparse as sp
thr = 3.0 / np.sqrt(Z_eff := (t1 - t0) / 2.0)   # ~3 sd of a null correlation with ~1 independent sample per 2 s
Cs = C[np.ix_(sel, sel)]; adj = sp.csr_matrix(Cs > thr)
ncomp, lab = cg.connected_components(adj, directed=False)
lab_full = -np.ones(N, int)
for c in range(ncomp):
    mem = sel[lab == c]
    if len(mem) >= 3: lab_full[mem] = c
from local_null import report
report(lab_full, f"Hollowed spectral selection (corr threshold {thr:.3f})")
np.save('C_hollow.npy', C)
