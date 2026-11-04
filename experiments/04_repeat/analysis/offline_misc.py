exec(open('load7.py').read())
import time, scipy.sparse as sp
from scipy.sparse.linalg import LinearOperator, eigsh
# ---------- (1) SLA impact of the watermark
dur_send = E - S; dur_true = Cc - S
for name, x in (("without watermark", dur_true), ("with watermark", dur_send)):
    print(f"{name:18s} p50 {np.median(x):.3f}s p90 {np.quantile(x,.9):.3f}s p99 {np.quantile(x,.99):.3f}s")
# ---------- (2) harmonics of the jittered ring from its own profile
for g in ['P5_rrjitter', 'P3_rrshared', 'P8_relay_rr']:
    m = (d.grp == g).values; a = d.pos.values[m].astype(int)
    X = fplib.highpass(fplib.occupancy_matrix(a, S[m], E[m], 40, t0, t1, 0.1).astype(float), 200)
    C = fplib.corr_matrix(X); Dm = np.abs(np.subtract.outer(np.arange(40), np.arange(40))); Dm = np.minimum(Dm, 40 - Dm)
    c = np.array([C[Dm == k].mean() for k in range(1, 21)])
    th = [sum(2 * c[dd - 1] * np.cos(2 * np.pi * k * dd / 40) for dd in range(1, 20)) + c[19] * np.cos(np.pi * k) for k in range(1, 6)]
    print(g, "theta_k k=1..5:", np.round(th, 2))
# ---------- (3) matrix-free spectral: Lanczos on the sparse occupancy matrix (no N x N matrix)
N = len(keys); dt = 0.1; blk = 200
t = time.time()
Xs = sp.csr_matrix(fplib.occupancy_matrix(A, S, E, N, t0, t1, dt))      # in production built directly sparse from intervals
Tn = (Xs.shape[1] // blk) * blk; Xs = Xs[:, :Tn]
nnz = Xs.nnz
def hp_T(u):                       # (high-pass of time vector): subtract block means
    U = u.reshape(-1, blk); return (U - U.mean(1, keepdims=True)).ravel()
# per-key mean and sd of the high-pass series, computed with sparse ops
row_sum = np.asarray(Xs.sum(1)).ravel()
Xb = Xs @ sp.kron(sp.eye(Tn // blk), np.ones((blk, 1)) / blk).tocsr()           # N x nblocks block means
sq = np.asarray(Xs.multiply(Xs).sum(1)).ravel() - blk * np.asarray(Xb.multiply(Xb).sum(1)).ravel()
sd = np.sqrt(np.maximum(sq / Tn, 0)); ok = sd > 0; inv = np.where(ok, 1 / np.where(ok, sd, 1), 0)
# common mode g = normalised mean of standardised rows (computed by one pass)
gvec = hp_T(np.asarray((Xs.T @ inv)).ravel()); gvec /= np.linalg.norm(gvec)
def Zt_v(v):   # Z^T v : T-vector, Z = P_g^perp( D^-1 HP(X) )
    y = hp_T(np.asarray(Xs.T @ (inv * v)).ravel()); return y - gvec * (gvec @ y)
def Z_u(u):    # Z u : N-vector
    u = u - gvec * (gvec @ u); return inv * np.asarray(Xs @ hp_T(u)).ravel()
diagZZ = None
# diagonal of Z Z^T / T is 1 for active keys minus the common-mode share; compute exactly via the projection
proj = inv * np.asarray(Xs @ gvec).ravel()            # Z-row components along g before removal (HP(g)=g)
diagC = np.where(ok, 1.0 - proj ** 2 / Tn, 0.0)
op = LinearOperator((N, N), matvec=lambda v: Z_u(Zt_v(v)) / Tn - diagC * v, dtype=float)
vals = eigsh(op, k=10, which='LA', return_eigenvectors=False)
t_lanczos = time.time() - t
dense = np.linalg.eigvalsh(np.load('C7.npy'))[-10:]
print(f"nnz(X) = {nnz:,}  (N={N}, T={Tn}); Lanczos top-10 in {t_lanczos:.1f}s")
print("  matrix-free:", np.round(np.sort(vals)[::-1], 3)); print("  dense      :", np.round(dense[::-1], 3))
