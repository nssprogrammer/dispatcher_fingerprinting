import numpy as np, sys, time, json
import sys as _s, pathlib as _p; ROOT = next(q for q in _p.Path(__file__).resolve().parents if (q / 'lib' / 'fplib.py').exists()); _s.path.insert(0, str(ROOT / 'lib'))
import fplib; from gens import site
from occlib import occupancy
rng = np.random.default_rng(int(sys.argv[1]) if len(sys.argv) > 1 else 7); T = float(sys.argv[2]) if len(sys.argv) > 2 else 2400.0
a, s, e, lab, truth = site(rng, T); N = len(lab); t0, t1 = 60.0, T
def sub(keys):
    m = np.isin(a, keys); mp = {k: i for i, k in enumerate(keys)}
    return np.array([mp[x] for x in a[m]]), s[m], e[m]
t = time.time()
hum = np.arange(*truth["hum"][:2])
nullR = {k: [] for k in ["R", "R5", "R20", "R120"]}; nullF = []
for _ in range(60):
    ks = rng.choice(hum, 40, replace=False); pa, ps, pe = sub(ks)
    f, _ = fplib.fingerprint(pa, ps, pe, 40, t0, t1, rng, n_sur=0, occupancy_fn=occupancy)
    for k in nullR: nullR[k].append(f[k])
    nullF.append(f)
NR = {k: (np.mean(v), np.std(v)) for k, v in nullR.items()}
print("human null:", {k: (round(m, 3), round(sd, 3)) for k, (m, sd) in NR.items()}, "%.0fs" % (time.time() - t))
# human-pool ring/conc as a second null reference
print("human-pool ring  mean %.3f q99 %.3f | conc mean %.3f q99 %.3f" % (np.mean([f['ring'] for f in nullF]), np.quantile([f['ring'] for f in nullF], .99),
      np.nanmean([f['conc'] for f in nullF]), np.nanquantile([f['conc'] for f in nullF], .99)))
print(f"{'pool':14s} {'R':>5s} {'R5':>5s} {'R120':>5s} {'coll':>5s} {'ringz':>6s} {'concz':>6s} {'Klitl':>5s} {'K':>3s} {'nbacc':>5s}  class win")
for name, (lo, hi, K) in truth.items():
    keys = np.arange(lo, hi) if name != "hum" else rng.choice(hum, 40, replace=False)
    pa, ps, pe = sub(keys); M = len(keys)
    f, order = fplib.fingerprint(pa, ps, pe, M, t0, t1, rng, n_sur=30, occupancy_fn=occupancy)
    zR = {k: (NR[k][0] - f[k]) / NR[k][1] for k in NR if k in f}
    cl = fplib.classify(f, zR)
    acc = fplib.neighbour_accuracy(order, np.arange(M))
    print(f"{name:14s} {f['R']:5.2f} {f['R5']:5.2f} {f['R120']:5.2f} {f['coll']:5.2f} {f.get("ring_z",0):6.1f} {f.get("conc_z",0):6.1f} {f["K_little"]:5.1f} {str(K):>3s} {acc:5.2f}  {cl}  {f['lag_win']}")
t = time.time()
comp, edges = fplib.site_handoff_graph(a, s, e, N, t0, t1)
print("\nsite graph: %d edges (%.1fs)" % (len(edges), time.time() - t))
for c in np.unique(comp[comp >= 0]):
    u, n = np.unique(lab[comp == c], return_counts=True)
    if (comp == c).sum() >= 3: print("  component size", (comp == c).sum(), dict(zip(u.tolist(), n.tolist())))
print("  humans in any component:", int(((comp >= 0) & (lab == "hum")).sum()))
