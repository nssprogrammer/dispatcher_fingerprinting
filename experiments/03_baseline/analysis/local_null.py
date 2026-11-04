exec(open('load.py').read())
from scipy.stats import poisson
N = len(keys)
starts = {k: np.sort(S[A == k]) for k in range(N)}; ends = {k: E[A == k] for k in range(N)}
def local_expect(i, j, w=1.0, B=60.0):
    """observed and sideband-expected count of j-starts within (e, e+w] after i-ends; sideband = [e-B, e+B] minus (e-w, e+w]"""
    sj, ei = starts[j], ends[i]
    obs = (np.searchsorted(sj, ei + w, 'right') - np.searchsorted(sj, ei, 'right')).sum()
    side = (np.searchsorted(sj, ei + B, 'right') - np.searchsorted(sj, ei - B, 'left')).sum() - \
           (np.searchsorted(sj, ei + w, 'right') - np.searchsorted(sj, ei - w, 'left')).sum()
    return obs, side * w / (2 * B - 2 * w)
def recheck(edges, w=1.0, B=60.0, alpha=1e-3):
    keep = []
    for i, j in edges:
        o1, e1 = local_expect(i, j, w, B); o2, e2 = local_expect(j, i, w, B)
        p = poisson.sf(o1 + o2 - 1, max(e1 + e2, 1e-9))
        if p < alpha / max(len(edges), 1): keep.append((i, j))
    return keep
def components(edges):
    parent = list(range(N))
    def f(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i, j in edges: parent[f(i)] = f(j)
    deg = np.bincount(np.array(edges).ravel(), minlength=N) if edges else np.zeros(N, int)
    return np.where(deg > 0, [f(i) for i in range(N)], -1)
def report(lab, name):
    print(name)
    tot = pd.Series(grp).value_counts()
    for g in ['P1_shuffle', 'P2_least', 'P3_rrshared', 'P4_rrworker', 'P5_rrjitter', 'P6_agents', 'P7_relay_random', 'P8_relay_rr', 'C_company']:
        best, pur = 0, None
        for c in np.unique(lab[lab >= 0]):
            mem = grp[lab == c]; n = (mem == g).sum()
            if n > best: best, pur = n, n / len(mem)
        print(f"   {g:16s} recall {best / tot[g]:.2f} purity {pur if pur is None else round(pur, 2)}")
    print("   humans in components:", int(((lab >= 0) & (grp == 'hum')).sum()))
if __name__ == "__main__" and True:
    comp, edges = fplib.site_handoff_graph(A, S, E, N, t0, t1)
    report(comp, "Poisson null (as in NB6)")
    kept = recheck(edges)
    print(f"edges kept after sideband recheck: {len(kept)}/{len(edges)}")
    report(components(kept), "Poisson screen + sideband (local-rate) null")
