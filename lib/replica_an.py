"""NB9 analysis (replicas, quotas, held-out dispatchers): one function per phase. Labels are used only to say which keys form a pool and to score answers."""
import json, glob, os, time
import numpy as np, pandas as pd
import occlib, fplib
from occlib import occupancy, fano_khat

RKEYS = ["R", "R5", "R20", "R120"]


def load(run_dir, cfg):
    labels = {}
    for f in glob.glob(os.path.join(run_dir, "labels_*.json")):
        labels.update(json.load(open(f)))
    log = pd.read_csv(os.path.join(run_dir, "gateway_log.csv"), dtype={"rid": str})
    log["grp"] = log.key.map(lambda k: labels.get(k, {}).get("pop"))
    log.loc[log.grp.isin(["hum_a", "hum_b"]), "grp"] = "hum"
    log["pos"] = log.key.map(lambda k: labels.get(k, {}).get("pos"))
    okm = (log.status == 200) & log.grp.notna()
    r429 = log[(log.status == 429) & log.grp.notna()].copy()
    t0 = cfg["t_start"] + cfg["warmup_s"]; t1 = min(float(log.t_end.max()), cfg["t_start"] + cfg["run_s"]) - 30
    d = log[okm & (log.t_end > t0) & (log.t_arrival < t1)].copy()
    d["dur"] = d.t_end - d.t_arrival
    cl = []
    for f in glob.glob(os.path.join(run_dir, "clientlog_*.csv")):
        x = pd.read_csv(f, dtype={"rid": str}); x["src"] = os.path.basename(f)[len("clientlog_"):-4]; cl.append(x)
    cl = pd.concat(cl, ignore_index=True) if cl else pd.DataFrame()
    r429 = r429[(r429.t_arrival > t0) & (r429.t_arrival < t1)]
    return d, labels, cl, t0, t1, dict(n=len(log), ok_frac=round(float(okm.mean()), 4), by_pop=log[okm].grp.value_counts().to_dict(),
                                       n429_by_pop=r429.grp.value_counts().to_dict()), r429



# ---------------------------------------------------------------- forensic seed expansion (identical to the NB7 post-hoc analysis)
def _seed_prep(A, S, E, N, t0, t1w):
    m = (S < t1w) & (E > t0); a, s, e = A[m], S[m], E[m]
    active = np.bincount(a, minlength=N) > 0
    H, Ex = fplib.handoff_excess(a, s, e, N, t0, t1w, 1.0)
    X = fplib.highpass(fplib.occupancy_matrix(a, s, e, N, t0, t1w, 0.1).astype(np.float32), 200)
    X -= X.mean(1, keepdims=True); sd = X.std(1); sd[sd == 0] = 1; X /= sd[:, None]
    return np.nonzero(active)[0], H + H.T, Ex + Ex.T, X


def _seed_scores(seed, cand, Hs, Es, X):
    agg = X[seed].sum(0); agg /= np.linalg.norm(agg) + 1e-12; proj = X[cand] @ agg
    hz = (Hs[np.ix_(cand, seed)].sum(1) - Es[np.ix_(cand, seed)].sum(1)) / np.sqrt(Es[np.ix_(cand, seed)].sum(1) + 1)
    return {"turn": -proj, "handoff": hz, "co": proj}


def _seed_pick(seed, act, Hs, Es, X):
    """label-free choice of linking score: the one under which held-out seed members rank highest"""
    perc = {}
    for j in range(len(seed)):
        rest = np.delete(seed, j); cand = np.setdiff1d(act, rest); sc = _seed_scores(rest, cand, Hs, Es, X)
        for k, v in sc.items(): perc.setdefault(k, []).append((v < v[np.searchsorted(cand, seed[j])]).mean())
    return max(perc, key=lambda k: np.mean(perc[k]))


def seed_expansion(A, S, E, N, t0, t1, grp_of, pools, rng, frac=0.2, reps=15, windows=(600, 1200, None)):
    from sklearn.metrics import roc_auc_score
    res = {}
    for tw in windows:
        t1w = t1 if tw is None else min(t1, t0 + tw); lab = "full" if tw is None else f"{tw // 60}min"
        act, Hs, Es, X = _seed_prep(A, S, E, N, t0, t1w)
        for g in pools:
            mem = np.intersect1d(np.nonzero(grp_of == g)[0], act)
            if len(mem) < 5: continue
            aucs, precs, picks = [], [], []
            for _ in range(reps):
                seed = np.sort(rng.choice(mem, max(2, int(round(frac * len(mem)))), replace=False))
                cand = np.setdiff1d(act, seed); y = np.isin(cand, mem)
                k = _seed_pick(seed, act, Hs, Es, X); sc = _seed_scores(seed, cand, Hs, Es, X)[k]
                aucs.append(roc_auc_score(y.astype(int), sc)); top = np.argsort(-sc)[: int(y.sum())]
                precs.append(float(y[top].mean())); picks.append(k)
            res.setdefault(g, {})[lab] = dict(auc=round(float(np.mean(aucs)), 3), prec_at_pool=round(float(np.mean(precs)), 3),
                                               score=max(set(picks), key=picks.count))
        del X, Hs, Es
    return res

def analyze(run_dir, cfg, rng, phase):
    t_an = time.time()
    d, labels, cl, t0, t1, logsum, r429 = load(run_dir, cfg)
    out = {"log": logsum, "window_s": round(t1 - t0), "durations_s": d.groupby("grp").dur.median().round(2).to_dict()}
    keys_by = {g: sorted(v) for g, v in d.groupby("grp").key.unique().items()}

    def pool_arrays(ks_):
        ks_ = sorted(ks_); idx = {k: i for i, k in enumerate(ks_)}; m = d.key.isin(idx)
        return d.key[m].map(idx).values.astype(np.int64), d.t_arrival.values[m], d.t_end.values[m], ks_

    # ---- human null for R-type statistics
    hum = keys_by.get("hum", []); bat = keys_by.get("bat", [])
    null = []
    for _ in range(200):
        a, s, e, ks = pool_arrays(list(rng.choice(hum, min(40, len(hum)), replace=False)))
        st = {"R": float(occupancy(a, s, e, t0, t1, None, n_accounts=len(ks))["R"])}
        for tau in (5.0, 20.0, 120.0):
            st[f"R{int(tau)}"] = float(occupancy(a, s, e, t0, t1, tau, n_accounts=len(ks))["R"])
        null.append(st)
    NULL = {k: np.array([x[k] for x in null]) for k in RKEYS}
    out["null_humans_40"] = {k: dict(mean=round(float(v.mean()), 4), sd=round(float(v.std()), 4),
                                     frac_abs_z_gt3=round(float((np.abs(v - v.mean()) / v.std() > 3).mean()), 4)) for k, v in NULL.items()}
    zR = lambda st: {k: float((NULL[k].mean() - st[k]) / NULL[k].std()) for k in RKEYS if k in st}
    snull = []
    for _ in range(100):
        a, s_, e_, ks = pool_arrays(list(rng.choice(hum, min(40, len(hum)), replace=False)))
        snull.append(fplib.start_count_ratio(a, s_, len(ks), t0, t1))
    snull = np.array([x for x in snull if np.isfinite(x)])
    out["null_start_ratio"] = dict(mean=round(float(snull.mean()), 4), sd=round(float(snull.std()), 4))
    zS = lambda v: float((snull.mean() - v) / snull.std())

    # ---- fingerprint + frozen classifier
    exp = cfg["expected"][phase]
    pools = {p: keys_by[p] for p in exp if p in keys_by}
    pools["hum_random40"] = list(rng.choice(hum, min(40, len(hum)), replace=False))
    if bat:
        pools["benign_36hum_4bat"] = list(rng.choice(hum, min(36, len(hum)), replace=False)) + list(bat)
    rows = {}
    for name, ks_ in pools.items():
        a, s, e, ks = pool_arrays(ks_); M = len(ks)
        f, order = fplib.fingerprint(a, s, e, M, t0, t1, rng, n_sur=40, occupancy_fn=occupancy)
        z = zR(f); cls = fplib.classify(f, z); fk = fano_khat(a, s, e, t0, t1)
        rz6, sc6, order6 = fplib.ring_multiscale(a, s, e, M, t0, t1, rng, n_sur=40)
        f6 = dict(f, ring_z=max(f["ring_z"], rz6)); cls6 = fplib.classify(f6, z)
        z_start = zS(fplib.start_count_ratio(a, s, M, t0, t1)); cls7 = fplib.classify_v07(f6, z, z_start)
        if rz6 > f["ring_z"] and order6 is not None: order = order6
        true_pos = np.array([labels[k]["pos"] for k in ks]) if all(k in labels for k in ks) else np.arange(M)
        P = cfg["pops"].get(name, {})
        lam_ = f["n_req"] / (t1 - t0)
        ok_ = lambda c: (c != "none") if exp.get(name) == "ANY" else (c in exp.get(name, "none").split(" | "))
        rows[name] = dict(expected=exp.get(name, "none"), predicted=cls, correct=ok_(cls), predicted_v07=cls7, correct_v07=ok_(cls7),
                          z_start=round(z_start, 1),
                          predicted_v06=cls6, correct_v06=ok_(cls6),
                          ring_z_multiscale=round(float(rz6), 1) if np.isfinite(rz6) else None, ring_scale_s=sc6,
                          ring_period_s=round(M / lam_, 1) if lam_ > 0 else None,
                          n_req=f["n_req"], req_per_s=round(f["n_req"] / (t1 - t0), 3),
                          req_per_s_per_worker=round(f["n_req"] / (t1 - t0) / P["K"], 3) if P.get("K") else None,
                          R=round(f["R"], 3), R5=round(f["R5"], 3), R20=round(f["R20"], 3), R120=round(f["R120"], 3),
                          z_R=round(z["R"], 1), z_R20=round(z["R20"], 1), coll=round(f["coll"], 2), ring_z=round(f["ring_z"], 1),
                          conc_z=round(f["conc_z"], 1) if np.isfinite(f["conc_z"]) else None,
                          ring_acc=round(fplib.neighbour_accuracy(order, np.argsort(np.argsort(true_pos))), 2),
                          K_true=P.get("K") if P.get("kind") not in ("open_loop",) else None,
                          K_fano=round(float(fk["K_hat"]), 1) if np.isfinite(fk["K_hat"]) else None,
                          K_little=round(f["K_little"], 1) if np.isfinite(f["K_little"]) else None)
        print(f"[{phase}] {name:20s} v0.6 -> {cls6:38s} {'OK' if rows[name]['correct_v06'] else 'WRONG':5s} | v0.7 -> {cls7:38s} {'OK' if rows[name]['correct_v07'] else 'WRONG'}", flush=True)
    out["pools"] = rows

    # ---- site-wide discovery (frozen fplib v0.5)
    allk = sorted(d.key.unique()); idx = {k: i for i, k in enumerate(allk)}
    A = d.key.map(idx).values.astype(np.int64); S_, E_ = d.t_arrival.values, d.t_end.values; NK = len(allk)
    grp_of = np.array([labels[k]["pop"] if labels[k]["pop"] not in ("hum_a", "hum_b") else "hum" for k in allk])
    pos_of = np.array([labels[k]["pos"] for k in allk])

    def recovery(lab):
        res = {}
        for p in pools:
            if p.startswith(("hum", "benign")): continue
            tot = int((grp_of == p).sum()); best, pur = 0, None
            for c in np.unique(lab[lab >= 0]):
                n = int(((lab == c) & (grp_of == p)).sum())
                if n > best: best, pur = n, round(n / int((lab == c).sum()), 2)
            res[p] = dict(recall=round(best / tot, 2) if tot else None, purity=pur)
        return res
    lab_g, kept = fplib.site_handoff_graph_sideband(A, S_, E_, NK, t0, t1)
    lab_g8, kept8 = fplib.site_handoff_graph_coverage(A, S_, E_, NK, t0, t1)
    hum_links = sorted({grp_of[j] if grp_of[i] == "hum" else grp_of[i] for i, j in kept if "hum" in (grp_of[i], grp_of[j]) and grp_of[i] != grp_of[j]})
    lab_s, groups, bulk_edge, orders = fplib.spectral_rings(A, S_, E_, NK, t0, t1, rng)
    lab_s6 = fplib.spectral_rings_multiscale(A, S_, E_, NK, t0, t1, rng)
    lab_s7, n_trim = fplib.spectral_rings_trimmed(A, S_, E_, NK, t0, t1, rng)
    lab_s8, n_trim8 = fplib.spectral_rings_trimmed_v08(A, S_, E_, NK, t0, t1, rng)
    ring_orders = {}
    for o in orders:
        if len(o) >= 5:
            g = pd.Series(grp_of[o]).mode()[0]
            ring_orders[g] = round(fplib.neighbour_accuracy(np.arange(len(o)), np.argsort(np.argsort(pos_of[o]))), 2)
    out["site"] = dict(sideband_graph=dict(edges=len(kept), humans=int(((lab_g >= 0) & (grp_of == "hum")).sum()), recovery=recovery(lab_g), human_linked_to=hum_links),
                       graph_v08=dict(edges=len(kept8), humans=int(((lab_g8 >= 0) & (grp_of == "hum")).sum()), recovery=recovery(lab_g8)),
                       spectral=dict(bulk_edge=round(bulk_edge, 2), eigen_groups=[[round(x, 2) for x in g] for g in groups],
                                     humans=int(((lab_s >= 0) & (grp_of == "hum")).sum()), recovery=recovery(lab_s), ring_order_acc=ring_orders),
                       spectral_v06=dict(humans=int(((lab_s6 >= 0) & (grp_of == "hum")).sum()), recovery=recovery(lab_s6)),
                       spectral_v07=dict(trimmed=n_trim, humans=int(((lab_s7 >= 0) & (grp_of == "hum")).sum()), recovery=recovery(lab_s7)),
                       spectral_v08=dict(trimmed=n_trim8, humans=int(((lab_s8 >= 0) & (grp_of == "hum")).sum()), recovery=recovery(lab_s8)))

    # ---- provider-side coupling: does decode speed depend on load, and do same-replica human pools take turns?
    from scipy.stats import spearmanr
    allv = d.sort_values("t_arrival"); tt = np.r_[allv.t_arrival.values, allv.t_end.values]; dd = np.r_[np.ones(len(allv)), -np.ones(len(allv))]
    o = np.lexsort((dd, tt)); lvl = pd.Series(np.cumsum(dd[o]), index=tt[o]).groupby(level=0).last()
    n_at = lvl.reindex(allv.t_arrival.values, method="ffill").values
    per_byte = (allv.t_end - allv.t_first).values / np.maximum(allv.nbytes.values, 1)
    out["coupling"] = dict(spearman_load_vs_decode=round(float(spearmanr(n_at, per_byte)[0]), 3),
                           replica_share=d.replica.value_counts(normalize=True).round(3).to_dict() if "replica" in d else None)
    if "replica" in d and d.replica.nunique() > 1:
        rep_of = d.groupby("key").replica.agg(lambda x: x.value_counts().index[0])
        same_frac = d.groupby("key").replica.agg(lambda x: x.value_counts(normalize=True).iloc[0])
        hk = [k for k in hum if k in rep_of.index]
        if float(same_frac.reindex(hk).median()) > 0.9:          # affinity routing: keys live on one replica
            res = {}
            for r_ in sorted(d.replica.unique()):
                pool_r = [k for k in hk if rep_of[k] == r_]
                vals = []
                for _ in range(100):
                    a, s_, e_, ks = pool_arrays(list(rng.choice(pool_r, min(40, len(pool_r)), replace=False)))
                    vals.append(float(occupancy(a, s_, e_, t0, t1, None, n_accounts=len(ks))["R"]))
                vals = np.array(vals); res[f"replica_{int(r_)}"] = dict(R_mean=round(float(vals.mean()), 4),
                                                                         z_mean=round(float((NULL["R"].mean() - vals.mean()) / NULL["R"].std()), 2),
                                                                         frac_abs_z_gt3=round(float((np.abs(NULL["R"].mean() - vals) / NULL["R"].std() > 3).mean()), 3))
            out["same_replica_human_pools"] = res
    # ---- 429 failover hand-offs: after a key is refused, does another key of the pool start within 0.3 s?
    fo = {}
    for p in [x for x in exp if x in keys_by]:
        ks = set(keys_by[p]); rr = r429[r429.key.isin(ks)]
        if len(rr) < 5: fo[p] = dict(n429=int(len(rr))); continue
        st = d[d.key.isin(ks)]; ss = np.sort(st.t_arrival.values); sk = st.sort_values("t_arrival").key.values
        obs = exp_ = 0.0
        for kk, te in zip(rr.key.values, rr.t_end.values):
            x, y = np.searchsorted(ss, te), np.searchsorted(ss, te + 0.3)
            obs += np.sum(sk[x:y] != kk)
            side = np.searchsorted(ss, te + 30) - np.searchsorted(ss, te - 30)
            exp_ += side * 0.3 / 60.0
        fo[p] = dict(n429=int(len(rr)), handoffs=int(obs), expected=round(exp_, 1), z=round(float((obs - exp_) / np.sqrt(exp_ + 1)), 1))
    out["failover_429"] = fo
    # ---- forensic expansion from a random 20% seed of each campaign pool, label-free score choice
    exp_pools = [p for p in ("F7_shuffle", "C_company", "F3_hashed", "F5_hedged", "F1_rot429_shared", "F2_rot429_worker",
                             "F4_sticky_ttl", "F6_rrshared") if p in set(grp_of)]
    out["expansion"] = seed_expansion(A, S_, E_, NK, t0, t1, grp_of, exp_pools, np.random.default_rng(cfg.get("seed", 9) + 77))
    print(f"[{phase}] expansion:", json.dumps(out["expansion"]), flush=True)
    out["analysis_min"] = round((time.time() - t_an) / 60, 1)
    return out
