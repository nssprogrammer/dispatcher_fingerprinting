"""NB8 analysis: one function per phase. Labels are used only to say which keys form a pool and to score answers."""
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
    t0 = cfg["t_start"] + cfg["warmup_s"]; t1 = min(float(log.t_end.max()), cfg["t_start"] + cfg["run_s"]) - 30
    d = log[okm & (log.t_end > t0) & (log.t_arrival < t1)].copy()
    d["dur"] = d.t_end - d.t_arrival
    cl = []
    for f in glob.glob(os.path.join(run_dir, "clientlog_*.csv")):
        x = pd.read_csv(f, dtype={"rid": str}); x["src"] = os.path.basename(f)[len("clientlog_"):-4]; cl.append(x)
    cl = pd.concat(cl, ignore_index=True) if cl else pd.DataFrame()
    return d, labels, cl, t0, t1, dict(n=len(log), ok_frac=round(float(okm.mean()), 4), by_pop=log[okm].grp.value_counts().to_dict())


def analyze(run_dir, cfg, rng, phase):
    t_an = time.time()
    d, labels, cl, t0, t1, logsum = load(run_dir, cfg)
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
        if rz6 > f["ring_z"] and order6 is not None: order = order6
        true_pos = np.array([labels[k]["pos"] for k in ks]) if all(k in labels for k in ks) else np.arange(M)
        P = cfg["pops"].get(name, {})
        lam_ = f["n_req"] / (t1 - t0)
        rows[name] = dict(expected=exp.get(name, "none"), predicted=cls, correct=cls in exp.get(name, "none").split(" | "),
                          predicted_v06=cls6, correct_v06=cls6 in exp.get(name, "none").split(" | "),
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
        print(f"[{phase}] {name:20s} v0.5 -> {cls:38s} {'OK' if rows[name]['correct'] else 'WRONG':5s} | v0.6 -> {cls6:38s} {'OK' if rows[name]['correct_v06'] else 'WRONG'}", flush=True)
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
    lab_s, groups, bulk_edge, orders = fplib.spectral_rings(A, S_, E_, NK, t0, t1, rng)
    lab_s6 = fplib.spectral_rings_multiscale(A, S_, E_, NK, t0, t1, rng)
    ring_orders = {}
    for o in orders:
        if len(o) >= 5:
            g = pd.Series(grp_of[o]).mode()[0]
            ring_orders[g] = round(fplib.neighbour_accuracy(np.arange(len(o)), np.argsort(np.argsort(pos_of[o]))), 2)
    out["site"] = dict(sideband_graph=dict(edges=len(kept), humans=int(((lab_g >= 0) & (grp_of == "hum")).sum()), recovery=recovery(lab_g)),
                       spectral=dict(bulk_edge=round(bulk_edge, 2), eigen_groups=[[round(x, 2) for x in g] for g in groups],
                                     humans=int(((lab_s >= 0) & (grp_of == "hum")).sum()), recovery=recovery(lab_s), ring_order_acc=ring_orders),
                       spectral_v06=dict(humans=int(((lab_s6 >= 0) & (grp_of == "hum")).sum()), recovery=recovery(lab_s6)))

    # ---- watermark (exact randomisation tests)
    dw = d.copy(); dw["c"] = dw.t_complete; dw["b"] = dw.wm_bit.astype(int)
    out["watermark_cost_ms"] = dict(mean_added=round(float((dw.t_end - dw.c).mean() * 1000), 1),
                                    p50_with=round(float(np.median(dw.t_end - dw.t_arrival)), 3), p50_without=round(float(np.median(dw.c - dw.t_arrival)), 3),
                                    p99_with=round(float(np.quantile(dw.t_end - dw.t_arrival, .99)), 3), p99_without=round(float(np.quantile(dw.c - dw.t_arrival, .99)), 3))

    def wm(src, dst, w=1.0, same=False, n_perm=2000):
        kk = {k: i for i, k in enumerate(sorted(set(src) | set(dst)))}
        m = dw.key.isin(kk).values; a = dw.key[m].map(kk).values
        r = fplib.watermark_echo(a, dw.t_arrival.values[m], dw.c.values[m], dw.b.values[m], [kk[k] for k in src], [kk[k] for k in dst],
                                 w=w, n_perm=n_perm, rng=rng, same_key=same)
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
    wmres = {}
    for p in [x for x in exp if x in keys_by]:
        ks = keys_by[p]; half = list(rng.permutation(ks)); hA, hB = half[: len(ks) // 2], half[len(ks) // 2:]
        wmres[p] = dict(pool_w1=wm(ks, ks, 1.0), pool_w8=wm(ks, ks, 8.0), halves_w1=wm(hA, hB, 1.0), same_key_w1=wm(ks, ks, 1.0, same=True))
    hp = []
    for _ in range(50):
        ks = list(rng.choice(hum, min(40, len(hum)), replace=False)); hp.append(wm(ks, ks, 1.0, n_perm=1000)["p"])
    hp = np.array([x for x in hp if np.isfinite(x)])
    out["watermark"] = wmres
    out["watermark_human_null"] = dict(n=int(len(hp)), n_p_below_0_01=int((hp < 0.01).sum()), n_p_below_0_05=int((hp < 0.05).sum()),
                                       median_p=round(float(np.median(hp)), 3) if len(hp) else None)
    # ---- can the client see the watermark? last inter-chunk gap by coin
    if len(cl) and "gap_last" in cl:
        j = dw.merge(cl[["rid", "gap_last", "gap_med"]], on="rid", how="inner")
        j = j[np.isfinite(j.gap_last)]
        if len(j):
            out["client_visibility_ms"] = dict(mode=cfg["wm_mode"][phase],
                                               last_gap_bit1=round(float(j.gap_last[j.b == 1].median() * 1000), 1),
                                               last_gap_bit0=round(float(j.gap_last[j.b == 0].median() * 1000), 1),
                                               median_gap=round(float(j.gap_med.median() * 1000), 1),
                                               auc_bit_from_last_gap=round(float(__import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(j.b, j.gap_last)), 3))
    out["analysis_min"] = round((time.time() - t_an) / 60, 1)
    return out
