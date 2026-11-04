# %% [code]
# Cell 7 — run every population at once, each in its own process
for f in glob.glob("clientlog_*.csv") + glob.glob("labels_*.json") + glob.glob("driverstats_*.json"):
    os.remove(f)
CFG["t_start"] = time.time() + 30                    # common start, after every process has imported its libraries
dcfg = dict(CFG, model=BACKEND["model"])
json.dump(dcfg, open("driver_cfg.json", "w"))
procs = {p: subprocess.Popen([sys.executable, "driver7.py", "driver_cfg.json", p], stdout=open(f"drv_{p}.out", "w"), stderr=subprocess.STDOUT)
         for p in CFG["pops"]}
t_run = time.time()
while any(pr.poll() is None for pr in procs.values()):
    time.sleep(120)
    nlines = sum(1 for _ in open("gateway_log.csv")) - 1
    print(f"[run] t={time.time() - CFG['t_start']:.0f}s  gateway requests={nlines}  running={sum(pr.poll() is None for pr in procs.values())}", flush=True)
RUN = dict(rc={p: pr.returncode for p, pr in procs.items()}, wall_min=round((time.time() - t_run) / 60, 1),
           driver_tail={p: open(f"drv_{p}.out").read().strip().splitlines()[-1][:300] if open(f"drv_{p}.out").read().strip() else "" for p in procs})
for p, line in RUN["driver_tail"].items(): print(line)

# %% [code]
# Cell 8 — analysis (labels are used only to say which keys form a pool and to score the answers)
import numpy as np, pandas as pd
import importlib, occlib, fplib
importlib.reload(occlib); importlib.reload(fplib)
from occlib import occupancy, fano_khat
rng = np.random.default_rng(CFG["seed"])
t_an = time.time()
labels = {}
for f in glob.glob("labels_*.json"): labels.update(json.load(open(f)))
log = pd.read_csv("gateway_log.csv", dtype={"rid": str})
log["grp"] = log.key.map(lambda k: labels.get(k, {}).get("pop"))
log.loc[log.grp.isin(["hum_a", "hum_b"]), "grp"] = "hum"
log["pos"] = log.key.map(lambda k: labels.get(k, {}).get("pos"))
okm = (log.status == 200) & log.grp.notna()
RES = {"cfg": {k: v for k, v in CFG.items() if k not in ("expected",)}, "backend": {k: v for k, v in BACKEND.items() if k != "proc"},
       "run": RUN, "log": dict(n=len(log), ok_frac=round(float(okm.mean()), 4), by_pop=log[okm].grp.value_counts().to_dict())}
t0 = CFG["t_start"] + CFG["warmup_s"]; t1 = min(float(log.t_end.max()), CFG["t_start"] + CFG["run_s"]) - 30
d = log[okm & (log.t_end > t0) & (log.t_arrival < t1)].copy()
d["dur"] = d.t_end - d.t_arrival
RES["window_s"] = round(t1 - t0)
RES["durations_s"] = d.groupby("grp").dur.median().round(2).to_dict()
keys_by = {g: sorted(v) for g, v in d.groupby("grp").key.unique().items()}

def pool_arrays(keys_):
    keys_ = sorted(keys_); idx = {k: i for i, k in enumerate(keys_)}
    m = d.key.isin(idx)
    return d.key[m].map(idx).values.astype(np.int64), d.t_arrival.values[m], d.t_end.values[m], keys_

# ---- 8a. client <-> gateway timing (confound diagnostics)
cl = []
for f in glob.glob("clientlog_*.csv"):
    x = pd.read_csv(f, dtype={"rid": str}); x["src"] = f[len("clientlog_"):-4]; cl.append(x)
cl = pd.concat(cl, ignore_index=True)
j = d.merge(cl, on="rid", how="inner")
q = lambda v: {f"p{int(p * 100)}": round(float(np.quantile(v, p)) * 1000, 1) for p in (0.5, 0.9, 0.99)} if len(v) else {}
RES["timing_ms"] = dict(joined_frac=round(len(j) / max(len(d), 1), 3),
                        client_send_to_gateway_arrival=q((j.t_arrival - j.t_send).values),
                        gateway_end_to_client_done=q((j.t_done - j.t_end).values),
                        relay_schedule_lag={g: q((j.t_send - j.t_sched)[j.grp == g].values) for g in ("P7_relay_random", "P8_relay_rr")})

# ---- 8b. human null for R-type statistics
hum = keys_by.get("hum", []); bat = keys_by.get("bat", [])
RKEYS = ["R", "R5", "R20", "R120"]
def r_stats(a, s, e, M):
    out = {"R": float(occupancy(a, s, e, t0, t1, None, n_accounts=M)["R"])}
    for tau in (5.0, 20.0, 120.0):
        out[f"R{int(tau)}"] = float(occupancy(a, s, e, t0, t1, tau, n_accounts=M)["R"])
    return out
null = []
for _ in range(200):
    a, s, e, ks = pool_arrays(list(rng.choice(hum, min(40, len(hum)), replace=False)))
    null.append(r_stats(a, s, e, len(ks)))
NULL = {k: np.array([x[k] for x in null]) for k in RKEYS}
RES["null_humans_40"] = {k: dict(mean=round(float(v.mean()), 4), sd=round(float(v.std()), 4)) for k, v in NULL.items()}
def zR(st):   # positive = turn-taking (below the human null), negative = bunching
    return {k: float((NULL[k].mean() - st[k]) / NULL[k].std()) for k in RKEYS if k in st}
# human pools through the full fingerprint (no surrogates): reference values for ring / conc / coll
href = []
for _ in range(20):
    a, s, e, ks = pool_arrays(list(rng.choice(hum, min(40, len(hum)), replace=False)))
    f, _ = fplib.fingerprint(a, s, e, len(ks), t0, t1, rng, n_sur=0)
    href.append(f)
RES["human_reference"] = {k: dict(mean=round(float(np.nanmean([h[k] for h in href])), 4), q95=round(float(np.nanquantile([h[k] for h in href], 0.95)), 4))
                          for k in ("ring", "conc", "coll")}

# ---- 8c. fingerprint every pool, classify blind, score
TRUE_K = {p: CFG["pops"][p].get("K") for p in CFG["pops"] if CFG["pops"][p]["kind"] not in ("human", "batch", "open_loop")}
pools = {p: keys_by[p] for p in CFG["pops"] if p in keys_by and CFG["pops"][p]["kind"] not in ("human", "batch")}
pools["hum_random40"] = list(rng.choice(hum, min(40, len(hum)), replace=False))
pools["benign_36hum_4bat"] = list(rng.choice(hum, min(36, len(hum)), replace=False)) + list(bat)
rows = {}
for name, ks_ in pools.items():
    a, s, e, ks = pool_arrays(ks_); M = len(ks)
    f, order = fplib.fingerprint(a, s, e, M, t0, t1, rng, n_sur=40, occupancy_fn=occupancy)
    z = zR(f); cls = fplib.classify(f, z)
    fk = fano_khat(a, s, e, t0, t1)
    used = np.unique(a); busy = [float(np.sum(np.minimum(e[a == k], t1) - np.maximum(s[a == k], t0)) / (t1 - t0)) for k in used]
    true_pos = np.array([labels[k]["pos"] for k in ks]) if all(k in labels for k in ks) else np.arange(M)
    rows[name] = dict(expected=CFG["expected"].get(name), predicted=cls, correct=cls == CFG["expected"].get(name),
                      n_req=f["n_req"], R=round(f["R"], 3), R5=round(f["R5"], 3), R20=round(f["R20"], 3), R120=round(f["R120"], 3),
                      z_R=round(z["R"], 1), z_R20=round(z["R20"], 1), coll=round(f["coll"], 2), ring_z=round(f["ring_z"], 1),
                      conc_z=round(f["conc_z"], 1) if np.isfinite(f["conc_z"]) else None, lag_win=f["lag_win"],
                      ring_neighbour_acc=round(fplib.neighbour_accuracy(order, np.argsort(np.argsort(true_pos))), 2),
                      K_true=TRUE_K.get(name), K_fano=round(float(fk["K_hat"]), 1) if np.isfinite(fk["K_hat"]) else None,
                      K_little=round(f["K_little"], 1), think_s=round(f["think_s"], 2),
                      keys_used=int(len(used)), mean_inflight_per_used_key=round(float(np.mean(busy)), 2))
    print(f"{name:18s} -> {cls:38s} {'OK' if rows[name]['correct'] else 'WRONG'}", flush=True)
RES["pools"] = rows

# ---- 8d. site-wide blind discovery with the detectors frozen after NB6 (no labels used)
allk = sorted(d.key.unique()); idx = {k: i for i, k in enumerate(allk)}
A = d.key.map(idx).values.astype(np.int64); S_, E_ = d.t_arrival.values, d.t_end.values; NK = len(allk)
grp_of = np.array([labels[k]["pop"] if labels[k]["pop"] not in ("hum_a", "hum_b") else "hum" for k in allk])
pos_of = np.array([labels[k]["pos"] for k in allk])
def recovery(lab):
    out = {}
    for p in pools:
        if p.startswith(("hum", "benign")): continue
        tot = (grp_of == p).sum(); best, pur = 0, None
        for c in np.unique(lab[lab >= 0]):
            n = int(((lab == c) & (grp_of == p)).sum())
            if n > best: best, pur = n, round(n / int((lab == c).sum()), 2)
        out[p] = dict(recall=round(best / tot, 2) if tot else None, purity=pur)
    return out
lab_g, kept = fplib.site_handoff_graph_sideband(A, S_, E_, NK, t0, t1)
lab_s, groups, bulk_edge, orders = fplib.spectral_rings(A, S_, E_, NK, t0, t1, rng)
ring_orders = {}
for o in orders:
    if len(o) >= 5:
        g = pd.Series(grp_of[o]).mode()[0]
        ring_orders[g] = round(fplib.neighbour_accuracy(np.arange(len(o)), np.argsort(np.argsort(pos_of[o]))), 2)
RES["site"] = dict(
    sideband_graph=dict(edges=len(kept), humans=int(((lab_g >= 0) & (grp_of == "hum")).sum()), recovery=recovery(lab_g)),
    spectral=dict(bulk_edge=round(bulk_edge, 2), eigen_groups=[[round(x, 2) for x in g] for g in groups],
                  humans=int(((lab_s >= 0) & (grp_of == "hum")).sum()), recovery=recovery(lab_s), ring_order_acc=ring_orders))
print(json.dumps(RES["site"], indent=1), flush=True)

# ---- 8e. active timing watermark: exact randomisation tests
dw = d.copy(); dw["c"] = dw.t_complete; dw["b"] = dw.wm_bit.astype(int)
RES["watermark_cost_ms"] = dict(mean_added=round(float((dw.t_end - dw.t_complete).mean() * 1000), 1),
                                bit1=round(float((dw.t_end - dw.t_complete)[dw.b == 1].mean() * 1000), 1),
                                bit0=round(float((dw.t_end - dw.t_complete)[dw.b == 0].mean() * 1000), 1),
                                share_of_duration_pct=round(float((dw.t_end - dw.t_complete).mean() / dw.dur.mean() * 100), 2))
def wm(keys_src, keys_dst, w=1.0, same=False, n_perm=2000):
    m = dw.key.isin(set(keys_src) | set(keys_dst)).values
    kk = {k: i for i, k in enumerate(sorted(set(keys_src) | set(keys_dst)))}
    a = dw.key[m].map(kk).values
    r = fplib.watermark_echo(a, dw.t_arrival.values[m], dw.c.values[m], dw.b.values[m], [kk[k] for k in keys_src], [kk[k] for k in keys_dst],
                             w=w, n_perm=n_perm, rng=rng, same_key=same)
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}
wmres = {}
for p in [x for x in CFG["pops"] if x in keys_by and CFG["pops"][x]["kind"] not in ("human",)]:
    ks = keys_by[p]; half = list(rng.permutation(ks))
    hA, hB = half[: len(ks) // 2], half[len(ks) // 2:]
    wmres[p] = dict(pool_w1=wm(ks, ks, 1.0), pool_w8=wm(ks, ks, 8.0), halves_w1=wm(hA, hB, 1.0), same_key_w1=wm(ks, ks, 1.0, same=True))
    print(p, {k: (v["shift_ms"], v["p"]) for k, v in wmres[p].items()}, flush=True)
hp = []
for _ in range(50):
    ks = list(rng.choice(hum, min(40, len(hum)), replace=False)); hp.append(wm(ks, ks, 1.0, n_perm=1000)["p"])
hp = np.array([x for x in hp if np.isfinite(x)])
mix = list(rng.choice(hum, min(36, len(hum)), replace=False)) + list(bat)
wmres["benign_36hum_4bat"] = dict(pool_w1=wm(mix, mix, 1.0))
RES["watermark"] = wmres
RES["watermark_human_null"] = dict(n=int(len(hp)), frac_p_below_0_01=round(float((hp < 0.01).mean()), 3),
                                   frac_p_below_0_05=round(float((hp < 0.05).mean()), 3), median_p=round(float(np.median(hp)), 3))
print(RES["watermark_human_null"], RES["watermark_cost_ms"], flush=True)

# ---- 8f. scorecard against the locked predictions
pr, W, st = RES["pools"], RES["watermark"], RES["site"]
P = lambda p, k="pool_w1": W.get(p, {}).get(k, {}).get("p", float("nan"))
RES["scorecard"] = dict(
    W1_echo_closed_loops={p: P(p) for p in ("P1_shuffle", "P2_least", "P3_rrshared", "P4_rrworker", "C_company")},
    W1_pass=all(P(p) < 0.001 for p in ("P1_shuffle", "P2_least", "P3_rrshared", "P4_rrworker", "C_company")),
    W2_pinned=dict(cross_key_p=P("P9_pinned"), same_key_p=P("P9_pinned", "same_key_w1"),
                   pass_=bool(P("P9_pinned") > 0.01 and P("P9_pinned", "same_key_w1") < 0.001)),
    W3_relays={p: P(p) for p in ("P7_relay_random", "P8_relay_rr")}, W3_pass=all(P(p) > 0.01 for p in ("P7_relay_random", "P8_relay_rr")),
    W4_humans=dict(**RES["watermark_human_null"], pass_=bool((hp < 0.01).sum() <= 2)),
    W5_halves={p: P(p, "halves_w1") for p in ("P1_shuffle", "P3_rrshared", "P4_rrworker")},
    W5_pass=all(P(p, "halves_w1") < 0.001 for p in ("P1_shuffle", "P3_rrshared", "P4_rrworker")),
    F1_classification=f"{sum(r['correct'] for r in pr.values())}/{len(pr)} correct (pass if >= 10)",
    F2_spectral=dict(**{p: st["spectral"]["recovery"].get(p) for p in ("P3_rrshared", "P5_rrjitter", "P8_relay_rr")}, humans=st["spectral"]["humans"]),
    F3_sideband_graph=dict(**{p: st["sideband_graph"]["recovery"].get(p) for p in ("P3_rrshared", "P4_rrworker")}, humans=st["sideband_graph"]["humans"]),
)
RES["analysis_min"] = round((time.time() - t_an) / 60, 1)
print(json.dumps(RES["scorecard"], indent=1, default=str))

# %% [code]
# Cell 9 — shut down and print the summary
for proc in (gw, BACKEND.get("proc")):
    try: proc.terminate()
    except Exception: pass
d.to_csv("nb7_requests.csv", index=False)
RES["runtime_min"] = round((time.time() - T_START) / 60, 1)
json.dump(RES, open("nb7_results.json", "w"), indent=1, default=str)
print("===== PASTE FROM HERE =====")
print(json.dumps(RES, separators=(",", ":"), default=str))
print("===== PASTE TO HERE =====")

