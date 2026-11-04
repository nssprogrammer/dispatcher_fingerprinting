# %% [code]
# Cell 7 — run every population at once, each in its own process
for f in glob.glob("clientlog_*.csv") + glob.glob("labels_*.json") + glob.glob("driverstats_*.json"):
    os.remove(f)
CFG["t_start"] = time.time() + 30                    # common start, after every process has imported its libraries
dcfg = dict(CFG, model=BACKEND["model"])
json.dump(dcfg, open("driver_cfg.json", "w"))
procs = {p: subprocess.Popen([sys.executable, "driver6.py", "driver_cfg.json", p], stdout=open(f"drv_{p}.out", "w"), stderr=subprocess.STDOUT)
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

# ---- 8d. ring robustness: shorter windows and partial observation (only half of the ring's keys at this provider)
rob = {}
for name in ("P3_rrshared", "P5_rrjitter", "P8_relay_rr"):
    if name not in keys_by: continue
    out = {}
    for w in (600, 1200):
        a, s, e, ks = pool_arrays(keys_by[name]); m = (s < t0 + w)
        f, order = fplib.fingerprint(a[m], s[m], e[m], len(ks), t0, min(t1, t0 + w), rng, n_sur=20)
        out[f"first_{w}s"] = dict(ring_z=round(f["ring_z"], 1), acc=round(fplib.neighbour_accuracy(order, np.arange(len(ks))), 2))
    accs, zs = [], []
    for _ in range(5):
        sub = sorted(rng.choice(keys_by[name], len(keys_by[name]) // 2, replace=False))
        a, s, e, ks = pool_arrays(sub)
        f, order = fplib.fingerprint(a, s, e, len(ks), t0, t1, rng, n_sur=20)
        tp = np.array([labels[k]["pos"] for k in ks])
        accs.append(fplib.neighbour_accuracy(order, np.argsort(np.argsort(tp)))); zs.append(f["ring_z"])
    out["half_of_keys"] = dict(ring_z_median=round(float(np.median(zs)), 1), acc_median=round(float(np.median(accs)), 2))
    rob[name] = out
RES["ring_robustness"] = rob

# ---- 8e. site-wide blind discovery: successor graph over every key, no labels, BH over all key pairs
allk = sorted(d.key.unique()); idx = {k: i for i, k in enumerate(allk)}
A = d.key.map(idx).values.astype(np.int64)
comp, edges = fplib.site_handoff_graph(A, d.t_arrival.values, d.t_end.values, len(allk), t0, t1)
grp_of = np.array([labels[k]["pop"] if labels[k]["pop"] not in ("hum_a", "hum_b") else "hum" for k in allk])
comps = []
for c in np.unique(comp[comp >= 0]):
    mem = grp_of[comp == c]
    if len(mem) >= 3:
        u, n = np.unique(mem, return_counts=True); comps.append(dict(size=int(len(mem)), members=dict(zip(u.tolist(), n.tolist()))))
comps.sort(key=lambda c: -c["size"])
recovery = {}
for p in pools:
    if p.startswith(("hum", "benign")): continue
    tot = (grp_of == p).sum()
    best = max([c["members"].get(p, 0) for c in comps] or [0])
    host = next((c for c in comps if c["members"].get(p, 0) == best and best > 0), None)
    recovery[p] = dict(recall=round(best / tot, 2) if tot else None, purity=round(best / host["size"], 2) if host else None)
RES["site_graph"] = dict(n_keys=len(allk), n_edges=len(edges), humans_in_components=int(((comp >= 0) & (grp_of == "hum")).sum()),
                         components=comps[:12], recovery=recovery)

# ---- 8f. pre-registered scorecard
pr = RES["pools"]
closed = [p for p in ("P1_shuffle", "P2_least", "P3_rrshared", "P4_rrworker", "C_company") if p in pr]
RES["scorecard"] = dict(
    S1_classification=f"{sum(r['correct'] for r in pr.values())}/{len(pr)} correct (pass if >= {len(pr) - 2})",
    S2_ring_recovery={p: pr[p]["ring_neighbour_acc"] for p in ("P3_rrshared", "P5_rrjitter", "P8_relay_rr") if p in pr},
    S3_K_little_within_25pct={p: bool(pr[p]["K_little"] and abs(pr[p]["K_little"] / pr[p]["K_true"] - 1) <= 0.25) for p in closed},
    S3b_K_fano_within_25pct={p: bool(pr[p]["K_fano"] and abs(pr[p]["K_fano"] / pr[p]["K_true"] - 1) <= 0.25) for p in closed},
    S4_site_graph={p: RES["site_graph"]["recovery"].get(p) for p in ("P3_rrshared", "P4_rrworker", "P5_rrjitter", "P8_relay_rr")},
    S4_humans_in_components=RES["site_graph"]["humans_in_components"],
    S5_relay_random_zR=pr.get("P7_relay_random", {}).get("z_R"), S5_heavy_benign_zR20=pr.get("benign_36hum_4bat", {}).get("z_R20"))
RES["analysis_min"] = round((time.time() - t_an) / 60, 1)
print(json.dumps(RES["scorecard"], indent=1))

# %% [code]
# Cell 9 — shut down and print the summary
for proc in (gw, BACKEND.get("proc")):
    try: proc.terminate()
    except Exception: pass
d.to_csv("nb6_requests.csv", index=False)
RES["runtime_min"] = round((time.time() - T_START) / 60, 1)
json.dump(RES, open("nb6_results.json", "w"), indent=1, default=str)
print("===== PASTE FROM HERE =====")
print(json.dumps(RES, separators=(",", ":"), default=str))
print("===== PASTE TO HERE =====")
