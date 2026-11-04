"""Adaptive attacker on real background: for a fixed budget (K workers, M keys, quota Q), every evasion family is run through
the full frozen detector suite; we record throughput and the strongest detector z.  usage: attacker.py WINDOW.parquet OUT.json"""
import sys, json, time, importlib.util
import numpy as np, pandas as pd
import pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import rt, fplib

win, out = sys.argv[1], sys.argv[2]; QUICK = "--quick" in sys.argv
W = 3600.0; rng = np.random.default_rng(77); T0 = time.time()
log = lambda *x: print(f"[{time.time() - T0:6.0f}s]", *x, flush=True)
real, durs, act, strat = rt.load_window(win, W, floor=10, heavy_max=2000)
# human think times: gaps between consecutive requests of light accounts (end -> next start), 0.2 s .. 30 min
g = real[real.grp == "real_light"].sort_values(["key", "t_arrival"])
nxt = g.groupby("key").t_arrival.shift(-1); gaps = (nxt - g.t_end).dropna().values; gaps = gaps[(gaps > 0.2) & (gaps < 1800)]
log("human gaps", len(gaps), "median", round(float(np.median(gaps)), 1))

spec = importlib.util.spec_from_file_location("an", str(rt.ROOT / "lib/replica_an.py"))
an = importlib.util.module_from_spec(spec); spec.loader.exec_module(an)

def failover_z(site, grp):
    r = site.r429[site.r429.grp == grp]
    if len(r) < 5: return None
    st = site.d[site.d.grp == grp]; ss = np.sort(st.t_arrival.values); sk = st.sort_values("t_arrival").key.values
    obs = ex = 0.0
    for kk, te in zip(r.key.values, r.t_end.values):
        x, y = np.searchsorted(ss, te), np.searchsorted(ss, te + 0.3); obs += np.sum(sk[x:y] != kk)
        ex += (np.searchsorted(ss, te + 30) - np.searchsorted(ss, te - 30)) * 0.3 / 60.0
    return float((obs - ex) / np.sqrt(ex + 1))

# (workers K, keys M, per-key quota Q per minute): a slack quota, then binding quotas with scarce and with ample concurrency
BUDGETS = {"K8_M40_Q15": (8, 40, 15), "K8_M40_Q3": (8, 40, 3), "K40_M40_Q3": (40, 40, 3)}
FAMILIES = []
for J in (0, 2, 6, 15, 30): FAMILIES.append((f"counter_J{J}", dict(kind="rr_shared", jitter=float(J))))
for H in (5, 20, 100): FAMILIES.append((f"hold_{H}", dict(kind="hold", H=H)))
FAMILIES += [("shuffle", dict(kind="random")), ("rot429_shared", dict(kind="rot429", shared=True)), ("pinned", dict(kind="pinned")),
             ("pinned_human_gaps", dict(kind="pinned_gaps", gaps=gaps))]
for frac in (0.5, 0.8, 0.95): FAMILIES.append((f"open_cap_{int(frac * 100)}", dict(kind="open_cap", frac=frac)))
if QUICK: FAMILIES = [f for f in FAMILIES if f[0] in ("counter_J0", "counter_J15", "pinned", "pinned_human_gaps", "open_cap_80", "shuffle")]

rows = []
for bname, (K, M, Q) in BUDGETS.items():
    cap_rate = min(K / (np.mean(durs) + 0.2), M * Q / 60.0)              # ideal closed-loop rate, bounded by the quota
    for fname, P0 in FAMILIES:
        P = dict(P0, K=K, M=M)
        if P["kind"] in ("pinned", "pinned_gaps"): P["M"] = min(K, M)    # one key per worker
        if P["kind"] == "open_cap": P["lam"] = P0["frac"] * cap_rate
        name = f"A_{bname}_{fname}"
        sim = rt.simulate_pool(name, P, durs, W, rng, quota=Q)
        site = rt.Site(pd.concat([real, sim], ignore_index=True), W)
        ks = list(site.by.get(name, []))
        if len(ks) < 2: continue
        med = {st: float(real[real.grp == f"real_{st}"].groupby("key").size().median()) for st in ("light", "heavy")}
        inj = float(sim[sim.status == 200].groupby("key").size().median())
        st0 = min(med, key=lambda k: abs(med[k] - inj)); NULL, sn = site.null(f"real_{st0}", rng, n=60 if QUICK else 200)
        f = rt.fingerprint_pool(site, ks, NULL, sn, rng)
        fo = failover_z(site, name)
        ex = an.seed_expansion(site.A, site.S, site.E, len(site.keys), 0, W, site.grp, [name], rng, reps=5, windows=(1200,))
        auc = (ex.get(name, {}).get("20min") or {}).get("auc")
        zmax = max(abs(f["z_R"]), f["ring_z"], f["conc_z"] or 0.0, abs(f["z_start"]), fo or 0.0)
        thr = int((sim.status == 200).sum()) / W
        rows.append(dict(budget=bname, K=K, M=M, Q=Q, family=fname, cls=f["cls"], throughput=round(thr, 3), thr_frac_of_cap=round(thr / cap_rate, 3),
                         per_worker=round(thr / K, 3), keys_used=len(ks), z_R=f["z_R"], ring_z=f["ring_z"], conc_z=f["conc_z"],
                         z_start=f["z_start"], failover_z=None if fo is None else round(fo, 1), expansion_auc20=auc, z_max=round(zmax, 1),
                         evades=bool(f["cls"] == "none" and zmax < 3 and (auc or 0) < 0.7)))
        log(rows[-1])
json.dump(dict(window=win, rows=rows, gaps_median=float(np.median(gaps)), runtime_min=round((time.time() - T0) / 60, 1)), open(out, "w"), indent=1)
