"""Run locked predictions R1-R8 on one Chutes window.  usage: analyze_window.py WINDOW.parquet OUT.json [--quick]"""
import sys, json, time, argparse, importlib.util
import numpy as np, pandas as pd
sys.path.insert(0, "/tmp/claude-0/-home-claude/78f29ddf-32f5-54b2-bcc0-86fa7c435eb9/scratchpad/nb10")
import rt, fplib
from occlib import occupancy

ap = argparse.ArgumentParser(); ap.add_argument("win"); ap.add_argument("out"); ap.add_argument("--W", type=float, default=3600.0)
ap.add_argument("--quick", action="store_true"); ap.add_argument("--nmax", type=int, default=4000); ap.add_argument("--seed", type=int, default=10)
ap.add_argument("--only", default="R1,R2,R3,R4,R5,R6,R7,R8,R9"); ap.add_argument("--floor", type=int, default=3); ap.add_argument("--heavy_max", type=int, default=None)
args = ap.parse_args(); ONLY = set(args.only.split(","))
G_ = {}
def _fp(job):
    ks, sd = job
    return rt.fingerprint_pool(G_["site"], ks, G_["NULL"], G_["sn"], np.random.default_rng(sd))
NP = 12 if args.quick else 200
rng = np.random.default_rng(args.seed); W = args.W; T0 = time.time(); RES = {"window": args.win, "W": W, "floor": args.floor, "seed": args.seed}
log = lambda *x: print(f"[{time.time() - T0:6.0f}s]", *x, flush=True)

real, durs, act, strat = rt.load_window(args.win, W, floor=args.floor, heavy_max=args.heavy_max)
RES["real"] = dict(active=int(len(act)), light=int((strat == "light").sum()), heavy=int((strat == "heavy").sum()), platform=int((strat == "platform").sum()),
                   requests=int(len(real)), dur_median=round(float(np.median(durs)), 2), dur_p90=round(float(np.percentile(durs, 90)), 2))
log("real", RES["real"])
site_r = rt.Site(real.assign(status=200), W)

# ---- R1 / R2: random real pools through the frozen classifier
if ONLY & {"R1", "R2"}:
    RES["R1"] = {}; RES["R2"] = {}
    for st in ("light", "heavy"):
        g = f"real_{st}"
        if len(site_r.by.get(g, [])) < 80: RES["R1"][st] = "too few accounts"; continue
        NULL, sn = site_r.null(g, rng, n=NP)
        jobs = [(list(rng.choice(list(site_r.by[g]), 40, replace=False)), int(rng.integers(2**31))) for _ in range(NP)]
        G_.update(site=site_r, NULL=NULL, sn=sn)
        import multiprocessing as mp
        with mp.get_context("fork").Pool(2) as pool_:
            fs = pool_.map(_fp, jobs, chunksize=4)
        cls = [f["cls"] for f in fs]; ring = [f["ring_z"] for f in fs]; conc = [f["conc_z"] if f["conc_z"] is not None else 0.0 for f in fs]
        RES.setdefault("R1_pools", {})[st] = [dict(keys=j[0], cls=f["cls"], ring_z=f["ring_z"], conc_z=f["conc_z"]) for j, f in zip(jobs, fs)]
        vc = pd.Series(cls).value_counts().to_dict(); none = vc.get("none", 0) / NP
        RES["R1"][st] = dict(n=NP, frac_none=round(none, 3), classes=vc, threshold=0.95 if st == "light" else 0.90,
                             pass_=bool(none >= (0.95 if st == "light" else 0.90)),
                             null_R=dict(mean=round(float(NULL["R"].mean()), 4), sd=round(float(NULL["R"].std()), 4)))
        r6, c6 = float(np.mean(np.array(ring) > 6)), float(np.mean(np.array(conc) > 6))
        RES["R2"][st] = dict(frac_ring_gt6=round(r6, 3), frac_conc_gt6=round(c6, 3), pass_=bool(r6 <= 0.01 and c6 <= 0.01))
        log("R1/R2", st, RES["R1"][st], RES["R2"][st])

# ---- R3: replica coupling with real serving instances
if "R3" in ONLY and "inst" in real:
    top = real.groupby("key").inst.agg(lambda x: x.value_counts(normalize=True).iloc[0])
    home = real.groupby("key").inst.agg(lambda x: x.value_counts().index[0])
    one = top[top >= 0.9].index; res = {}
    for st in ("light", "heavy"):
        g = f"real_{st}"; ks_st = set(site_r.by.get(g, []))
        cand = pd.Series(home[one]); cand = cand[cand.index.isin(ks_st)]
        hosts = cand.value_counts(); hosts = hosts[hosts >= 40]
        if len(hosts) < 3: res[st] = dict(testable=False, instances_ge40=int(len(hosts))); continue
        NULL, sn = site_r.null(g, rng, n=NP); zs = []
        for inst_, n_ in hosts.items():
            pool_ = cand[cand == inst_].index.values
            for _ in range(min(10, max(1, n_ // 40 * 3))):
                _, a, s, e, ks = site_r.pool(rng.choice(pool_, 40, replace=False))
                R = float(occupancy(a, s, e, 0, W, None, n_accounts=40)["R"]); zs.append((NULL["R"].mean() - R) / NULL["R"].std())
        zs = np.array(zs)
        res[st] = dict(testable=True, instances=int(len(hosts)), pools=int(len(zs)), mean_z=round(float(zs.mean()), 2),
                       frac_abs_z_gt3=round(float((np.abs(zs) > 3).mean()), 3), pass_=bool(abs(zs.mean()) <= 0.5 and (np.abs(zs) > 3).mean() <= 0.05))
    RES["R3"] = res; log("R3", res)

# ---- injection
allr = rt.inject(real, durs, W, np.random.default_rng(args.seed + 1000)); site = rt.Site(allr, W)
RES["inject"] = {g: dict(ok=int(((allr.grp == g) & (allr.status == 200)).sum()), refused=int(((allr.grp == g) & (allr.status == 429)).sum()),
                         keys_used=int(allr[(allr.grp == g) & (allr.status == 200)].key.nunique())) for g in rt.INJECT}
log("inject", RES["inject"])

if "R4" in ONLY:
    med = {st: float(real[real.grp == f"real_{st}"].groupby("key").size().median()) for st in ("light", "heavy") if (real.grp == f"real_{st}").any()}
    inj_med = float(allr[allr.grp.str.startswith("I_") & (allr.status == 200)].groupby("key").size().median())
    st0 = min(med, key=lambda k: abs(med[k] - inj_med)); NULL, sn = site.null(f"real_{st0}", rng, n=NP)
    rows = {}
    for g, exp in rt.EXPECTED.items():
        ks = list(site.by.get(g, []))
        if len(ks) < 2: rows[g] = dict(cls="no keys", correct=False); continue
        tp = np.array([site.pos[site.kid[k]] for k in sorted(ks)])
        f = rt.fingerprint_pool(site, ks, NULL, sn, rng, true_pos=tp); f["expected"] = exp; f["correct"] = rt.ok_class(exp, f["cls"])
        rows[g] = f; log("R4", g, f["cls"], "OK" if f["correct"] else "WRONG", f"R={f['R']} zR={f['z_R']} ring={f['ring_z']} conc={f['conc_z']} K={f['K_little']}")
    n_ok = sum(r["correct"] for r in rows.values())
    RES["R4"] = dict(null_stratum=st0, pools=rows, correct=f"{n_ok}/{len(rows)}", pass_=bool(n_ok >= 10))

# ---- site-wide (R5, R8): all active accounts + injected keys, capped at nmax real accounts
real_keys = [k for k in site.keys if k.startswith("r")]
if len(real_keys) > args.nmax:
    keep_real = set(np.random.default_rng(args.seed + 2000).choice(real_keys, args.nmax, replace=False))
else:
    keep_real = set(real_keys)
RES["site_N_real"] = len(keep_real)
sub = allr[(~allr.key.str.startswith("r")) | allr.key.isin(keep_real)]
S2 = rt.Site(sub, W)
if ONLY & {"R5", "R8", "R9"}:
    lab8, ntr = fplib.spectral_rings_trimmed_v08(S2.A, S2.S, S2.E, len(S2.keys), 0, W, rng)
    rec, pur = rt.recovered(lab8, S2.grp, "I_rrshared")
    groups = []
    for c in np.unique(lab8[lab8 >= 0]):
        m = lab8 == c; comp = pd.Series(S2.grp[m]).value_counts().to_dict(); groups.append(comp)
    mixed = any(("I_rrshared" in g) and any(k.startswith("real") for k in g) for g in groups)
    RES["R5"] = dict(N_real=len(keep_real), trimmed=ntr, recall=round(rec, 3), purity=round(pur, 3), groups=groups,
                     pass_=bool(rec >= 0.9 and pur >= 0.9 and not mixed)); log("R5", RES["R5"])
    labg, kept = fplib.site_handoff_graph_coverage(S2.A, S2.S, S2.E, len(S2.keys), 0, W)
    ggroups = [pd.Series(S2.grp[labg == c]).value_counts().to_dict() for c in np.unique(labg[labg >= 0])]
    real_only = [g for g in groups if all(k.startswith("real") for k in g)]
    # R9: classify each real-only spectral group and test the uniformity of its request counts
    from scipy.stats import chi2 as _chi2
    NULLh, snh = S2.null("real_heavy", np.random.default_rng(args.seed + 3000), n=200)
    r9 = []
    for c in np.unique(lab8[lab8 >= 0]):
        mem = [S2.keys[i] for i in np.nonzero(lab8 == c)[0]]
        if not all(S2.grp[S2.kid[k]].startswith("real") for k in mem): continue
        f = rt.fingerprint_pool(S2, mem, NULLh, snh, np.random.default_rng(args.seed + 4000))
        n = S2.d[S2.d.key.isin(mem) & (S2.d.t_arrival >= 0)].groupby("key").size().values.astype(float)
        stat = float(((n - n.mean()) ** 2 / n.mean()).sum()); p_low = float(_chi2.cdf(stat, len(n) - 1))
        rot = f["cls"] in ("shared round-robin (closed loop)", "shared round-robin (open-loop relay)", "per-worker round-robin")
        r9.append(dict(size=len(mem), members=mem, cls=f["cls"], ring_z=f["ring_z"], conc_z=f["conc_z"], R=f["R"], z_R=f["z_R"],
                       req_per_acct=[float(n.min()), float(np.median(n)), float(n.max())], chi2=round(stat, 2), p_uniform_low=p_low,
                       rotation=rot, corroborated=bool(p_low < 0.01) if rot else None))
    RES["R9"] = dict(groups=r9, pass_=bool(all(g["corroborated"] for g in r9 if g["rotation"])) if any(g["rotation"] for g in r9) else None)
    log("R9", [{k: v for k, v in g.items() if k != "members"} for g in r9])
    real_only_g = [g for g in ggroups if all(k.startswith("real") for k in g)]
    RES["R8"] = dict(spectral_real_groups=real_only, graph_real_groups=real_only_g, graph_groups=ggroups,
                     graph_injected_recovery={g: round(rt.recovered(labg, S2.grp, g)[0], 2) for g in rt.INJECT})
    log("R8", {k: v for k, v in RES["R8"].items() if k != "graph_groups"})

# ---- R7: seed expansion among all (capped) active accounts
if "R7" in ONLY:
    spec = importlib.util.spec_from_file_location("an", "/tmp/claude-0/-home-claude/78f29ddf-32f5-54b2-bcc0-86fa7c435eb9/scratchpad/nb9/nb9an.py")
    an = importlib.util.module_from_spec(spec); spec.loader.exec_module(an)
    ex = an.seed_expansion(S2.A, S2.S, S2.E, len(S2.keys), 0, W, S2.grp, ["I_shuffle", "I_company", "I_hashed", "I_hedged", "I_rrworker", "I_sticky_ttl", "I_rot429_shared"],
                           rng, reps=5 if args.quick else 15, windows=(600, 1200))
    a20 = {p: (ex.get(p, {}).get("20min") or {}).get("auc") for p in ("I_shuffle", "I_company", "I_hashed", "I_hedged")}
    RES["R7"] = dict(expansion=ex, auc_20min=a20, pass_=bool(all((v or 0) >= 0.85 for v in a20.values()))); log("R7", a20)

# ---- R6: discoverability law on real noise
if "R6" in ONLY:
    cells = []
    for ring_name, jit in (("ring_J0", 0.0), ("ring_J6", 6.0)):
        ring = rt.simulate_pool(ring_name, dict(kind="rr_shared", K=8, M=40), durs, W, rng, jitter=jit)
        Ns = [n for n in (500, 2000) if n < len(real_keys)] + [min(len(real_keys), args.nmax)]
        for N in sorted(set(Ns)):
            bg = set(rng.choice(real_keys, N, replace=False))
            sub6 = pd.concat([real[real.key.isin(bg)], ring], ignore_index=True)
            S6 = rt.Site(sub6, W)
            th, p = rt.theta1(S6, ring_name, rng)
            bgm = np.isin(S6.grp, ["real_light", "real_heavy", "real_platform"])
            tau = fplib.key_corr_time(S6.A, S6.S, S6.E, len(S6.keys), 0, W); tau_c = float(np.mean(tau[bgm]))
            Tstar = 1.9 ** 2 * N * tau_c / th ** 2 if th > 0 else np.inf
            Tdet = None
            for Tm in (5, 10, 15, 20, 30, 45, 60):
                T = Tm * 60.0
                if T > W: break
                m = S6.S < T
                lab, _ = fplib.spectral_rings_trimmed_v08(S6.A[m], S6.S[m], np.minimum(S6.E[m], T), len(S6.keys), 0, T, rng)
                rec, pur = rt.recovered(lab, S6.grp, ring_name)
                if rec >= 0.9 and pur >= 0.9: Tdet = T; break
            within = (Tdet is not None and Tstar / 2 <= Tdet <= max(2 * Tstar, 300.0)) or (Tdet is None and 2 * Tstar > min(W, 3600))
            cells.append(dict(ring=ring_name, N=N, theta1=round(th, 2), duty=round(p, 3), tau_c=round(tau_c, 2), T_star_s=round(Tstar, 0),
                              T_det_s=Tdet, within_factor2=bool(within)))
            log("R6", cells[-1])
    frac = float(np.mean([c["within_factor2"] for c in cells])) if cells else 0.0
    RES["R6"] = dict(cells=cells, frac_within=round(frac, 2), pass_=bool(frac >= 0.75))

RES["runtime_min"] = round((time.time() - T0) / 60, 1)
json.dump(RES, open(args.out, "w"), indent=1, default=str)
def _passes(v):
    if not isinstance(v, dict): return None
    if "pass_" in v: return v["pass_"]
    return {k: _passes(x) for k, x in v.items() if isinstance(x, dict)}
log("done", {k: _passes(v) for k, v in RES.items() if k.startswith("R")})
