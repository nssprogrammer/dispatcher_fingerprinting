# %% [markdown]
# # Notebook 4 — Closing the WildChat analysis: a server-cap test, and a scan with exact p-values
# Notebook 3 left three problems. This notebook fixes them and closes the WildChat thread.
#
# **A. Is there a site-wide concurrency cap?** WildChat gives only response end times, so intervals are *generation only*. If WildChat's backend processed only a few requests at once, the whole site would take turns, and pools that contain most of the site (1-hour pools) would read R < 1. Tests: the site-wide Fano factor and the share of time with 2+ requests in flight, against the Poisson expectation, by load decile; plus the same with durations halved.
#
# **B. A scan whose error rate we can trust.**
# - Exact one-sided surrogate p-values (100 time-shifts, escalated to 1,100 when p ≤ 0.05) instead of z-scores.
# - *Diverse* controls: activity-matched accounts drawn from different user-agent+country cells and different persons, so a control cannot contain one operator's IPs.
# - Benjamini–Hochberg false-discovery control.
# - A corrected oracle: a prompt prefix is 'generic' only if it is used in ≥ 5 different user-agent+country cells (in Notebook 3 an operator with > 20 IPs made its own template 'generic').
#
# **Pre-registered reading.** Controls should give p-values close to uniform (≈1% below 0.01). The scan supports real hidden operators only if real groups yield BH discoveries while controls yield about none, and the oracle rate among discoveries exceeds the rate among non-discoveries.
#
# **Run settings.** Accelerator *None*, Internet *On*. Uses all CPU cores. Expected runtime 25–45 min.

# %% [code]
# Cell 1 — settings
import os, sys, json, time, glob
T_START = time.time()
CFG = dict(repo="allenai/WildChat-1M", n_shards=14, data_dir=os.environ.get("NB4_DATA_DIR", ""),
           tau=60.0, max_dur_s=900.0, max_component_ips=50,
           generic_cells=5,                 # a prefix used in >= this many user-agent+country cells is generic
           scan_min_accounts=5, scan_min_requests=30,
           n_sur_initial=100, n_sur_full=1000, escalate_p=0.05, fdr_q=0.05,
           n_jobs=os.cpu_count(), seed=11)
print(json.dumps(CFG, indent=1))

# %% [code]
# Cell 2 — library
LIB_PLACEHOLDER = True

# %% [code]
# Cell 3 — download
import numpy as np, pandas as pd
import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from occlib import *
RES = {"cfg": CFG}
if CFG["data_dir"]:
    files = sorted(glob.glob(os.path.join(CFG["data_dir"], "*.parquet")))[: CFG["n_shards"]]
else:
    from huggingface_hub import HfApi, hf_hub_download
    names = sorted(f.path for f in HfApi().list_repo_tree(CFG["repo"], repo_type="dataset", recursive=True)
                   if getattr(f, "path", "").endswith(".parquet"))[: CFG["n_shards"]]
    files = [hf_hub_download(CFG["repo"], n, repo_type="dataset") for n in names]
print(len(files), "shards")

# %% [code]
# Cell 4 — parse: one row per turn, keeping each user turn's own IP and a normalised prompt prefix
def norm_prefix(content, n):
    x = pc.utf8_slice_codeunits(content, 0, 300)
    x = pc.utf8_lower(x)
    x = pc.replace_substring_regex(x, r"[0-9]+", "0")
    x = pc.replace_substring_regex(x, r"\s+", " ")
    x = pc.utf8_trim_whitespace(x)
    return pc.utf8_slice_codeunits(x, 0, n)

def parse_shard(path):
    names = pq.ParquetFile(path).schema_arrow.names
    t = pq.read_table(path, columns=[c for c in ["model", "conversation", "language", "country", "hashed_ip", "header"] if c in names])
    conv = t.column("conversation").combine_chunks()
    flat = pc.list_flatten(conv)
    parent = pc.list_parent_indices(conv).to_numpy()
    pos = np.arange(len(flat)) - conv.offsets.to_numpy()[parent]
    content = flat.field("content")
    turns = pd.DataFrame(dict(conv=parent, pos=pos, role=flat.field("role").to_numpy(zero_copy_only=False),
                              ts=pc.cast(flat.field("timestamp"), pa.int64()).to_numpy(zero_copy_only=False),
                              clen=pc.utf8_length(content).to_numpy(zero_copy_only=False),
                              tip=flat.field("hashed_ip").to_numpy(zero_copy_only=False),
                              p64=norm_prefix(content, 64).to_numpy(zero_copy_only=False)))
    hdr = t.column("header").combine_chunks()
    convs = pd.DataFrame(dict(model=t.column("model").to_numpy(), language=t.column("language").to_numpy(),
                              country=t.column("country").to_numpy(), ip=t.column("hashed_ip").to_numpy(),
                              ua=hdr.field("user-agent").to_numpy(zero_copy_only=False)))
    return turns, convs

tp, cp, base = [], [], 0
for f in files:
    tu, cv = parse_shard(f); tu["conv"] += base; cv.index = np.arange(base, base + len(cv)); base += len(cv)
    tp.append(tu); cp.append(cv)
turns = pd.concat(tp, ignore_index=True); convs = pd.concat(cp); del tp, cp
turns = turns.sort_values(["conv", "pos"], kind="stable").reset_index(drop=True)
print(len(convs), "conversations,", len(turns), "turns")

# %% [code]
# Cell 5 — requests (end = assistant timestamp; start imputed from response length, as in Notebook 2)
prev = lambda col: np.r_[[None], turns[col].values[:-1]]
a = turns[turns.role.values == "assistant"].copy()
pr, pc_, ptip, pp = prev("role")[a.index], np.r_[-1, turns.conv.values[:-1]][a.index], prev("tip")[a.index], prev("p64")[a.index]
okp = (pr == "user") & (pc_ == a.conv.values)
a["p64"] = np.where(okp, pp, None); a["tip"] = np.where(okp, ptip, None)
a["e"] = a.ts.astype(float) / 1e6
a = a[np.isfinite(a.e.values)].join(convs, on="conv")
a = a[~pd.isna(a.ip.values)]
cjk = a.language.isin(["Chinese", "Japanese", "Korean"]).values
tok = np.where(cjk, a.clen.values / 1.1, a.clen.values / 4.0)
g4 = a.model.astype(str).str.contains("gpt-4").values
a["s"] = a.e.values - np.clip(np.where(g4, 1.0, 0.8) + tok / np.where(g4, 20.0, 50.0), 0.3, CFG["max_dur_s"])
ip_codes, ip_names = pd.factorize(a.ip.values)
a["acct"] = ip_codes
req = a[["acct", "conv", "s", "e", "country", "ua", "p64", "tip", "ip"]].sort_values("s").reset_index(drop=True)
print(len(req), "requests,", req.acct.nunique(), "IP accounts")

# %% [code]
# Cell 6 — H1: identity resolution. Link IPs that appear in the same conversation; components = 'persons'
ip_index = pd.Index(ip_names)
u = turns[(turns.role.values == "user") & ~pd.isna(turns.tip.values)][["conv", "tip"]]
u = u[u.conv.isin(req.conv.unique())]
u["conv_ip"] = convs.ip.reindex(u.conv.values).values
u = u[~pd.isna(u.conv_ip.values)]
i_a = ip_index.get_indexer(u.conv_ip.values); i_b = ip_index.get_indexer(u.tip.values)
ok = (i_a >= 0) & (i_b >= 0)
multi_ip_conv = u.groupby("conv").tip.nunique()
n_ip = len(ip_index)
G = coo_matrix((np.ones(ok.sum()), (i_a[ok], i_b[ok])), shape=(n_ip, n_ip))
ncomp, comp = connected_components(G, directed=False)
csize = np.bincount(comp)
big = csize[comp] > CFG["max_component_ips"]
person = np.where(big, ncomp + np.arange(n_ip), comp)          # IPs in giant components stay separate
person = pd.factorize(person)[0]
req["person"] = person[req.acct.values]
RES["identity"] = dict(conversations_with_multiple_user_ips=round(float((multi_ip_conv > 1).mean()), 4),
                       n_ip_accounts=int(n_ip), n_persons=int(req.person.nunique()),
                       components_with_2plus_ips=int((csize >= 2).sum()), largest_component_ips=int(csize.max()),
                       ips_in_components_over_cap=int(big.sum()),
                       share_requests_from_multi_ip_persons=round(float((np.bincount(person)[req.person.values] > 1).mean()), 4))
print(json.dumps(RES["identity"], indent=1))


# %% [code]
# Cell 7 — A: site-wide concurrency, hour by hour (all accounts together)
S, E = req.s.values, req.e.values
t_min, t_max = S.min(), S.max()
def window(t0, t1, S_=None, E_=None):
    S_ = S if S_ is None else S_; E_ = E if E_ is None else E_
    i0 = np.searchsorted(S_, t0 - CFG["max_dur_s"]); i1 = np.searchsorted(S_, t1)
    ix = np.arange(i0, i1)
    return ix[E_[ix] > t0]

def site_hours(S_, E_, order):
    rows = []
    for t0 in np.arange(np.floor(t_min / 3600) * 3600, t_max, 3600.0):
        t1 = t0 + 3600
        ix = window(t0, t1, S_, E_)
        n_req = int((S_[ix] >= t0).sum())
        if n_req < 20: continue
        acc, _ = relabel(req.acct.values[order][ix]); s, e = S_[ix], E_[ix]
        # time-share with 1+ and 2+ requests in flight, site-wide
        tt = np.concatenate([np.clip(s, t0, t1), np.clip(e, t0, t1)]); dd = np.concatenate([np.ones(len(s)), -np.ones(len(e))])
        oo = np.lexsort((dd, tt)); lvl = np.cumsum(dd[oo]); dt = np.diff(np.append(tt[oo], t1))
        mean_N = float((lvl * dt).sum() / 3600)
        frac2 = float(dt[lvl >= 2].sum() / 3600)
        pois2 = 1 - np.exp(-mean_N) * (1 + mean_N)
        rows.append(dict(load=n_req, mean_N=mean_N, frac_2plus=frac2, poisson_2plus=pois2,
                         fano=fano_khat(acc, s, e, t0, t1)["F"], R60=occupancy(acc, s, e, t0, t1, CFG["tau"])["R"],
                         peak=int(lvl.max())))
    return pd.DataFrame(rows)

def site_summary(df):
    from scipy.stats import spearmanr
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    df["decile"] = pd.qcut(df.load, 10, labels=False, duplicates="drop")
    by = df.groupby("decile").agg(load=("load", "median"), mean_N=("mean_N", "median"), fano=("fano", "median"),
                                  R60=("R60", "median"), frac_2plus=("frac_2plus", "median"),
                                  poisson_2plus=("poisson_2plus", "median"), peak_p99=("peak", lambda x: float(np.quantile(x, 0.99))))
    return dict(n_hours=len(df), fano_median=round(float(df.fano.median()), 3), R60_median=round(float(df.R60.median()), 3),
                ratio_obs_to_poisson_2plus=round(float(df.frac_2plus.sum() / df.poisson_2plus.sum()), 3),
                spearman_fano_vs_load=round(float(spearmanr(df.fano, df.load)[0]), 3),
                spearman_R60_vs_load=round(float(spearmanr(df.R60, df.load)[0]), 3),
                by_load_decile=by.round(3).to_dict(orient="list"))

order_id = np.arange(len(req))
site = site_hours(S, E, order_id)
dur = E - S
S_half = E - 0.5 * dur; o_half = np.argsort(S_half)
site_half = site_hours(S_half[o_half], E[o_half], o_half)
RES["A_site_wide"] = dict(imputed=site_summary(site), durations_halved=site_summary(site_half))
print(json.dumps(RES["A_site_wide"], indent=1))

# %% [code]
# Cell 8 — oracle v2: generic = used in >= 5 user-agent+country cells
p32 = req.p64.astype(str).str.slice(0, 32)
req["p32"] = np.where(pd.isna(req.p64.values) | (p32.str.len() < 12), None, p32)
cell = req.ua.astype(str) + "|" + req.country.astype(str)
cells_per_prefix = pd.Series(cell.values)[pd.notna(req.p32.values)].groupby(req.p32.dropna().values).nunique()
generic = set(cells_per_prefix[cells_per_prefix >= CFG["generic_cells"]].index)
req["tmpl"] = [None if (p is None or p in generic) else p for p in req.p32.values]
RES["oracle_v2"] = dict(n_prefixes=int(len(cells_per_prefix)), n_generic=len(generic),
                        share_requests_with_template=round(float(pd.notna(req.tmpl).mean()), 4))
print(RES["oracle_v2"])

def oracle(sub):
    t = sub.dropna(subset=["tmpl"])
    tmpl_share = float((t.tmpl.map(t.groupby("tmpl").acct.nunique()).values >= 2).mean()) if len(t) else 0.0
    pc_ = sub.groupby("person").acct.nunique()
    linked = float(sub.acct.isin(sub.acct[sub.person.map(pc_).values >= 2]).mean()) if len(sub) else 0.0
    return tmpl_share, linked

# %% [code]
# Cell 9 — B: scan with exact surrogate p-values; diverse activity-matched controls; parallel over windows
A_ = req.acct.values; P_ = req.person.values; UA_ = req.ua.astype(str).values; CT_ = req.country.astype(str).values
CELL_ = cell.values

def perm_p(acc, s, e, t0, t1, rng):
    r = occupancy(acc, s, e, t0, t1, CFG["tau"])["R"]
    if not np.isfinite(r): return np.nan, np.nan, 0
    sur = surrogate_null(acc, s, e, t0, t1, CFG["tau"], CFG["n_sur_initial"], rng)
    k = int((sur <= r).sum()); n = len(sur)
    if (k + 1) / (n + 1) <= CFG["escalate_p"]:
        sur2 = surrogate_null(acc, s, e, t0, t1, CFG["tau"], CFG["n_sur_full"], rng)
        k += int((sur2 <= r).sum()); n += len(sur2)
    return r, (k + 1) / (n + 1), n

def scan_window(t0):
    rng = np.random.default_rng([CFG["seed"], int(t0)])
    t1 = t0 + 6 * 3600.0
    ix = window(t0, t1)
    out = []
    act_ix = ix[S[ix] >= t0]
    if len(act_ix) < CFG["scan_min_requests"]: return out
    act = pd.DataFrame(dict(acct=A_[act_ix], person=P_[act_ix], ua=UA_[act_ix], ct=CT_[act_ix], cell=CELL_[act_ix]))
    cnt = act.acct.value_counts(); rank = cnt.rank(method="first", pct=True)
    acct_cell = act.drop_duplicates("acct").set_index("acct")[["cell", "person"]]
    g = act.groupby(["ua", "ct"]).agg(n_acct=("acct", "nunique"), n_req=("acct", "size"))
    g = g[(g.n_acct >= CFG["scan_min_accounts"]) & (g.n_req >= CFG["scan_min_requests"])]
    for (u_, c_), r_ in g.iterrows():
        real = act.acct[(act.ua.values == u_) & (act.ct.values == c_)].unique()
        # diverse control: activity-matched, each account from a different cell and person, none from this cell
        cand = rank.drop(index=real, errors="ignore")
        cand = cand[acct_cell.cell.reindex(cand.index).values != f"{u_}|{c_}"]
        ctrl, used_cells, used_persons, tol_used = [], set(), set(), 0.0
        for a_ in real:
            pick = None
            for tol in (0.05, 0.1, 0.2, 0.4):
                near = rng.permutation(cand.index.values[np.abs(cand.values - rank[a_]) <= tol])
                for b in near:
                    cc, pp = acct_cell.at[b, "cell"], acct_cell.at[b, "person"]
                    if cc not in used_cells and pp not in used_persons and b not in ctrl:
                        pick = b; break
                if pick is not None:
                    tol_used = max(tol_used, tol); break
            if pick is None: break
            ctrl.append(pick); used_cells.add(acct_cell.at[pick, "cell"]); used_persons.add(acct_cell.at[pick, "person"])
        groups = [("real", real)] + ([("control", np.array(ctrl))] if len(ctrl) == len(real) else [])
        for kind, accts in groups:
            sub_ix = ix[np.isin(A_[ix], accts)]
            acc, _ = relabel(A_[sub_ix]); s, e = S[sub_ix], E[sub_ix]
            nreq = int((s >= t0).sum())
            if nreq < CFG["scan_min_requests"]: continue
            R60, p, nsur = perm_p(acc, s, e, t0, t1, rng)
            if not np.isfinite(p): continue
            fk = fano_khat(acc, s, e, t0, t1)
            ts_, lk = oracle(req.iloc[sub_ix[s >= t0]])
            out.append(dict(kind=kind, t0=t0, match_tol=tol_used if kind == "control" else 0.0, ua=str(u_)[:60], country=c_, n_acct=len(accts), n_req=nreq, R60=R60, p=p,
                            n_sur=nsur, K_hat=fk["K_hat"], fano=fk["F"], tmpl_share=ts_, linked=lk, accts=list(map(int, accts))))
    return out

import multiprocessing as mp
starts = np.arange(np.floor(t_min / 21600) * 21600, t_max, 21600.0)
t_scan = time.time()
ctx = mp.get_context("fork")
with ctx.Pool(CFG["n_jobs"]) as pool:
    parts = pool.map(scan_window, starts, chunksize=8)
scan = pd.DataFrame([r for part in parts for r in part])
real, ctrl = scan[scan.kind == "real"].copy(), scan[scan.kind == "control"].copy()
print(f"scan: {len(real)} real, {len(ctrl)} control groups in {(time.time() - t_scan) / 60:.1f} min")

# %% [code]
# Cell 10 — B analysis: calibration, FDR, oracle enrichment
from sklearn.metrics import roc_auc_score
def bh(p, q):
    p = np.asarray(p); n = len(p); o = np.argsort(p)
    thr = q * np.arange(1, n + 1) / n
    passed = p[o] <= thr
    k = np.max(np.nonzero(passed)[0]) + 1 if passed.any() else 0
    out = np.zeros(n, bool); out[o[:k]] = True
    return out
def calib(p):
    p = np.asarray(p)
    return {f"p<{a}": round(float((p < a).mean()), 4) for a in (0.05, 0.01, 0.002)}
for df in (real, ctrl):
    df["oracle_pos"] = (df.tmpl_share.values >= 0.3) | (df.linked.values >= 0.5)
real["disc"] = bh(real.p.values, CFG["fdr_q"]); ctrl["disc"] = bh(ctrl.p.values, CFG["fdr_q"])
def auc(y, x):
    y = np.asarray(y, bool)
    return round(float(roc_auc_score(y, x)), 3) if 0 < y.sum() < len(y) else None
B = dict(n_real=len(real), n_control=len(ctrl),
         calibration_control=calib(ctrl.p), calibration_real=calib(real.p),
         control_match_tol_share={str(k): round(float(v), 3) for k, v in ctrl.match_tol.value_counts(normalize=True).items()},
         min_p_resolution=round(1 / (CFG["n_sur_initial"] + CFG["n_sur_full"] + 1), 5),
         bh_discoveries_real=int(real.disc.sum()), bh_discoveries_control=int(ctrl.disc.sum()),
         oracle_rate_real=round(float(real.oracle_pos.mean()), 4), oracle_rate_control=round(float(ctrl.oracle_pos.mean()), 4),
         oracle_rate_discoveries=round(float(real.oracle_pos[real.disc].mean()), 4) if real.disc.any() else None,
         oracle_rate_nondiscoveries=round(float(real.oracle_pos[~real.disc].mean()), 4),
         AUC_minuslogp_vs_oracle=auc(real.oracle_pos, -np.log(real.p)), AUC_minuslogp_vs_template=auc(real.tmpl_share >= 0.3, -np.log(real.p)),
         AUC_minuslogp_vs_linked=auc(real.linked >= 0.5, -np.log(real.p)),
         AUC_nreq_vs_oracle=auc(real.oracle_pos, real.n_req), AUC_minusFano_vs_oracle=auc(real.oracle_pos, -real.fano))
RES["B_scan"] = B
print(json.dumps(B, indent=1))

# %% [code]
# Cell 11 — case files for BH discoveries (joined across windows by shared accounts)
disc = real[real.disc]
parent_ = {}
def find(x):
    while parent_.get(x, x) != x:
        parent_[x] = parent_.get(parent_[x], parent_[x]); x = parent_[x]
    return x
for accts in disc.accts:
    r0 = find(accts[0])
    for a_ in accts[1:]:
        ra = find(a_)
        if ra != r0: parent_[ra] = r0
disc = disc.assign(entity=[find(a[0]) for a in disc.accts])
cases = []
for ent, grp in disc.groupby("entity"):
    accts = sorted(set(a_ for x in grp.accts for a_ in x))
    # restrict to the discovered windows
    m = np.zeros(len(req), bool)
    for t0 in grp.t0.unique():
        ix = window(t0, t0 + 21600); m[ix[np.isin(A_[ix], accts)]] = True
    sub = req[m]
    tt = np.concatenate([sub.s.values, sub.e.values]); dd = np.concatenate([np.ones(len(sub)), -np.ones(len(sub))])
    oo = np.lexsort((dd, tt)); lvl = np.cumsum(dd[oo]); dtt = np.diff(np.append(tt[oo], tt[oo][-1]))
    busy = dtt[lvl >= 1].sum(); multi = dtt[lvl >= 2].sum()
    cases.append(dict(n_windows=len(grp), n_ips=len(accts), n_persons=int(sub.person.nunique()), n_requests=len(sub),
                      first_window=str(pd.to_datetime(grp.t0.min(), unit="s")), last_window=str(pd.to_datetime(grp.t0.max(), unit="s")),
                      country=grp.country.iloc[0], ua=grp.ua.iloc[0][:50], p_min=float(grp.p.min()),
                      R60_median=round(float(grp.R60.median()), 3), K_hat_median=round(float(grp.K_hat.replace(np.inf, np.nan).median()), 2),
                      share_time_2plus_in_flight=round(float(multi / busy), 4) if busy > 0 else None,
                      req_per_active_hour=round(float(len(sub) / max(1, pd.to_datetime(sub.s, unit="s").dt.floor("h").nunique())), 1),
                      tmpl_share=round(float(grp.tmpl_share.mean()), 3), linked=round(float(grp.linked.mean()), 3),
                      top_prefixes={k[:40]: int(v) for k, v in sub.p32.dropna().value_counts().head(3).items()}))
cases = sorted(cases, key=lambda d: -d["n_requests"])[:15]
RES["B_cases"] = cases
if cases:
    print(pd.DataFrame(cases).drop(columns=["top_prefixes"]).to_string())

# %% [code]
# Cell 12 — summary to paste back
real.drop(columns=["accts"]).to_csv("nb4_scan_real.csv", index=False); ctrl.drop(columns=["accts"]).to_csv("nb4_scan_control.csv", index=False)
site.to_csv("nb4_site_hours.csv", index=False)
RES["runtime_min"] = round((time.time() - T_START) / 60, 1)
json.dump(RES, open("nb4_results.json", "w"), indent=1, default=str)
print("===== PASTE FROM HERE =====")
print(json.dumps(RES, separators=(",", ":"), default=str))
print("===== PASTE TO HERE =====")
