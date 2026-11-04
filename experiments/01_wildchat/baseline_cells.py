# %% [markdown]
# # Notebook 2 — Real human timing baseline (WildChat-1M)
# **Question.** Do real people, grouped the way a detector would group them, give an occupancy ratio near 1? This is the first go/no-go of the project.
#
# **Data.** WildChat-1M (about 840k ChatGPT conversations collected through a public chatbot, Apr 2023 – Apr 2024). Each assistant turn has the time its full response arrived. Accounts are approximated by the hashed IP address.
#
# **What it measures.**
# 1. Whether user turns carry timestamps, so request intervals can be observed directly rather than imputed from response length.
# 2. The null distribution of R and R60 for four kinds of benign pool: random active accounts, same country, same user agent + country (what blocking keys would produce), and heavy users.
# 3. A discovery scan: every user-agent + country group in every 6-hour window, scored against time-shift surrogates. Strongly negative z-scores are candidates for real hidden worker pools (or for confounds we must understand).
#
# **Go/no-go (pre-registered).** Stop or revise if more than 1% of benign 6-hour pools fall below R60 = 0.85.
#
# **Run settings.** Accelerator: *None*. Internet: *On*. Disk ≈ 4 GB. Expected runtime: 30–60 min.
# Paste back only the block between `===== PASTE FROM HERE =====` and `===== PASTE TO HERE =====`.

# %% [code]
# Cell 1 — settings
import os, sys, json, time, glob
T_START = time.time()
CFG = dict(
    repo="allenai/WildChat-1M",
    n_shards=14,                 # all shards; lower for a quick test
    data_dir=os.environ.get("NB2_DATA_DIR", ""),   # leave empty on Kaggle (downloads from Hugging Face)
    pool_size=40,
    n_windows_6h=400, n_windows_1h=400,
    tau=60.0,
    max_obs_duration_s=900.0,    # observed request durations above this are treated as missing
    scan_min_accounts=5, scan_min_requests=30, scan_surrogates=30,
    seed=2026,
)
print(json.dumps(CFG, indent=1))

# %% [code]
# Cell 2 — library (identical to Notebook 1's occlib)
LIB_PLACEHOLDER = True

# %% [code]
# Cell 3 — download WildChat-1M shards
import numpy as np, pandas as pd
import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
from occlib import *
RES = {"cfg": CFG}
if CFG["data_dir"]:
    files = sorted(glob.glob(os.path.join(CFG["data_dir"], "*.parquet")))[: CFG["n_shards"]]
else:
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    names = sorted(f.path for f in api.list_repo_tree(CFG["repo"], repo_type="dataset", recursive=True)
                   if getattr(f, "path", "").endswith(".parquet"))[: CFG["n_shards"]]
    files = []
    for n in names:
        files.append(hf_hub_download(CFG["repo"], n, repo_type="dataset"))
        print("downloaded", n, flush=True)
print(len(files), "shards")

# %% [code]
# Cell 4 — flatten the nested conversations into one row per turn (vectorised with pyarrow)
def parse_shard(path):
    pf = pq.ParquetFile(path)
    names = pf.schema_arrow.names
    want = [c for c in ["conversation_hash", "model", "conversation", "language", "country", "hashed_ip", "header"] if c in names]
    t = pq.read_table(path, columns=want)
    conv = t.column("conversation").combine_chunks()
    flat = pc.list_flatten(conv)
    parent = pc.list_parent_indices(conv).to_numpy()
    offs = conv.offsets.to_numpy()
    pos = np.arange(len(flat)) - offs[parent]
    def fld(arr, name):
        return arr.field(name)
    role = fld(flat, "role").to_numpy(zero_copy_only=False)
    ts = pc.cast(fld(flat, "timestamp"), pa.int64()).to_numpy(zero_copy_only=False)   # microseconds, NaN if null
    clen = pc.utf8_length(fld(flat, "content")).to_numpy(zero_copy_only=False)
    turn_ip = fld(flat, "hashed_ip").to_numpy(zero_copy_only=False)
    # conversation-level identity (fall back to the first user turn's values)
    def conv_col(name, sub=None):
        if name in t.column_names:
            c = t.column(name).combine_chunks()
            if sub is not None:
                c = c.field(sub)
            return c.to_numpy(zero_copy_only=False)
        return None
    ip = conv_col("hashed_ip")
    ua = conv_col("header", "user-agent")
    al = conv_col("header", "accept-language")
    if ip is None or ua is None:
        first_user = pd.Series(np.arange(len(flat)))[role == "user"].groupby(parent[role == "user"]).first()
        hdr = fld(flat, "header")
        u_ua = fld(hdr, "user-agent").to_numpy(zero_copy_only=False); u_al = fld(hdr, "accept-language").to_numpy(zero_copy_only=False)
        n = t.num_rows
        ip = np.full(n, None, object) if ip is None else ip
        ua = np.full(n, None, object) if ua is None else ua
        al = np.full(n, None, object) if al is None else al
        ip[first_user.index.values] = np.where(pd.isna(ip[first_user.index.values]), turn_ip[first_user.values], ip[first_user.index.values])
        ua[first_user.index.values] = u_ua[first_user.values]; al[first_user.index.values] = u_al[first_user.values]
    turns = pd.DataFrame(dict(conv=parent, pos=pos, role=role, ts=ts, clen=clen))
    convs = pd.DataFrame(dict(model=conv_col("model"), language=conv_col("language"), country=conv_col("country"),
                              ip=ip, ua=ua, al=al))
    return turns, convs

turn_parts, conv_parts, base = [], [], 0
for f in files:
    tu, cv = parse_shard(f)
    tu["conv"] += base; cv.index = np.arange(base, base + len(cv)); base += len(cv)
    turn_parts.append(tu); conv_parts.append(cv)
    print(os.path.basename(f), len(cv), "conversations", flush=True)
turns = pd.concat(turn_parts, ignore_index=True); convs = pd.concat(conv_parts)
del turn_parts, conv_parts
is_user = turns.role.values == "user"; is_asst = turns.role.values == "assistant"
data = dict(n_conversations=int(len(convs)), n_turns=int(len(turns)),
            user_turns=int(is_user.sum()), assistant_turns=int(is_asst.sum()),
            user_ts_present=round(float(np.isfinite(turns.ts.values[is_user].astype(float)).mean()), 4),
            assistant_ts_present=round(float(np.isfinite(turns.ts.values[is_asst].astype(float)).mean()), 4),
            ip_missing=round(float(pd.isna(convs.ip).mean()), 4),
            n_ips=int(convs.ip.nunique()), n_user_agents=int(convs.ua.nunique()), n_countries=int(convs.country.nunique()))
RES["data"] = data
print(json.dumps(data, indent=1))

# %% [code]
# Cell 5 — build request intervals: end = assistant timestamp; start = preceding user timestamp (observed) or imputed
turns = turns.sort_values(["conv", "pos"], kind="stable").reset_index(drop=True)
prev_role = np.r_[None, turns.role.values[:-1]]; prev_conv = np.r_[-1, turns.conv.values[:-1]]
prev_ts = np.r_[np.nan, turns.ts.values[:-1].astype(float)]
a = turns[(turns.role.values == "assistant")].copy()
ok_prev = (prev_role[a.index] == "user") & (prev_conv[a.index] == a.conv.values)
a["e"] = a.ts.astype(float) / 1e6
a["s_obs"] = np.where(ok_prev, prev_ts[a.index] / 1e6, np.nan)
a = a[np.isfinite(a.e.values)]
a = a.join(convs[["model", "language", "country", "ip", "ua", "al"]], on="conv")
a = a[~pd.isna(a.ip.values)]
dur_obs = a.e.values - a.s_obs.values
valid = np.isfinite(dur_obs) & (dur_obs > 0) & (dur_obs < CFG["max_obs_duration_s"])
cjk = a.language.isin(["Chinese", "Japanese", "Korean"]).values
tokens = np.where(cjk, a.clen.values / 1.1, a.clen.values / 4.0)
is_gpt4 = a.model.astype(str).str.contains("gpt-4").values
calib = {}
if valid.mean() > 0.2:
    # calibrate the imputation on observed durations: duration ~ TTFT + tokens / rate, robust per model family
    for name, m in [("gpt4", is_gpt4), ("gpt35", ~is_gpt4)]:
        mm = valid & m
        if mm.sum() > 1000:
            x = tokens[mm]; y = dur_obs[mm]
            q = np.quantile(x, [0.2, 0.8]); lo = (x < q[0]); hi = (x > q[1])
            rate = (np.median(x[hi]) - np.median(x[lo])) / max(np.median(y[hi]) - np.median(y[lo]), 1e-6)
            ttft = max(np.median(y[lo]) - np.median(x[lo]) / rate, 0.0)
            calib[name] = dict(rate_tok_s=round(float(rate), 1), ttft_s=round(float(ttft), 2), n=int(mm.sum()),
                               dur_median_s=round(float(np.median(y)), 1), dur_p90_s=round(float(np.quantile(y, 0.9)), 1))
rate = np.where(is_gpt4, calib.get("gpt4", {}).get("rate_tok_s", 20.0), calib.get("gpt35", {}).get("rate_tok_s", 50.0))
ttft = np.where(is_gpt4, calib.get("gpt4", {}).get("ttft_s", 1.0), calib.get("gpt35", {}).get("ttft_s", 0.8))
a["dur_imp"] = np.clip(ttft + tokens / np.maximum(rate, 1.0), 0.3, CFG["max_obs_duration_s"])
a["s_imp"] = a.e.values - a.dur_imp.values
INTERVAL = "observed" if valid.mean() > 0.8 else "imputed"
a["s"] = np.where(valid, a.s_obs.values, a.s_imp.values) if INTERVAL == "observed" else a.s_imp.values
acct_codes, acct_names = pd.factorize(a.ip.values)
a["acct"] = acct_codes
req = a[["acct", "s", "e", "s_imp", "country", "ua", "al", "model"]].sort_values("s").reset_index(drop=True)
RES["intervals"] = dict(choice=INTERVAL, observed_valid_frac=round(float(valid.mean()), 4), calibration=calib,
                        imputed_dur_median_s=round(float(np.median(a.dur_imp)), 1),
                        n_requests=int(len(req)), n_accounts=int(req.acct.nunique()),
                        span_days=round(float((req.s.max() - req.s.min()) / 86400), 1),
                        first=str(pd.to_datetime(req.s.min(), unit="s")), last=str(pd.to_datetime(req.s.max(), unit="s")))
print(json.dumps(RES["intervals"], indent=1))

# %% [code]
# Cell 6 — descriptive statistics of accounts
cnt = req.groupby("acct").size().values
# within-account overlap: share of requests that start while the same account already has one in flight
r2 = req.sort_values(["acct", "s"])
prev_e = np.r_[-np.inf, r2.e.values[:-1]]; same = np.r_[False, r2.acct.values[1:] == r2.acct.values[:-1]]
overlap = (same & (r2.s.values < prev_e)).mean()
hourly = req.groupby((req.s // 3600).astype(np.int64)).size()
RES["accounts"] = dict(req_per_acct_median=float(np.median(cnt)), req_per_acct_p99=float(np.quantile(cnt, 0.99)),
                       req_per_acct_max=int(cnt.max()), share_req_top1pct_accts=round(float(np.sort(cnt)[::-1][: max(1, len(cnt) // 100)].sum() / cnt.sum()), 3),
                       within_account_overlap_frac=round(float(overlap), 4),
                       requests_per_hour_median=float(hourly.median()), requests_per_hour_p95=float(hourly.quantile(0.95)))
print(json.dumps(RES["accounts"], indent=1))

# %% [code]
# Cell 7 — null distribution on real benign pools
rng = np.random.default_rng(CFG["seed"])
S_ALL, E_ALL, A_ALL = req.s.values, req.e.values, req.acct.values
t_min, t_max = S_ALL.min(), S_ALL.max()
MAXD = CFG["max_obs_duration_s"]

def window(t0, t1):
    i0 = np.searchsorted(S_ALL, t0 - MAXD); i1 = np.searchsorted(S_ALL, t1)
    sl = slice(i0, i1)
    m = E_ALL[sl] > t0
    return np.arange(i0, i1)[m]

def pool_stats(idx, t0, t1, tau, load):
    acc, _ = relabel(A_ALL[idx])
    s, e = S_ALL[idx], E_ALL[idx]
    o_full = occupancy(acc, s, e, t0, t1, None); o_hp = occupancy(acc, s, e, t0, t1, tau)
    fk = fano_khat(acc, s, e, t0, t1)
    return dict(R=o_full["R"], R60=o_hp["R"], n_req=int(((s >= t0) & (s < t1)).sum()), M=int(acc.max() + 1),
                K_hat=fk["K_hat"], load=load)

def sample_pools(hours, n_windows, M, min_M=10):
    out = {k: [] for k in ["random", "country", "ua_country", "heavy"]}
    L = hours * 3600.0
    for _ in range(n_windows):
        t0 = rng.uniform(t_min, t_max - L); t1 = t0 + L
        idx = window(t0, t1)
        if len(idx) < 20: continue
        w = req.iloc[idx]
        active = w[(w.s.values >= t0)]
        accts = active.acct.unique()
        load = int(len(active))
        if len(accts) < min_M: continue
        m = min(M, len(accts))
        pick = rng.choice(accts, m, replace=False)
        out["random"].append(pool_stats(idx[np.isin(A_ALL[idx], pick)], t0, t1, CFG["tau"], load))
        cc = active.groupby("country").acct.nunique(); cc = cc[cc >= min_M]
        if len(cc):
            c = rng.choice(cc.index.values)
            ac = active.acct[active.country.values == c].unique(); pick = rng.choice(ac, min(M, len(ac)), replace=False)
            out["country"].append(pool_stats(idx[np.isin(A_ALL[idx], pick)], t0, t1, CFG["tau"], load))
        g = active.groupby(["ua", "country"]).acct.nunique(); g = g[g >= 5]
        if len(g):
            key = g.index[rng.integers(len(g))]
            ac = active.acct[(active.ua.values == key[0]) & (active.country.values == key[1])].unique()
            pick = ac if len(ac) <= M else rng.choice(ac, M, replace=False)
            out["ua_country"].append(pool_stats(idx[np.isin(A_ALL[idx], pick)], t0, t1, CFG["tau"], load))
        top = active.groupby("acct").size().sort_values(ascending=False).index.values[: 2 * m]
        pick = rng.choice(top, m, replace=False)
        out["heavy"].append(pool_stats(idx[np.isin(A_ALL[idx], pick)], t0, t1, CFG["tau"], load))
    return out

def summarise(rows):
    if not rows: return None
    from scipy.stats import spearmanr
    R = np.array([r["R"] for r in rows]); R60 = np.array([r["R60"] for r in rows])
    n = np.array([r["n_req"] for r in rows]); Ms = np.array([r["M"] for r in rows]); load = np.array([r["load"] for r in rows])
    ok = np.isfinite(R60) & np.isfinite(R)
    R, R60, n, Ms, load = R[ok], R60[ok], n[ok], Ms[ok], load[ok]
    rho = spearmanr(R60, load)[0] if len(R60) > 10 else np.nan
    return dict(n=int(ok.sum()), M_median=int(np.median(Ms)), req_median=int(np.median(n)),
                R_mean=round(float(R.mean()), 4), R_sd=round(float(R.std()), 4),
                R60_mean=round(float(R60.mean()), 4), R60_sd=round(float(R60.std()), 4),
                R60_q01=round(float(np.quantile(R60, 0.01)), 4), R60_min=round(float(R60.min()), 4),
                frac_R60_below_085=round(float((R60 < 0.85).mean()), 4), frac_R60_below_090=round(float((R60 < 0.90).mean()), 4),
                spearman_R60_vs_site_load=round(float(rho), 3))

null_real = {}
for hours, nw in [(6, CFG["n_windows_6h"]), (1, CFG["n_windows_1h"])]:
    pools = sample_pools(hours, nw, CFG["pool_size"])
    null_real[f"{hours}h"] = {k: summarise(v) for k, v in pools.items()}
    print(hours, "h:", json.dumps(null_real[f"{hours}h"], indent=1), flush=True)
RES["null_real"] = null_real
six = null_real["6h"]
RES["go_no_go"] = {k: (None if v is None else ("GO" if v["frac_R60_below_085"] <= 0.01 else "REVISE")) for k, v in six.items()}
print("go/no-go (6 h, R60 < 0.85 in <= 1% of pools):", RES["go_no_go"])

# %% [code]
# Cell 8 — robustness of the null to how request start times are obtained (random pools, 6 h)
def null_with_starts(s_alt, n_windows=150):
    global S_ALL, E_ALL, A_ALL, req
    bak = (S_ALL, E_ALL, A_ALL, req)
    order = np.argsort(s_alt)
    req = req.iloc[order].reset_index(drop=True); req["s"] = s_alt[order]
    S_ALL, E_ALL, A_ALL = req.s.values, req.e.values, req.acct.values
    out = summarise(sample_pools(6, n_windows, CFG["pool_size"])["random"])
    S_ALL, E_ALL, A_ALL, req = bak
    return out
rob = {}
dur_imp = req.e.values - req.s_imp.values
if INTERVAL == "observed":
    rob["imputed_starts"] = null_with_starts(req.s_imp.values)
rob["imputed_durations_x0.5"] = null_with_starts(req.e.values - 0.5 * dur_imp)
rob["imputed_durations_x2"] = null_with_starts(req.e.values - 2.0 * dur_imp)
RES["robustness_6h_random"] = rob
print(json.dumps(rob, indent=1))

# %% [code]
# Cell 9 — discovery scan: every user-agent + country group, every 6-hour window, vs time-shift surrogates
L = 6 * 3600.0
starts = np.arange(np.floor(t_min / L) * L, t_max, L)
cands = []; zs = []; flagged_accts = set()
t_scan = time.time()
for t0 in starts:
    t1 = t0 + L
    idx = window(t0, t1)
    if len(idx) < CFG["scan_min_requests"]: continue
    w = req.iloc[idx]
    act = w[w.s.values >= t0]
    g = act.groupby(["ua", "country"]).agg(n_acct=("acct", "nunique"), n_req=("acct", "size"))
    g = g[(g.n_acct >= CFG["scan_min_accounts"]) & (g.n_req >= CFG["scan_min_requests"])]
    for (u, c), row in g.iterrows():
        accs = act.acct[(act.ua.values == u) & (act.country.values == c)].unique()
        sub = idx[np.isin(A_ALL[idx], accs)]
        acc, _ = relabel(A_ALL[sub]); s, e = S_ALL[sub], E_ALL[sub]
        r60 = occupancy(acc, s, e, t0, t1, CFG["tau"])["R"]
        if not np.isfinite(r60): continue
        sur = surrogate_null(acc, s, e, t0, t1, CFG["tau"], CFG["scan_surrogates"], rng)
        sd = sur.std() if sur.std() > 0 else np.nan
        z = (sur.mean() - r60) / sd
        zs.append(z)
        if np.isfinite(z) and z > 5:
            flagged_accts.update(map(int, accs))
        cands.append(dict(window_start=str(pd.to_datetime(t0, unit="s")), country=c, ua=str(u)[:60], n_acct=int(row.n_acct),
                          n_req=int(row.n_req), R60=round(float(r60), 3), R=round(float(occupancy(acc, s, e, t0, t1, None)["R"]), 3),
                          z=round(float(z), 2), K_hat=round(float(fano_khat(acc, s, e, t0, t1)["K_hat"]), 1)))
zs = np.array(zs); zs = zs[np.isfinite(zs)]
top = sorted(cands, key=lambda d: -d["z"] if np.isfinite(d["z"]) else 0)[:25]
RES["scan"] = dict(n_groups_tested=int(len(zs)), runtime_min=round((time.time() - t_scan) / 60, 1),
                   frac_z_gt_3=round(float((zs > 3).mean()), 5), frac_z_gt_4=round(float((zs > 4).mean()), 5),
                   frac_z_gt_5=round(float((zs > 5).mean()), 5), n_z_gt_5=int((zs > 5).sum()),
                   frac_z_lt_minus3=round(float((zs < -3).mean()), 5),
                   expected_frac_gt_3_if_normal=0.00135, top=top)
print(json.dumps({k: v for k, v in RES["scan"].items() if k != "top"}, indent=1))
print(pd.DataFrame(top).to_string())
pd.DataFrame(cands).to_csv("nb2_scan_all_groups.csv", index=False)

# %% [code]
# Cell 9b — benign null again, with every account from a z > 5 group removed (guards against contamination)
RES["scan"]["n_flagged_accounts"] = len(flagged_accts)
if flagged_accts:
    keep = ~np.isin(req.acct.values, np.fromiter(flagged_accts, int))
    bak = (S_ALL, E_ALL, A_ALL, req)
    req = req[keep].reset_index(drop=True); S_ALL, E_ALL, A_ALL = req.s.values, req.e.values, req.acct.values
    clean = {f"{h}h": {k: summarise(v) for k, v in sample_pools(h, nw, CFG["pool_size"]).items()}
             for h, nw in [(6, CFG["n_windows_6h"]), (1, CFG["n_windows_1h"])]}
    S_ALL, E_ALL, A_ALL, req = bak
    RES["null_real_excluding_flagged"] = clean
    RES["go_no_go_excluding_flagged"] = {k: (None if v is None else ("GO" if v["frac_R60_below_085"] <= 0.01 else "REVISE"))
                                         for k, v in clean["6h"].items()}
    print(json.dumps(clean, indent=1)); print("go/no-go excluding flagged:", RES["go_no_go_excluding_flagged"])
else:
    print("no flagged accounts")

# %% [code]
# Cell 10 — summary to paste back
RES["runtime_min"] = round((time.time() - T_START) / 60, 1)
with open("nb2_results.json", "w") as f:
    json.dump(RES, f, indent=1, default=str)
print("===== PASTE FROM HERE =====")
print(json.dumps(RES, separators=(",", ":"), default=str))
print("===== PASTE TO HERE =====")
