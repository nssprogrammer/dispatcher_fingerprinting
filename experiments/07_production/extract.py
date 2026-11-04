"""NB10 extractor: three 1-hour windows from the Chutes production trace, compact parquet + profile."""
import os, json, time, math
import numpy as np, pandas as pd
import pyarrow as pa, pyarrow.parquet as pq

REPO = "harvardMadsys/chutes_metadata"
WIN_S = 3600.0          # analysis window
PAD_S = 1800.0          # keep requests that started up to 30 min before the window (in flight at its start)
MAX_ROWS = 2_000_000    # shorten the window if a single hour has more requests than this
COLS = ["user_id", "rehash_round", "instance_id", "chute_id", "function_name", "started_at", "completed_at", "ttft", "it", "ot", "ct"]


def list_shards():
    from huggingface_hub import HfApi
    files = HfApi().list_repo_files(REPO, repo_type="dataset")
    return sorted(f for f in files if f.startswith("data/trace/") and f.endswith(".parquet"))


def open_parquet(path):
    """remote row-group reads via HfFileSystem; local paths open directly"""
    if os.path.exists(path):
        return pq.ParquetFile(path), "local"
    from huggingface_hub import HfFileSystem
    fs = HfFileSystem()
    f = fs.open(f"datasets/{REPO}/{path}", "rb", block_size=64 * 2**20)
    return pq.ParquetFile(f), "remote"


def _us(v):
    """statistics value -> int64 microseconds"""
    if v is None: return None
    if isinstance(v, (int, np.integer)): return int(v)
    return int(pd.Timestamp(v).value // 1000)


def rowgroup_spans(pf, col="started_at"):
    md = pf.metadata; j = pf.schema_arrow.get_field_index(col); out = []
    for i in range(md.num_row_groups):
        st = md.row_group(i).column(j).statistics
        if st is None or not st.has_min_max: return None
        out.append((i, _us(st.min), _us(st.max), md.row_group(i).num_rows))
    return out


def to_seconds(col):
    a = col
    if pa.types.is_timestamp(a.type):
        a = a.cast(pa.timestamp("us")).cast(pa.int64())
    return np.asarray(a.to_numpy(zero_copy_only=False), dtype=np.float64) / 1e6


def read_window(path, frac=0.5, log=print):
    t_ = time.time()
    pf, mode = open_parquet(path)
    spans = rowgroup_spans(pf)
    cols = [c for c in COLS if c in pf.schema_arrow.names]
    if spans is None:                     # no statistics: read the whole shard's started_at first
        log(f"  {path}: no row-group statistics, reading started_at for all row groups")
        st = to_seconds(pf.read(columns=["started_at"]).column(0)) * 1e6
        lo, hi = float(np.nanmin(st)), float(np.nanmax(st)); rgs = list(range(pf.metadata.num_row_groups))
    else:
        lo = min(s[1] for s in spans); hi = max(s[2] for s in spans); rgs = None
    ws_us = lo + frac * (hi - lo); ws_us = math.floor(ws_us / 1e6) * 1e6
    a_us, b_us = ws_us - PAD_S * 1e6, ws_us + WIN_S * 1e6
    if rgs is None:
        rgs = [i for i, mn, mx, n in spans if mx >= a_us and mn < b_us]
    log(f"  {path} [{mode}]: shard span {(hi - lo) / 3.6e9:.1f} h, {pf.metadata.num_row_groups} row groups, reading {len(rgs)}")
    tab = pf.read_row_groups(rgs, columns=cols)
    s = to_seconds(tab.column("started_at")); c = to_seconds(tab.column("completed_at"))
    keep = (s >= a_us / 1e6) & (s < b_us / 1e6)
    tab = tab.filter(pa.array(keep)); s, c = s[keep], c[keep]
    ws = ws_us / 1e6; win = WIN_S
    n_in = int(((s >= ws) & (s < ws + win)).sum())
    if n_in > MAX_ROWS:                   # shorten the window so the file stays small
        ss = np.sort(s[s >= ws]); win = float(ss[MAX_ROWS - 1] - ws); win = max(600.0, math.floor(win / 60) * 60)
        keep = s < ws + win; tab = tab.filter(pa.array(keep)); s, c = s[keep], c[keep]
        log(f"  window shortened to {win:.0f} s ({n_in} requests in the full hour)")
    df = pd.DataFrame({"started": (s - ws).astype(np.float32), "completed": (c - ws).astype(np.float32)})
    uid = tab.column("user_id").to_pandas().astype(str)
    if "rehash_round" in tab.column_names:
        uid = uid + "_" + tab.column("rehash_round").to_pandas().astype(str)
    df["acct"] = pd.factorize(uid)[0].astype(np.int32)
    for name, out in (("instance_id", "inst"), ("chute_id", "chute")):
        if name in tab.column_names:
            df[out] = pd.factorize(tab.column(name).to_pandas().astype(str))[0].astype(np.int32)
    for name in ("ttft", "ot"):
        if name in tab.column_names:
            df[name] = tab.column(name).to_pandas().astype(np.float32)
    df = df.sort_values("started", kind="stable").reset_index(drop=True)
    meta = dict(shard=path, frac=frac, window_start_abs_s=ws, window_s=win, pad_s=PAD_S, rows_read=int(tab.num_rows),
                seconds=round(time.time() - t_, 1))
    return df, meta


def profile(df, win):
    d = df[df.completed.notna()].copy(); d["dur"] = d.completed - d.started
    bad = int((d.dur < 0).sum()); d = d[d.dur >= 0]
    inw = d[(d.started >= 0) & (d.started < win)]
    per = inw.groupby("acct").size()
    act = per[per >= 3]
    q = lambda x, ps=(10, 50, 90, 99): {f"p{p}": round(float(np.percentile(x, p)), 3) for p in ps} if len(x) else {}
    # max concurrency per active account (sweep line)
    conc = {}
    for a, g in d[d.acct.isin(act.index)].groupby("acct"):
        ev = np.r_[g.started.values, g.completed.values]; sg = np.r_[np.ones(len(g)), -np.ones(len(g))]
        o = np.lexsort((sg, ev)); conc[a] = int(np.cumsum(sg[o]).max())
    conc = pd.Series(conc, dtype=float)
    out = dict(rows=int(len(df)), rows_completed=int(len(d)), negative_durations=bad, missing_completed=int(df.completed.isna().sum()),
               requests_in_window=int(len(inw)), accounts=int(inw.acct.nunique()), active_accounts=int(len(act)),
               light_3_30=int(((act >= 3) & (act <= 30)).sum()), heavy_gt30=int((act > 30).sum()),
               req_per_active_account=q(act.values), duration_s=q(inw.dur.values), ttft_s=q(inw.ttft.dropna().values) if "ttft" in inw else None,
               out_tokens=q(inw.ot.dropna().values) if "ot" in inw else None,
               max_concurrency=dict(frac_gt1=round(float((conc > 1).mean()), 3), frac_gt8=round(float((conc > 8).mean()), 3),
                                    frac_gt64=round(float((conc > 64).mean()), 3), p99=float(np.percentile(conc, 99)) if len(conc) else None),
               share_top10_accounts=round(float(per.sort_values().iloc[-10:].sum() / per.sum()), 3) if len(per) else None,
               share_top1pct=round(float(per.sort_values().iloc[-max(1, len(per) // 100):].sum() / per.sum()), 3) if len(per) else None,
               rate_per_s=round(len(inw) / win, 2))
    if "inst" in d:
        out["instances"] = int(inw.inst.nunique()); out["chutes"] = int(inw.chute.nunique())
        top = inw[inw.acct.isin(act.index)].groupby("acct").inst.agg(lambda x: x.value_counts(normalize=True).iloc[0])
        one = top[top >= 0.9].index
        host = inw[inw.acct.isin(one)].groupby("acct").inst.agg(lambda x: x.value_counts().index[0]).value_counts()
        out["single_instance_accounts"] = int(len(one)); out["instances_hosting_ge40"] = int((host >= 40).sum())
        out["accounts_per_instance_p50"] = float(inw.groupby("inst").acct.nunique().median())
    return out
