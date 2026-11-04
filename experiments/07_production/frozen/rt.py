"""Real-traffic analysis (NB10): Chutes windows + injected campaign pools, using the frozen fp-0.8 detectors."""
import sys, heapq, zlib, json, time
import numpy as np, pandas as pd
SCR = "/tmp/claude-0/-home-claude/78f29ddf-32f5-54b2-bcc0-86fa7c435eb9/scratchpad"
sys.path.insert(0, SCR + "/fp"); sys.path.insert(0, SCR + "/nb1")
import fplib, occlib
from occlib import occupancy, fano_khat

RKEYS = ("R", "R5", "R20", "R120")
EXPECTED = {"I_rrshared": "shared round-robin (closed loop)", "I_shuffle": "random / shuffle", "I_rrworker": "per-worker round-robin",
            "I_rot429_shared": "ANY", "I_rot429_worker": "ANY", "I_hashed": "random / shuffle", "I_sticky_ttl": "sticky / episodic",
            "I_hedged": "ANY", "I_pinned": "none", "I_open_steady": "none", "I_company": "random / shuffle"}
INJECT = {"I_rrshared": dict(kind="rr_shared", K=8, M=40), "I_shuffle": dict(kind="random", K=8, M=40),
          "I_rrworker": dict(kind="rr_worker", K=8, M=40), "I_rot429_shared": dict(kind="rot429", K=8, M=40, shared=True),
          "I_rot429_worker": dict(kind="rot429", K=8, M=40, shared=False), "I_hashed": dict(kind="hashed", K=8, M=40),
          "I_sticky_ttl": dict(kind="sticky_ttl", K=8, M=40, ttl=60.0), "I_hedged": dict(kind="hedged", K=6, M=40),
          "I_pinned": dict(kind="pinned", K=8, M=8), "I_open_steady": dict(kind="open_loop", lam=2.0, M=40),
          "I_company": dict(kind="random", K=4, M=10)}


# ------------------------------------------------------------------ real window
def load_window(fn, W=3600.0, floor=3, light_max=30, heavy_max=None):
    df = pd.read_parquet(fn)
    df = df[df.completed.notna() & (df.completed >= df.started)]
    d = df[(df.started < W) & (df.completed > 0)].copy()
    per = d[(d.started >= 0)].groupby("acct").size()
    act = per[per >= floor]
    strat = pd.Series(np.where(act <= light_max, "light", "heavy"), index=act.index)
    if heavy_max is not None: strat[act > heavy_max] = "platform"
    d = d[d.acct.isin(act.index)]
    out = pd.DataFrame({"key": "r" + d.acct.astype(str), "grp": "real_" + d.acct.map(strat).values, "pos": -1,
                        "t_arrival": d.started.astype(np.float64).values, "t_end": d.completed.astype(np.float64).values, "status": 200})
    if "inst" in d: out["inst"] = d.inst.values
    durs = (df.completed - df.started).astype(np.float64).values
    durs = durs[(durs > 0) & np.isfinite(durs)]
    return out.reset_index(drop=True), durs, act, strat


# ------------------------------------------------------------------ injected closed-loop pools (discrete-event simulation)
def simulate_pool(name, P, durs, W, rng, t_start=-300.0, quota=15, gap_med=0.2, gap_sig=0.5, jitter=None):
    jitter = P.get("jitter", 0.0) if jitter is None else jitter
    M = P["M"]; keys = [f"{name}-{i:03d}" for i in range(M)]
    rows = []                                   # (key_index, t_arrival, t_end, status)
    win = [[] for _ in range(M)]                # accepted start times per key (sliding 60 s quota)
    def admit(k, t):
        q = win[k]
        while q and q[0] <= t - 60.0: q.pop(0)
        if len(q) >= quota: return False
        q.append(t); return True
    D = lambda: float(rng.choice(durs))
    G = lambda: float(rng.lognormal(np.log(gap_med), gap_sig)) + (rng.uniform(0, jitter) if jitter else 0.0)
    kind = P["kind"]
    if kind == "open_loop":
        t = t_start
        while t < W:
            t += rng.exponential(1.0 / P["lam"]); k = int(rng.integers(M))
            if admit(k, t): rows.append((k, t, t + D(), 200))
            else: rows.append((k, t, t + 0.01, 429))
        return _rows_df(rows, keys, name)
    if kind == "open_cap":                      # Poisson arrivals, but at most K requests in flight (a start waits for a free slot)
        K = P["K"]; t = t_start; busy = []
        while t < W:
            t += rng.exponential(1.0 / P["lam"]); start = t
            while busy and busy[0] <= start: heapq.heappop(busy)
            if len(busy) >= K: start = heapq.heappop(busy)
            k = int(rng.integers(M))
            if admit(k, start):
                e = start + D(); rows.append((k, start, e, 200)); heapq.heappush(busy, e)
            else: rows.append((k, start, start + 0.01, 429))
        return _rows_df(rows, keys, name)
    K = P["K"]; ev = [(t_start + rng.uniform(0, 2.0), w) for w in range(K)]; heapq.heapify(ev)
    shared = {"c": int(rng.integers(M)), "ptr": int(rng.integers(M))}
    own = {w: int(rng.integers(M)) for w in range(K)}
    sticky = {w: (int(rng.integers(M)), -1e9) for w in range(K)}
    if kind == "hold": sticky = {w: (0, -1) for w in range(K)}
    if kind == "pinned_gaps": G = lambda: float(rng.choice(P["gaps"]))
    task = [0]
    while ev:
        t, w = heapq.heappop(ev)
        if t >= W: continue
        if kind == "hedged":
            k1, k2 = rng.choice(M, 2, replace=False); d1, d2 = D(), D()
            ok1, ok2 = admit(k1, t), admit(k2, t)
            ends = [t + d1 if ok1 else None, t + d2 if ok2 else None]
            if not ok1 and not ok2:
                rows += [(k1, t, t + 0.01, 429), (k2, t, t + 0.01, 429)]; heapq.heappush(ev, (t + 1.0, w)); continue
            win_end = min(x for x in ends if x is not None)
            for k, ok, e in ((k1, ok1, ends[0]), (k2, ok2, ends[1])):
                if not ok: rows.append((k, t, t + 0.01, 429))
                else: rows.append((k, t, win_end, 200 if e == win_end else 499))
            heapq.heappush(ev, (win_end + G(), w)); continue
        if kind == "rr_shared":
            k = shared["c"] % M; shared["c"] += 1
        elif kind == "random":
            k = int(rng.integers(M))
        elif kind == "rr_worker":
            k = own[w] % M; own[w] += 1
        elif kind == "hashed":
            task[0] += 1; k = zlib.crc32(f"{name}-{task[0]}".encode()) % M
        elif kind == "pinned":
            k = w
        elif kind == "rot429":
            k = (shared["ptr"] if P["shared"] else own[w]) % M
        elif kind == "sticky_ttl":
            k, tp = sticky[w]
            if t - tp > P["ttl"]: k = int(rng.integers(M)); sticky[w] = (k, t)
        elif kind == "hold":
            k, used = sticky[w]
            if used < 0 or used >= P["H"]: k = shared["c"] % M; shared["c"] += 1; used = 0
            sticky[w] = (k, used + 1)
        elif kind == "pinned_gaps":
            k = w % M
        if admit(k, t):
            e = t + D(); rows.append((k, t, e, 200)); heapq.heappush(ev, (e + G(), w))
        else:
            rows.append((k, t, t + 0.01, 429))
            if kind == "rot429":
                if P["shared"]:
                    if shared["ptr"] % M == k: shared["ptr"] += 1
                else: own[w] += 1
                heapq.heappush(ev, (t + 0.005, w))
            elif kind == "sticky_ttl":
                sticky[w] = (int(rng.integers(M)), t + 0.005); heapq.heappush(ev, (t + 0.005, w))
            else:
                heapq.heappush(ev, (t + 1.0, w))
    return _rows_df(rows, keys, name)


def _rows_df(rows, keys, name):
    r = np.array(rows, dtype=float) if rows else np.zeros((0, 4))
    return pd.DataFrame({"key": [keys[int(k)] for k in r[:, 0]], "grp": name, "pos": r[:, 0].astype(int),
                         "t_arrival": r[:, 1], "t_end": r[:, 2], "status": r[:, 3].astype(int)})


def inject(real, durs, W, rng, pools=INJECT, **kw):
    sims = [simulate_pool(n, P, durs, W, rng, **kw) for n, P in pools.items()]
    allr = pd.concat([real] + sims, ignore_index=True)
    return allr


# ------------------------------------------------------------------ arrays and nulls
class Site:
    def __init__(self, allr, W):
        self.W = W; self.r429 = allr[allr.status == 429]
        d = allr[(allr.status == 200) & (allr.t_end > 0) & (allr.t_arrival < W)].copy()
        self.d = d; self.keys = sorted(d.key.unique()); self.kid = {k: i for i, k in enumerate(self.keys)}
        self.A = d.key.map(self.kid).values.astype(np.int64); self.S = d.t_arrival.values; self.E = d.t_end.values
        g = d.groupby("key").grp.first(); self.grp = g.reindex(self.keys).values
        self.pos = d.groupby("key").pos.first().reindex(self.keys).values
        self.by = {k: v for k, v in d.groupby("grp").key.unique().items()}

    def pool(self, ks):
        ks = sorted(ks); idx = {k: i for i, k in enumerate(ks)}; m = self.d.key.isin(idx).values
        return self.d.key.values[m], np.array([idx[k] for k in self.d.key.values[m]], dtype=np.int64), self.S[m], self.E[m], ks

    def null(self, grp, rng, n=200, size=40):
        ks_all = list(self.by[grp]); vals = []
        for _ in range(n):
            _, a, s, e, ks = self.pool(rng.choice(ks_all, size, replace=False))
            st = {"R": float(occupancy(a, s, e, 0, self.W, None, n_accounts=len(ks))["R"])}
            for tau in (5.0, 20.0, 120.0):
                st[f"R{int(tau)}"] = float(occupancy(a, s, e, 0, self.W, tau, n_accounts=len(ks))["R"])
            vals.append(st)
        NULL = {k: np.array([x[k] for x in vals]) for k in RKEYS}
        sn = []
        for _ in range(max(20, n // 2)):
            _, a, s, e, ks = self.pool(rng.choice(ks_all, size, replace=False))
            sn.append(fplib.start_count_ratio(a, s, len(ks), 0, self.W))
        sn = np.array([x for x in sn if np.isfinite(x)])
        return NULL, sn


def fingerprint_pool(site, ks, NULL, sn, rng, true_pos=None):
    _, a, s, e, ks = site.pool(ks); M = len(ks); W = site.W
    f, order = fplib.fingerprint(a, s, e, M, 0, W, rng, n_sur=40, occupancy_fn=occupancy)
    z = {k: float((NULL[k].mean() - f[k]) / NULL[k].std()) for k in RKEYS}
    rz6, sc6, order6 = fplib.ring_multiscale(a, s, e, M, 0, W, rng, n_sur=40)
    f6 = dict(f, ring_z=max(f["ring_z"], rz6))
    z_start = float((sn.mean() - fplib.start_count_ratio(a, s, M, 0, W)) / sn.std())
    cls7 = fplib.classify_v07(f6, z, z_start)
    if rz6 > f["ring_z"] and order6 is not None: order = order6
    fk = fano_khat(a, s, e, 0, W)
    out = dict(cls=cls7, R=round(f["R"], 3), z_R=round(z["R"], 1), z_R20=round(z["R20"], 1), ring_z=round(float(f6["ring_z"]), 1),
               conc_z=round(f["conc_z"], 1) if np.isfinite(f["conc_z"]) else None, coll=round(f["coll"], 2), z_start=round(z_start, 1),
               K_little=round(f["K_little"], 1) if np.isfinite(f["K_little"]) else None,
               K_fano=round(float(fk["K_hat"]), 1) if np.isfinite(fk["K_hat"]) else None, n_req=int(f["n_req"]))
    if true_pos is not None:
        out["ring_acc"] = round(fplib.neighbour_accuracy(order, np.argsort(np.argsort(true_pos))), 2)
    return out


def ok_class(exp, c):
    return (c != "none") if exp == "ANY" else (c in exp.split(" | "))


# ------------------------------------------------------------------ Proposition 1 strength of an injected ring
def theta1(site, ring_grp, rng, M=40):
    m = site.d.grp.values == ring_grp; a = site.d.pos.values[m].astype(int); s = site.d.t_arrival.values[m]
    e = site.d.t_end.values[m]; dur = e - s; lam = len(s) / site.W; Tc = M / lam; p = dur.mean() / Tc
    o = np.argsort(s); ss, aa = s[o], a[o]; c = []
    ov = lambda tau, D1, D2: np.clip(np.minimum(D1, tau + D2) - np.maximum(0, tau), 0, None)
    for k in range(1, M // 2 + 1):
        lags = []
        for i in range(0, len(ss), 5):
            j = np.searchsorted(ss, ss[i], "right"); nxt = np.nonzero(aa[j:j + 400] == (aa[i] + k) % M)[0]
            if len(nxt): lags.append(ss[j + nxt[0]] - ss[i])
        lags = np.array(lags); lags = np.where(lags > Tc / 2, lags - Tc, lags)
        D1 = rng.choice(dur, len(lags)); D2 = rng.choice(dur, len(lags))
        c.append((ov(lags, D1, D2).mean() / Tc - p * p) / (p - p * p))
    c = np.array(c); full = np.r_[c, c[:M - 1 - len(c)][::-1]] if M % 2 else np.r_[c, c[:-1][::-1]]
    d = np.arange(1, M)
    return float(np.sum(full[:M - 1] * np.cos(2 * np.pi * d / M))), float(p)


def recovered(lab, grp, g):
    m = grp == g; best, pur = 0, 0.0
    for c in np.unique(lab[lab >= 0]):
        n = int(((lab == c) & m).sum())
        if n > best: best, pur = n, n / int((lab == c).sum())
    return best / max(m.sum(), 1), pur
