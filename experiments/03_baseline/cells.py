# %% [markdown]
# # Notebook 6 — Dispatcher fingerprinting on a live vLLM testbed
# **Question.** Notebook 5 showed that pools of keys driven by one client leave a *turn-taking* signature (R < 1). This notebook asks the sharper question: **can a provider tell *how* a pool is being dispatched — and reconstruct it — from its own request log alone (key, arrival, end of stream)?**
#
# Four signals, all computed without labels (library `fplib.py`):
#
# | signal | what it measures | which dispatcher it exposes |
# |---|---|---|
# | **ring** | zero-lag high-pass occupancy correlation, keys ordered on a circle recovered by spectral seriation | one shared round-robin counter (`itertools.cycle` shared by workers) — also open-loop relays that round-robin, which R cannot see |
# | **conc** | how concentrated each key's short-lag *end → next start* successors are | per-worker round-robin counters (key *i* is always followed by key *i+1*) |
# | **coll** | same-key overlap relative to spreading the concurrent requests uniformly over the pool | least-busy / round-robin (avoid collisions) vs random / shuffle (≈ 1) |
# | **R(τ)** | occupancy variance ratio at τ = 5, 20, 120 s and the full window | closed loops (R < 1); sticky / episodic key holding (R ≈ 1 at short τ, < 1 at long τ) |
#
# plus a worker-count estimate **K = X · (E[D] + E[G])** (the interactive response-time law, N = X(R+Z)), where the client "think time" E[G] is read off the excess of *end → next start* lags.
#
# **Pre-registered classifier and thresholds** were fixed in simulation (10/10 dispatchers correct on three seeds) before this run: ring z > 6 → shared round-robin (closed loop if R is also significantly < 1, otherwise open-loop relay); else conc z > 6 → per-worker round-robin; else if R or R20 is > 3 sd below the human null → sticky if (1−R5) < ½(1−R), least-busy if coll < 0.5, otherwise random/shuffle; R > 3 sd above → bunching; otherwise none.
#
# **Fixes from Notebook 5.** Every population now runs in **its own process** (no shared event loop), the open-loop relays use an **absolute Poisson schedule**, and every request carries an `X-Req-Id` header so client-side and gateway-side times can be joined.
#
# | population | dispatcher | K workers | keys | expected class |
# |---|---|---|---|---|
# | `hum` | 400 humans (two processes) | — | 400 | none |
# | `bat` | 4 legitimate batch jobs (one key each) | — | 4 | (used in the heavy-benign control) |
# | `P1_shuffle` | LiteLLM Router `simple-shuffle` | 6 | 40 | random / shuffle |
# | `P2_least` | LiteLLM Router `least-busy` | 10 | 40 | **none** — the dry run showed LiteLLM breaks ties by list order, so the 10 workers settle on the first 10 keys and behave like 10 independent single-key clients (a known blind spot: pinned assignment gives R = 1) |
# | `P3_rrshared` | OpenAI SDK, one shared `itertools.cycle` | 12 | 40 | shared round-robin (closed loop) |
# | `P4_rrworker` | OpenAI SDK, one cycle per worker | 8 | 40 | per-worker round-robin |
# | `P5_rrjitter` | shared cycle + 0–6 s random delay | 8 | 40 | shared round-robin (closed loop) |
# | `P6_agents` | agents, one key per episode, tool calls | 6 | 32 | sticky / episodic |
# | `P7_relay_random` | open-loop relay, Poisson 2/s, random key | — | 40 | none (undetectable by design) |
# | `P8_relay_rr` | open-loop relay, Poisson 2/s, round-robin key | — | 40 | shared round-robin (open-loop relay) |
# | `C_company` | legitimate company, LiteLLM shuffle | 4 | 10 | random / shuffle (designed false positive) |
#
# **Run settings.** Accelerator **GPU T4 ×1**, Internet **On**. Runtime ≈ 75–90 min. Paste back the block between the PASTE markers.

# %% [code]
# Cell 1 — settings
import os, sys, json, time, subprocess, signal, glob
T_START = time.time()
NH = int(os.environ.get("NB6_HUMANS", 400))
CFG = dict(
    backend=os.environ.get("NB6_BACKEND", "vllm"),   # "vllm" = stop if vLLM cannot start; "auto" = fall back to the simulated server; "sim"
    model="Qwen/Qwen2.5-0.5B-Instruct",
    run_s=int(os.environ.get("NB6_RUN_S", 2400)), warmup_s=int(os.environ.get("NB6_WARMUP", 180)), seed=6,
    tok_med=96, tok_sig=0.6, client_gap_med=0.2,
    human=dict(gap_s=float(os.environ.get("NB6_HUMAN_GAP", 360)), mean_req=6, think_med_s=6.0, think_sig=0.9),
    agent=dict(mean_steps=8, tool_med_s=2.0, tok_med=64),
    pops={
        "hum_a": dict(kind="human", n=NH // 2, key_offset=0),
        "hum_b": dict(kind="human", n=NH - NH // 2, key_offset=NH // 2),
        "bat": dict(kind="batch", n=4),
        "P1_shuffle": dict(kind="litellm", strategy="simple-shuffle", K=6, M=40),
        "P2_least": dict(kind="litellm", strategy="least-busy", K=10, M=40),
        "P3_rrshared": dict(kind="rr_shared", K=12, M=40),
        "P4_rrworker": dict(kind="rr_worker", K=8, M=40),
        "P5_rrjitter": dict(kind="rr_shared", K=8, M=40, jitter=6.0),
        "P6_agents": dict(kind="agents", K=6, M=32),
        "P7_relay_random": dict(kind="open_loop", lam=2.0, policy="random", M=40),
        "P8_relay_rr": dict(kind="open_loop", lam=2.0, policy="rr", M=40),
        "C_company": dict(kind="litellm", strategy="simple-shuffle", K=4, M=10),
    },
    expected={
        "hum_random40": "none", "benign_36hum_4bat": "none",
        "P1_shuffle": "random / shuffle", "P2_least": "none",   # dry run: LiteLLM least-busy breaks ties by list order, so K workers settle on the first K keys (pinned)
        "P3_rrshared": "shared round-robin (closed loop)", "P4_rrworker": "per-worker round-robin",
        "P5_rrjitter": "shared round-robin (closed loop)", "P6_agents": "sticky / episodic",
        "P7_relay_random": "none", "P8_relay_rr": "shared round-robin (open-loop relay)", "C_company": "random / shuffle",
    },
    gateway="http://127.0.0.1:9000",
)
print(json.dumps({k: v for k, v in CFG.items() if k not in ("pops", "expected")}, indent=1))

# %% [code]
# Cell 2 — libraries
LIB_PLACEHOLDER = True

# %% [code]
FINGERPRINT_PLACEHOLDER = True

# %% [code]
# Cell 3 — testbed components
SIMSERVER_PLACEHOLDER = True

# %% [code]
GATEWAY_PLACEHOLDER = True

# %% [code]
DRIVER_PLACEHOLDER = True

# %% [code]
# Cell 4 — client libraries
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "aiohttp", "openai", "litellm"], check=False)
import importlib; importlib.invalidate_caches()
import aiohttp, openai
print("aiohttp", aiohttp.__version__, "openai", openai.__version__)


# %% [code]
# Cell 5 — model server: vLLM on the GPU if possible, otherwise the simulated server
import urllib.request
def wait_http(url, timeout_s, proc=None):
    t = time.time()
    while time.time() - t < timeout_s:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            urllib.request.urlopen(url, timeout=5); return True
        except Exception:
            time.sleep(5)
    return False

def tail(path, n=25):
    try:
        return "".join(open(path, errors="ignore").readlines()[-n:])
    except Exception:
        return ""

BACKEND = {"kind": None, "attempts": []}
def gpu_present():
    try:
        return subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0
    except Exception:
        return False

def vllm_import_check(py):
    r = subprocess.run([py, "-c", "import torch, vllm; print('vllm', vllm.__version__, '| torch', torch.__version__, '| cuda', torch.version.cuda, '| gpu', torch.cuda.is_available())"],
                       capture_output=True, text=True)
    return r.returncode == 0, (r.stdout + r.stderr).strip()[-1200:]

def launch_vllm(py, extra_env, extra_args):
    env = dict(os.environ, **extra_env)
    log = open("vllm.log", "w")
    p = subprocess.Popen([py, "-m", "vllm.entrypoints.openai.api_server", "--model", CFG["model"], "--dtype", "half",
                          "--max-model-len", "2048", "--gpu-memory-utilization", "0.85", "--max-num-seqs", "128", "--port", "8000"] + extra_args,
                         stdout=log, stderr=subprocess.STDOUT, env=env)
    ok = wait_http("http://127.0.0.1:8000/v1/models", 900, p)
    if not ok:
        p.kill(); time.sleep(5)
    return ok, p

if CFG["backend"] in ("auto", "vllm") and gpu_present():
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], capture_output=True, text=True).stdout)
    pythons = []
    # Route 1: Kaggle's own Python. vLLM upgrades torch; Kaggle's torchaudio then no longer matches, so remove it (not needed for text models).
    t_inst = time.time()
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"], capture_output=True, text=True)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchaudio"], capture_output=True, text=True)
    ok, msg = vllm_import_check(sys.executable)
    BACKEND["route1"] = dict(install_ok=r.returncode == 0, import_ok=ok, minutes=round((time.time() - t_inst) / 60, 1), msg=msg[-300:])
    print("route 1 (system python):", "OK" if ok else "FAILED", "\n", msg)
    if ok:
        pythons.append(sys.executable)
    else:
        # Route 2: an isolated virtual environment, so nothing preinstalled can clash with vLLM's own torch
        t_inst = time.time()
        venv = "/tmp/vllm_env"
        subprocess.run([sys.executable, "-m", "venv", venv], check=False)
        py = os.path.join(venv, "bin", "python")
        subprocess.run([py, "-m", "pip", "install", "-q", "--upgrade", "pip"], capture_output=True)
        r2 = subprocess.run([py, "-m", "pip", "install", "-q", "vllm"], capture_output=True, text=True)
        ok2, msg2 = vllm_import_check(py)
        BACKEND["route2"] = dict(install_ok=r2.returncode == 0, import_ok=ok2, minutes=round((time.time() - t_inst) / 60, 1), msg=msg2[-300:])
        print("route 2 (isolated venv):", "OK" if ok2 else "FAILED", "\n", msg2, "\n", r2.stderr[-800:])
        if ok2:
            pythons.append(py)
    for py in pythons:
        for extra_env, extra_args in (({}, []), ({}, ["--enforce-eager"]), ({"VLLM_ATTENTION_BACKEND": "TRITON_ATTN"}, ["--enforce-eager"])):
            ok, p = launch_vllm(py, extra_env, extra_args)
            BACKEND["attempts"].append(dict(python=py, env=extra_env, args=extra_args, ok=ok, log_tail=None if ok else tail("vllm.log", 15)[-1500:]))
            if ok:
                BACKEND.update(kind="vllm", proc=p, model=CFG["model"]); break
            print("vLLM launch failed:", extra_env, extra_args, "\n", tail("vllm.log", 15))
        if BACKEND["kind"]:
            break
if BACKEND["kind"] is None:
    if CFG["backend"] == "vllm":
        raise RuntimeError("vLLM requested but could not start; see BACKEND['attempts']")
    p = subprocess.Popen([sys.executable, "simserver.py", "--port", "8000"], stdout=open("sim.log", "w"), stderr=subprocess.STDOUT)
    assert wait_http("http://127.0.0.1:8000/health", 60, p), tail("sim.log")
    BACKEND.update(kind="sim", proc=p, model="sim")
print("backend:", BACKEND["kind"], [(a["env"], a["args"], a["ok"]) for a in BACKEND["attempts"]])


# %% [code]
# Cell 6 — gateway + smoke test through the real OpenAI SDK
for f in ("gateway_log.csv",):
    if os.path.exists(f): os.remove(f)
gw = subprocess.Popen([sys.executable, "gateway.py", "--port", "9000", "--backend", "http://127.0.0.1:8000", "--log", "gateway_log.csv"],
                      stdout=open("gateway.out", "w"), stderr=subprocess.STDOUT)
assert wait_http(CFG["gateway"] + "/v1/models", 60, gw), tail("gateway.out")
from openai import OpenAI
c = OpenAI(api_key="smoke-000", base_url=CFG["gateway"] + "/v1")
t = time.time()
s = c.chat.completions.create(model=BACKEND["model"], messages=[{"role": "user", "content": "Say hello."}], max_tokens=96, stream=True,
                              extra_body={"ignore_eos": True})
n = sum(1 for _ in s)
BACKEND["smoke"] = dict(chunks=n, seconds=round(time.time() - t, 2))
print("smoke test:", BACKEND["smoke"])


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
