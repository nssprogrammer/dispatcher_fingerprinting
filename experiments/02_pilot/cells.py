# %% [markdown]
# # Notebook 5 — Live serving testbed with ground truth
# **Question.** Do real client tools — a key-rotating OpenAI-SDK script, LiteLLM's Router, an agent fleet — leave the turn-taking signature when they hit a real LLM server, measured the way a provider would measure it (arrival → end of stream, at its own gateway)? And do honest users, batch jobs and an open-loop relay stay at R ≈ 1?
#
# **Setup.** An open-weights model served by vLLM on the T4 (or, if vLLM cannot start, a simulated server whose latency rises with load). A logging gateway in front of it records every request by API key. One driver process runs all populations at once, each on its own keys:
#
# | label | population | expected |
# |---|---|---|
# | `hum` | 400 simulated humans (sessions, think time) | R ≈ 1 |
# | `bat` | 4 legitimate single-key batch jobs | R ≈ 1 when pooled with humans |
# | `A1sem` | key-rotating script, 8 workers over 40 keys (OpenAI SDK) | R ≪ 1 |
# | `A2jit` | same with up to 6 s random delay per request | R < 1, weaker |
# | `A3llm` | LiteLLM Router, simple-shuffle over 40 keys, 8 workers | R ≪ 1 |
# | `A4agt` | 6 agents, one key per episode, tool calls between steps | R < 1 |
# | `A5rel` | open-loop relay, Poisson arrivals over 40 keys | R ≈ 1 (undetectable by design) |
# | `Ccomp` | legitimate company, one queue over 10 keys | R ≪ 1 (designed false positive) |
#
# Time is compressed: short generations (median 96 tokens) and short think times, so ~40 minutes give hundreds of request durations per key. The high-pass block is scaled to τ = 20 s.
#
# **Pre-registered go/no-go (proposal, month 3).** Stop the occupancy track if the default harnesses (A1, A3) do not give R_τ < 0.9.
#
# **Run settings.** Accelerator **GPU T4 ×1 (or ×2)**, Internet **On**. Runtime ≈ 60–75 min (≈ 10–15 of them installing vLLM).
# Paste back the block between the PASTE markers.

# %% [code]
# Cell 1 — settings
import os, sys, json, time, subprocess, signal
T_START = time.time()
CFG = dict(
    backend="vllm",                    # "vllm" = stop if vLLM cannot start; "auto" = fall back to the simulated server; "sim"
    model="Qwen/Qwen2.5-0.5B-Instruct",
    run_s=int(os.environ.get("NB5_RUN_S", 2400)), warmup_s=int(os.environ.get("NB5_WARMUP", 180)), tau=20.0, seed=5,
    tok_med=96, tok_sig=0.6,
    human=dict(gap_s=float(os.environ.get("NB5_HUMAN_GAP", 360)), mean_req=6, think_med_s=6.0, think_sig=0.9),
    agent=dict(mean_steps=8, tool_med_s=2.0, tok_med=64),
    workloads=dict(humans=int(os.environ.get("NB5_HUMANS", 400)), batch_clients=4, A1_K=8, A2_K=8, A2_jitter_s=6.0,
                   A3_K=8, A4_agents=6, A5_rate=2.0, C_K=4),
    gateway="http://127.0.0.1:9000",
)
print(json.dumps(CFG, indent=1))

# %% [code]
# Cell 2 — library
LIB_PLACEHOLDER = True

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
for f in ("gateway_log.csv", "labels.json"):
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
# Cell 7 — run every population at once
dcfg = dict(CFG, model=BACKEND["model"])
json.dump(dcfg, open("driver_cfg.json", "w"))
t_run = time.time()
drv = subprocess.Popen([sys.executable, "driver.py", "driver_cfg.json"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in drv.stdout:
    if line.startswith("[driver]"): print(line.rstrip(), flush=True)
drv.wait()
RUN = dict(driver_rc=drv.returncode, wall_min=round((time.time() - t_run) / 60, 1))
print(RUN)

# %% [code]
# Cell 8 — analysis with ground truth
import numpy as np, pandas as pd
from occlib import *
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
rng = np.random.default_rng(CFG["seed"])
log = pd.read_csv("gateway_log.csv")
labels = json.load(open("labels.json"))
log["label"] = log.key.map(labels)
ok = (log.status == 200) & log.label.notna()
RES = {"cfg": CFG, "backend": {k: v for k, v in BACKEND.items() if k != "proc"}, "run": RUN,
       "log": dict(n=len(log), ok_frac=round(float(ok.mean()), 4), by_label=log[ok].label.value_counts().to_dict())}
d = log[ok].copy()
t0 = d.t_arrival.min() + CFG["warmup_s"]; t1 = d.t_end.max() - 30
d = d[(d.t_end > t0) & (d.t_arrival < t1)]
d["dur"] = d.t_end - d.t_arrival
RES["durations_s"] = d.groupby("label").dur.median().round(2).to_dict()
keys_by = d.groupby("label").key.unique().to_dict()
kid = {k: i for i, k in enumerate(sorted(d.key.unique()))}
d["acct"] = d.key.map(kid)

def pool(keys_):
    m = d.key.isin(keys_)
    acc, _ = relabel(d.acct.values[m])
    return acc, d.t_arrival.values[m], d.t_end.values[m]

TAUS = (20.0, 60.0, 180.0)
def stats(keys_, n_sur=0):
    acc, s, e = pool(keys_)
    o, fk, b = occupancy(acc, s, e, t0, t1, None), fano_khat(acc, s, e, t0, t1), baselines(acc, s, e, t0, t1)
    out = dict(n_keys=len(keys_), req_h=round(b["req_per_h"]), peak=b["peak_N"], fano=round(float(fk["F"]), 3),
               R=round(float(o["R"]), 3), K_hat=round(float(fk["K_hat"]), 2) if np.isfinite(fk["K_hat"]) else None)
    for tau in TAUS:
        out[f"R{int(tau)}"] = round(float(occupancy(acc, s, e, t0, t1, tau)["R"]), 3) if (t1 - t0) >= 2 * tau else float("nan")
    if n_sur:
        sur = surrogate_null(acc, s, e, t0, t1, CFG["tau"], n_sur, rng)
        r = out[f"R{int(CFG['tau'])}"]
        out["p_low"] = round(float((1 + (sur <= r).sum()) / (n_sur + 1)), 4)     # turn-taking (R below independence)
        out["p_high"] = round(float((1 + (sur >= r).sum()) / (n_sur + 1)), 4)    # bunching (R above independence)
    return out

hum = list(keys_by.get("hum", [])); bat = list(keys_by.get("bat", []))
null = [stats(list(rng.choice(hum, min(40, len(hum)), replace=False))) for _ in range(200)]
NULL = {k: np.array([x[k] for x in null]) for k in ["R"] + [f"R{int(t)}" for t in TAUS]}
RES["null_humans_40"] = {k: dict(mean=round(float(v.mean()), 4), sd=round(float(v.std()), 4), q01=round(float(np.quantile(v, 0.01)), 4),
                                 q99=round(float(np.quantile(v, 0.99)), 4)) for k, v in NULL.items()}
def add_z(st):
    for k, v in NULL.items():
        st[f"z_{k}"] = round(float((v.mean() - st[k]) / v.std()), 1)      # positive = turn-taking, negative = bunching
    return st
pools = {}
for lab in ["A1sem", "A2jit", "A3llm", "A4agt", "A5rel", "Ccomp"]:
    if lab in keys_by:
        pools[lab] = add_z(stats(list(keys_by[lab]), n_sur=300))
heavy = [stats(list(rng.choice(hum, min(36, len(hum)), replace=False)) + bat) for _ in range(30)]
pools["benign_36hum_4batch"] = {k: round(float(np.mean([h[k] for h in heavy])), 3) for k in ["R", "R20", "R60", "fano", "req_h"]}
pools["benign_36hum_4batch"]["R20_min"] = round(float(np.min([h["R20"] for h in heavy])), 3)
for lab in ["A1sem", "A3llm", "A4agt"]:
    if lab in keys_by:
        camp = list(keys_by[lab])
        mixes = [add_z(stats(list(rng.choice(camp, min(20, len(camp)), replace=False)) + list(rng.choice(hum, min(20, len(hum)), replace=False)))) for _ in range(10)]
        pools[f"mix_random20{lab}_20hum"] = {k: round(float(np.median([m[k] for m in mixes])), 3)
                                             for k in ["R", "R20", "R60", "R180", "z_R", "z_R20", "z_R60", "z_R180"]}
# round-robin 'travelling wave': contiguous vs random halves of the rotated key ring
if "A1sem" in keys_by:
    ring = sorted(keys_by["A1sem"])
    pools["A1sem_ring_contiguous_half"] = add_z(stats(ring[:20]))
    pools["A1sem_ring_random_half"] = add_z(stats(list(rng.choice(ring, min(20, len(ring)), replace=False))))
RES["pools"] = pools
# theory check: R predicted from the pool's own Fano factor, p = 1 - F
th = {}
for lab in ["A1sem", "A2jit", "A3llm"]:
    if lab in pools:
        p_ = 1 - pools[lab]["fano"]; M = pools[lab]["n_keys"]
        th[lab] = dict(R_observed=pools[lab]["R"], R_predicted=round((1 - p_) / (1 - p_ / M), 3), K_hat=pools[lab]["K_hat"])
RES["theory_check"] = th
# pool growth: seed = half the campaign + 20 humans; rank the rest of the campaign among the other humans
growth = {}
for lab in ["A1sem", "A2jit", "A3llm", "A4agt"]:
    if lab not in keys_by: continue
    camp = list(keys_by[lab]); hh = list(rng.permutation(hum))
    seed_keys = camp[:20] + hh[:20]; cand_keys = camp[20:] + hh[20:]
    ms = d.key.isin(seed_keys).values; mc = d.key.isin(cand_keys).values
    cidx = {k: i for i, k in enumerate(cand_keys)}
    sc = corr_with_pool((d.acct.values[ms], d.t_arrival.values[ms], d.t_end.values[ms]),
                        (d.key[mc].map(cidx).values, d.t_arrival.values[mc], d.t_end.values[mc]), t0, t1, tau=CFG["tau"], n_cand=len(cand_keys))
    y = np.r_[np.ones(len(camp) - 20), np.zeros(len(hh) - 20)]
    growth[lab] = round(float(roc_auc_score(y, np.nan_to_num(-sc, nan=0.0))), 3)
RES["pool_growth_auc"] = growth
# serving-side effects: does latency rise with load (which would couple honest users)?
allv = d.sort_values("t_arrival")
tt = np.concatenate([allv.t_arrival.values, allv.t_end.values]); dd = np.concatenate([np.ones(len(allv)), -np.ones(len(allv))])
oo = np.lexsort((dd, tt)); lvl = np.cumsum(dd[oo])
level_at = pd.Series(lvl, index=tt[oo]).groupby(level=0).last()
n_at = level_at.reindex(allv.t_arrival.values, method="ffill").values
per_byte = allv.dur.values / np.maximum(allv.nbytes.values, 1)
RES["serving"] = dict(site_inflight_median=float(np.median(n_at)), site_inflight_p95=float(np.quantile(n_at, 0.95)),
                      spearman_latency_per_byte_vs_inflight=round(float(spearmanr(n_at, per_byte)[0]), 3))
RES["go_no_go"] = "GO" if all(pools.get(l, {}).get(f"R{int(CFG['tau'])}", 1) < 0.9 for l in ["A1sem", "A3llm"]) else "STOP"
print(json.dumps(RES, indent=1, default=str))

# %% [code]
# Cell 9 — shut down and print the summary
for proc in (gw, BACKEND.get("proc")):
    try: proc.terminate()
    except Exception: pass
d.to_csv("nb5_requests.csv", index=False)
RES["runtime_min"] = round((time.time() - T_START) / 60, 1)
json.dump(RES, open("nb5_results.json", "w"), indent=1, default=str)
print("===== PASTE FROM HERE =====")
print(json.dumps(RES, separators=(",", ":"), default=str))
print("===== PASTE TO HERE =====")
