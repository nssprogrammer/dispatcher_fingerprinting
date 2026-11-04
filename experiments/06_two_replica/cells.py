# %% [markdown]
# # Notebook 9 — Replicas, quotas and unseen dispatchers
# **The reviewers' most serious objections, tested.**
#
# - **Phase A (40 min):** two vLLM replicas behind a provider-side least-outstanding balancer.
# - **Phase B (40 min):** the same replicas with key-affinity routing, so each key always lands on one replica and shares its batching queue.
#
# In both phases the gateway enforces a per-key quota of 15 requests per minute and returns HTTP 429 beyond it.
#
# Populations:
# - humans (WildChat think times) and batch jobs;
# - five dispatchers never seen when thresholds were fixed: rotate on 429 with a shared pointer, rotate on 429 per worker, consistent hashing, sticky key with a 60 s time limit, and hedged requests (each request on 2 keys, first answer kept);
# - references from earlier runs: shared counter, shuffle, pinned, steady open loop, and the legitimate company.
#
# Detectors are frozen: v0.6 classifier and spectral, **v0.7** (bulk-edge trimming; bunching must be confirmed on start counts), the v0.4 sideband graph, and **v0.8** (spectral trimming by persistence time; hand-off graph with a coverage null corrected over all pairs).
# v0.8 was fixed after the NB9 *simulated* dry run, before this live run: v0.7 trimmed on integrated squared autocorrelation, which counts every periodic peak of a ring key, and so trimmed half the shared-counter ring (20/40 keys). v0.8 trims on autocorrelation integrated to its first zero crossing (long positive memory = held keys); on NB8-B it keeps 5/5 rings at 20/20. v0.7 is still reported unchanged.
#
# **Locked predictions:**
# - **M1** replica coupling does not create turn-taking in humans: human-pool R z-scores beyond ±3 in ≤ 5% of pools, human-pool ring z < 6, ≤ 2 humans selected by spectral v0.7/v0.8 and by the v0.8 hand-off graph. Both phases.
# - **M2** with affinity routing, human pools drawn from one replica stay within the human null (mean |z| < 2).
# - **M3** references: shared counter → shared rotation, shuffle → shuffle, company → shuffle, pinned → none, steady open loop → none (v0.7). Both phases.
# - **M4** unseen dispatchers are all flagged (class ≠ none, v0.7); consistent hashing → shuffle; sticky TTL → sticky. Both phases.
# - **M5** rotate-on-429 pools show failover hand-offs after 429s (z > 6); pinned keys and humans do not.
# - **M6** spectral v0.8 recovers the shared-counter ring (recall ≥ 0.9, purity ≥ 0.9, ≤ 2 humans) in both phases. (v0.7 is predicted to fail it: recall < 0.9.)
# - **M7** the quota caps pinned workers at ≤ 0.25 requests/s each (the cost of pinning under a quota).
# - **M9** the frozen v0.4 hand-off graph may select humans, and every human it selects is linked only to rotate-on-429 pools (F1/F2). Found on the 40-minute dry run (10 humans): keys held until refused end their requests in bursts, which the v0.4 sideband null counts as many hand-offs, and v0.4 corrects only over screened edges. The v0.8 graph uses a coverage null and corrects over all key pairs; on NB7, NB8 A, NB8 B and the dry run it selected 0 humans.
# - **M8** forensic expansion (frozen from the NB7 post-hoc analysis): from a random 20% seed, with the linking score chosen without labels, the rest of the pool ranks among all keys at AUC ≥ 0.85 within 20 minutes for shuffle (F7), company (C), consistent hashing (F3) and hedged requests (F5). Both phases. Other pools are reported without a prediction.
#
# Accelerator **GPU T4 ×1**, Internet **On**, about 2 h. Paste the PASTE block; upload `nb9_requests_A.csv` and `nb9_requests_B.csv`.

# %% [code]
# Cell 1 — settings
import os, sys, json, time, subprocess, signal, glob, threading, urllib.request
T_START = time.time()
NH = int(os.environ.get("NB9_HUMANS", 400))
POPS = {"hum_a": dict(kind="human", n=NH // 2, key_offset=0), "hum_b": dict(kind="human", n=NH - NH // 2, key_offset=NH // 2),
        "bat": dict(kind="batch", n=4),
        "F1_rot429_shared": dict(kind="rot429", K=8, M=40, shared=True), "F2_rot429_worker": dict(kind="rot429", K=8, M=40, shared=False),
        "F3_hashed": dict(kind="hashed", K=8, M=40), "F4_sticky_ttl": dict(kind="sticky_ttl", K=8, M=40, ttl=60.0),
        "F5_hedged": dict(kind="hedged", K=6, M=40),
        "F6_rrshared": dict(kind="rr_shared", K=8, M=40), "F7_shuffle": dict(kind="random", K=8, M=40),
        "F8_pinned": dict(kind="pinned", K=8, M=8), "F9_open_steady": dict(kind="open_loop", lam=2.0, policy="random", M=40),
        "C_company": dict(kind="litellm", strategy="simple-shuffle", K=4, M=10)}
EXP = {"hum_random40": "none", "benign_36hum_4bat": "none", "F1_rot429_shared": "ANY", "F2_rot429_worker": "ANY",
       "F3_hashed": "random / shuffle", "F4_sticky_ttl": "sticky / episodic", "F5_hedged": "ANY",
       "F6_rrshared": "shared round-robin (closed loop)", "F7_shuffle": "random / shuffle", "F8_pinned": "none",
       "F9_open_steady": "none", "C_company": "random / shuffle"}
CFG = dict(backend=os.environ.get("NB9_BACKEND", "vllm"), model="Qwen/Qwen2.5-0.5B-Instruct", seed=9,
           run_s={"A": int(os.environ.get("NB9_RUN", 2400)), "B": int(os.environ.get("NB9_RUN", 2400))},
           warmup_s=int(os.environ.get("NB9_WARMUP", 180)), tok_med=96, tok_sig=0.6, client_gap_med=0.2, max_out=1024, max_prompt=1500,
           human=dict(gap_s=float(os.environ.get("NB9_HUMAN_GAP", 360)), mean_req=6, think_med_s=6.0, think_sig=0.9),
           agent=dict(mean_steps=8, tool_med_s=2.0, tok_med=64), lengths={"A": "fixed", "B": "fixed"}, humans="wildchat",
           lb={"A": "least", "B": "affinity"}, rpm=15, replicas=2, pops={"A": POPS, "B": POPS}, expected={"A": EXP, "B": EXP},
           wm_mode={"A": "none", "B": "none"}, gateway="http://127.0.0.1:9000")
print(json.dumps({k: v for k, v in CFG.items() if k not in ("pops", "expected")}, indent=1))

# %% [code]
# Cell 2 — libraries and testbed components
LIB_PLACEHOLDER = True

# %% [code]
FINGERPRINT_PLACEHOLDER = True

# %% [code]
ANALYSIS_PLACEHOLDER = True

# %% [code]
SIMSERVER_PLACEHOLDER = True

# %% [code]
GATEWAY9_PLACEHOLDER = True

# %% [code]
DRIVER9_PLACEHOLDER = True

# %% [code]
# Cell 3 — client libraries
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "aiohttp", "openai", "litellm", "huggingface_hub", "pyarrow"], check=False)
import importlib; importlib.invalidate_caches()
import numpy as np, pandas as pd

# %% [code]
# Cell 4 — empirical human behaviour and request lengths from WildChat-1M (one shard)
WC_INFO = {}
try:
    import pyarrow.parquet as pq, pyarrow.compute as pc, pyarrow as pa
    from huggingface_hub import HfApi, hf_hub_download
    repo = "allenai/WildChat-1M"
    name = sorted(f.path for f in HfApi().list_repo_tree(repo, repo_type="dataset", recursive=True) if getattr(f, "path", "").endswith(".parquet"))[0]
    t = pq.read_table(hf_hub_download(repo, name, repo_type="dataset"), columns=["conversation"])
    conv = t.column("conversation").combine_chunks(); flat = pc.list_flatten(conv)
    parent = pc.list_parent_indices(conv).to_numpy()
    role = flat.field("role").to_numpy(zero_copy_only=False)
    clen = pc.utf8_length(flat.field("content")).to_numpy(zero_copy_only=False).astype(float)
    ts = pc.cast(flat.field("timestamp"), pa.int64()).to_numpy(zero_copy_only=False).astype(float) / 1e6
    tu = pd.DataFrame(dict(conv=parent, role=role, clen=clen, ts=ts))
    us, asst = tu[tu.role == "user"].reset_index(drop=True), tu[tu.role == "assistant"].reset_index(drop=True)
    n = min(len(us), len(asst))
    prompt_tok = np.clip(us.clen.values[:n] / 4, 4, 4000); out_tok = np.clip(asst.clen.values[:n] / 4, 1, 4000)
    same_conv = us.conv.values[:n] == asst.conv.values[:n]
    a = asst.sort_values(["conv", "ts"]); gaps = np.diff(a.ts.values); nxt_out = a.clen.values[1:] / 4
    same = np.diff(a.conv.values) == 0
    think = gaps[same] - nxt_out[same] / 40.0                 # time between answers minus the next answer's generation (~40 tok/s)
    think = think[np.isfinite(think) & (think > 1) & (think < 900)]
    sess = us.groupby("conv").size().values
    rs = np.random.default_rng(0)
    pick = lambda x, k=40000: rs.choice(x, min(k, len(x)), replace=False)
    ii = pick(np.nonzero(same_conv)[0])
    wc = dict(prompt_tok=prompt_tok[ii].round().tolist(), out_tok=out_tok[ii].round().tolist(), think_s=pick(think).round(2).tolist(),
              session_turns=pick(sess).tolist())
    WC_INFO = dict(source=name, prompt_tok_median=float(np.median(prompt_tok)), out_tok_median=float(np.median(out_tok)),
                   out_tok_p90=float(np.quantile(out_tok, .9)), think_median_s=float(np.median(think)), think_p90_s=float(np.quantile(think, .9)),
                   session_turns_median=float(np.median(sess)), session_turns_mean=float(np.mean(sess)))
except Exception as ex:
    print("WildChat unavailable, falling back to parametric humans and lengths:", repr(ex)[:300])
    rs = np.random.default_rng(0)
    wc = dict(prompt_tok=np.clip(rs.lognormal(np.log(60), 1.0, 40000), 4, 1500).round().tolist(),
              out_tok=np.clip(rs.lognormal(np.log(250), 0.8, 40000), 16, 1024).round().tolist(),
              think_s=np.clip(rs.lognormal(np.log(25), 1.0, 40000), 1, 900).round(2).tolist(), session_turns=rs.geometric(1 / 3, 40000).tolist())
    WC_INFO = dict(source="fallback-parametric")
json.dump(wc, open("wc.json", "w"))
print(json.dumps(WC_INFO, indent=1))

# %% [code]
# Cell 5 — install vLLM (same routes as NB5–NB7) and define start/stop of the model server
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

def tail(path, n=20):
    try: return "".join(open(path, errors="ignore").readlines()[-n:])
    except Exception: return ""

def gpu_present():
    try: return subprocess.run(["nvidia-smi"], capture_output=True).returncode == 0
    except Exception: return False

BACKEND = {"kind": None, "python": None, "install": {}}
if CFG["backend"] in ("auto", "vllm") and gpu_present():
    t_inst = time.time()
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"], capture_output=True, text=True)
    subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchaudio"], capture_output=True, text=True)
    chk = subprocess.run([sys.executable, "-c", "import torch, vllm; print('vllm', vllm.__version__, '| torch', torch.__version__)"], capture_output=True, text=True)
    if chk.returncode == 0:
        BACKEND.update(python=sys.executable)
    else:
        venv = "/tmp/vllm_env"; subprocess.run([sys.executable, "-m", "venv", venv]); py = os.path.join(venv, "bin", "python")
        subprocess.run([py, "-m", "pip", "install", "-q", "--upgrade", "pip"], capture_output=True)
        subprocess.run([py, "-m", "pip", "install", "-q", "vllm"], capture_output=True, text=True)
        chk = subprocess.run([py, "-c", "import torch, vllm; print('vllm', vllm.__version__, '| torch', torch.__version__)"], capture_output=True, text=True)
        if chk.returncode == 0: BACKEND.update(python=py)
    BACKEND["install"] = dict(minutes=round((time.time() - t_inst) / 60, 1), msg=(chk.stdout + chk.stderr).strip()[-300:])
    print(BACKEND["install"])
if BACKEND["python"] is None and CFG["backend"] == "vllm":
    raise RuntimeError("vLLM requested but not importable")

def start_replicas():
    """R model servers on ports 8000.. sharing the GPU; returns list of processes and backend kind"""
    procs = []
    for r in range(CFG["replicas"]):
        port = 8000 + r
        if BACKEND["python"] is None:
            p = subprocess.Popen([sys.executable, "simserver.py", "--port", str(port), "--capacity", "64"], stdout=open(f"sim_{r}.log", "w"), stderr=subprocess.STDOUT)
            assert wait_http(f"http://127.0.0.1:{port}/health", 60, p); procs.append(p); continue
        ok = False
        for extra in ([], ["--enforce-eager"]):
            p = subprocess.Popen([BACKEND["python"], "-m", "vllm.entrypoints.openai.api_server", "--model", CFG["model"], "--dtype", "half",
                                  "--max-model-len", "2048", "--max-num-seqs", "64", "--gpu-memory-utilization", "0.40", "--port", str(port)] + extra,
                                 stdout=open(f"vllm_{r}.log", "w"), stderr=subprocess.STDOUT)
            if wait_http(f"http://127.0.0.1:{port}/v1/models", 900, p): ok = True; break
            print("replica", r, "failed", extra, tail(f"vllm_{r}.log", 10)); p.kill(); time.sleep(5)
        if not ok: raise RuntimeError(f"replica {r} could not start")
        procs.append(p)
    return procs, ("vllm" if BACKEND["python"] else "sim")

def poll_metrics(stop, out):
    """vLLM Prometheus metrics every 10 s: KV-cache usage, preemptions, running and waiting requests"""
    while not stop.is_set():
        try:
            txt = urllib.request.urlopen("http://127.0.0.1:8000/metrics", timeout=5).read().decode()
            row = {"t": time.time()}
            for line in txt.splitlines():
                if line.startswith("vllm:") and any(s in line for s in ("cache_usage", "preempt", "num_requests_running", "num_requests_waiting")):
                    name = line.split("{")[0].split(" ")[0]
                    try: row[name] = row.get(name, 0.0) + float(line.rsplit(" ", 1)[1])
                    except ValueError: pass
            out.append(row)
        except Exception:
            pass
        stop.wait(10)

# %% [code]
# Cell 6 — run one phase
def run_phase(phase, reps, kind):
    rd = os.path.abspath(f"run_{phase}"); os.makedirs(rd, exist_ok=True)
    for f in glob.glob(os.path.join(rd, "*")): os.remove(f)
    backends = ",".join(f"http://127.0.0.1:{8000 + r}" for r in range(CFG["replicas"]))
    gw = subprocess.Popen([sys.executable, os.path.abspath("gateway9.py"), "--port", "9000", "--backends", backends, "--lb", CFG["lb"][phase],
                           "--rpm", str(CFG["rpm"]), "--log", os.path.join(rd, "gateway_log.csv")],
                          stdout=open(os.path.join(rd, "gateway.out"), "w"), stderr=subprocess.STDOUT)
    assert wait_http(CFG["gateway"] + "/v1/models", 60, gw), tail(os.path.join(rd, "gateway.out"))
    dcfg = dict(CFG, pops=CFG["pops"][phase], run_s=CFG["run_s"][phase], lengths=CFG["lengths"][phase],
                model=CFG["model"] if kind == "vllm" else "sim", wc_file=os.path.abspath("wc.json"), t_start=time.time() + 40)
    json.dump(dcfg, open(os.path.join(rd, "driver_cfg.json"), "w"))
    procs = {p: subprocess.Popen([sys.executable, os.path.abspath("driver9.py"), "driver_cfg.json", p], cwd=rd,
                                 stdout=open(os.path.join(rd, f"drv_{p}.out"), "w"), stderr=subprocess.STDOUT) for p in dcfg["pops"]}
    t_run = time.time(); deadline = dcfg["t_start"] + dcfg["run_s"] + 300
    while any(pr.poll() is None for pr in procs.values()):
        time.sleep(60)
        n = sum(1 for _ in open(os.path.join(rd, "gateway_log.csv"))) - 1
        print(f"[{phase}] t={time.time() - dcfg['t_start']:.0f}s log rows={n} running={sum(pr.poll() is None for pr in procs.values())}", flush=True)
        if time.time() > deadline:
            for pr in procs.values():
                if pr.poll() is None: pr.terminate()
    gw.terminate(); time.sleep(3)
    tails = {}
    for p in procs:
        txt = open(os.path.join(rd, f"drv_{p}.out")).read().strip().splitlines(); tails[p] = txt[-1][:200] if txt else ""
    cfg_phase = dict(dcfg, expected=CFG["expected"], wm_mode=CFG["wm_mode"], warmup_s=CFG["warmup_s"], pops=CFG["pops"][phase])
    return dict(lb=CFG["lb"][phase], wall_min=round((time.time() - t_run) / 60, 1), driver_tail=tails), cfg_phase, rd

# %% [code]
# Cell 7 — both phases on the same two replicas
import importlib, occlib, fplib, nb9an
importlib.reload(occlib); importlib.reload(fplib); importlib.reload(nb9an)
print("fplib", fplib.__version__)
reps, kind = start_replicas()
RES = {"wildchat": WC_INFO, "fplib": fplib.__version__, "backend": kind, "install": BACKEND.get("install")}
PHASES = os.environ.get("NB9_PHASES", "AB")
for ph in PHASES:
    run, cfgp, rd = run_phase(ph, reps, kind)
    RES[ph + "_run"] = run
    RES[ph] = nb9an.analyze(rd, cfgp, np.random.default_rng(CFG["seed"] + ord(ph)), ph)
    pd.read_csv(os.path.join(rd, "gateway_log.csv")).to_csv(f"nb9_requests_{ph}.csv", index=False)
    print(json.dumps({k: RES[ph][k] for k in ("coupling", "failover_429", "site")}, indent=1, default=str)[:3000])
for p in reps:
    try: p.terminate()
    except Exception: pass

# %% [code]
# Cell 8 — scorecard against the locked predictions, then the summary to paste
def card(R):
    pools = R["pools"]; site = R["site"]; nh = R["null_humans_40"]["R"]
    hum_ring = pools.get("hum_random40", {}).get("ring_z_multiscale")
    rec7 = site["spectral_v07"]["recovery"].get("F6_rrshared") or {}
    rec8 = site["spectral_v08"]["recovery"].get("F6_rrshared") or {}
    fo = R["failover_429"]
    return dict(
        M1=dict(R_frac_abs_z_gt3=nh["frac_abs_z_gt3"], human_pool_ring_z=hum_ring,
                humans_selected=dict(sideband_v04=site["sideband_graph"]["humans"], graph_v08=site["graph_v08"]["humans"], v06=site["spectral_v06"]["humans"], v07=site["spectral_v07"]["humans"], v08=site["spectral_v08"]["humans"]),
                pass_=bool(nh["frac_abs_z_gt3"] <= 0.05 and (hum_ring or 0) < 6 and max(site["graph_v08"]["humans"], site["spectral_v07"]["humans"], site["spectral_v08"]["humans"]) <= 2)),
        M9=dict(humans_v04=site["sideband_graph"]["humans"], linked_to=site["sideband_graph"].get("human_linked_to"),
                pass_=bool(set(site["sideband_graph"].get("human_linked_to") or []) <= {"F1_rot429_shared", "F2_rot429_worker"})),
        M2=R.get("same_replica_human_pools"),
        M3={p: (pools[p]["predicted_v07"], pools[p]["correct_v07"]) for p in ("F6_rrshared", "F7_shuffle", "C_company", "F8_pinned", "F9_open_steady") if p in pools},
        M4={p: (pools[p]["predicted_v07"], pools[p]["correct_v07"]) for p in ("F1_rot429_shared", "F2_rot429_worker", "F3_hashed", "F4_sticky_ttl", "F5_hedged") if p in pools},
        M5={p: fo.get(p) for p in ("F1_rot429_shared", "F2_rot429_worker", "F8_pinned")},
        M6=dict(F6_v08=rec8, humans_v08=site["spectral_v08"]["humans"], trimmed_v08=site["spectral_v08"]["trimmed"],
                pass_=bool((rec8.get("recall") or 0) >= 0.9 and (rec8.get("purity") or 0) >= 0.9 and site["spectral_v08"]["humans"] <= 2),
                F6_v07=rec7, trimmed_v07=site["spectral_v07"]["trimmed"], v07_fails_as_predicted=bool((rec7.get("recall") or 0) < 0.9)),
        M7=dict(pinned_req_s_per_worker=pools.get("F8_pinned", {}).get("req_per_s_per_worker")),
        M8=dict(auc_20min={p: (R.get("expansion", {}).get(p, {}).get("20min") or {}).get("auc") for p in ("F7_shuffle", "C_company", "F3_hashed", "F5_hedged")},
                pass_=bool(all(((R.get("expansion", {}).get(p, {}).get("20min") or {}).get("auc") or 0) >= 0.85 for p in ("F7_shuffle", "C_company", "F3_hashed", "F5_hedged")))),
        classification_v07=f"{sum(r['correct_v07'] for r in pools.values())}/{len(pools)}",
        classification_v06=f"{sum(r['correct_v06'] for r in pools.values())}/{len(pools)}")
RES["scorecard"] = {ph: card(RES[ph]) for ph in PHASES}
RES["runtime_min"] = round((time.time() - T_START) / 60, 1)
json.dump(RES, open("nb9_results.json", "w"), indent=1, default=str)
print(json.dumps(RES["scorecard"], indent=1, default=str))
print("===== PASTE FROM HERE =====")
print(json.dumps(RES, separators=(",", ":"), default=str))
print("===== PASTE TO HERE =====")
