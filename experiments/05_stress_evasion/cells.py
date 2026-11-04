# %% [markdown]
# # Notebook 8 — Stress test, frozen re-test of spectral v0.5, and the cost of evasion
# Two phases in one session (≈ 2.5–3 h on a T4).
#
# **Phase A — stress (60 min).**
# - Prompt and output lengths are drawn from WildChat (up to 1,024 output tokens), not a fixed 96 tokens.
# - vLLM's KV cache is squeezed so the server preempts sequences.
# - Humans use WildChat think times and session lengths.
# - Populations are as in NB7. The answer end carries the NB7 hold-back watermark (0 or 150 ms).
# - Every detector runs **frozen**: classifier thresholds from simulation, sideband graph (after NB6) and spectral rings **v0.5** (after NB7).
#
# **Phase B — evasion gradient (40 min).** NB7-style short answers. Twelve attacker variants on 20 keys each, including jitter, keys held for H requests, workers near the number of keys, pinned keys, and steady and bursty open loops. The watermark is **spread** over the stream instead of held at the end.
#
# **Locked predictions** (scored at the end):
# - **A0** the stress is real: KV-cache usage reaches ≥ 0.85, or vLLM reports preemptions.
# - **A1** frozen classifier (v0.5, 20 s high-pass) ≥ 10 of 12 pools correct. *Revised after the dry run on the simulated server, before any vLLM data:* with long answers a ring's period T_c = M/λ can exceed the 20 s block, which then removes the ring. So we expect v0.5 to **miss rings with T_c > 20 s**, and the A1 count may fail for that reason.
# - **A1b** multi-scale classifier (v0.6: ring statistic at 20, 60 and 180 s blocks, defined after the dry run and before this run) ≥ 10 of 12 correct.
# - **A2** spectral v0.5: P3, P5 and P8 each have recall ≥ 0.9 and purity ≥ 0.9; at most 2 humans selected (expected to fail for rings with T_c > 20 s).
# - **A2b** spectral v0.6 (union over 20, 60 and 180 s blocks): P3, P5 and P8 recall ≥ 0.9; at most 2 humans.
# - **A3** sideband graph: P3 and P4 recall ≥ 0.9; at most 2 humans.
# - **A4** random relay P7: no cross-key echo (p > 0.01), replicating NB7's anomaly as chance.
# - **A5** nulls hold under stress: ≤ 2 of 50 human pools at watermark p < 0.01, and ≤ 5% of human pools at |z_R| > 3.
# - **A6** worker count (interactive response-time law) within 25% for P1, P2, P3, P4 and C.
# - **B1** ring z > 6 for the shared counter at jitter 0, 2, 6 and 15 s, and with 24 workers at 15 s.
# - **B2** ring z > 3 when workers hold a key for 5 requests; with 20 requests the pool is still flagged by turn-taking (z_R > 3).
# - **B3** classification (v0.6) ≥ 10 of 12 phase-B variants correct (alternatives listed per pool).
# - **B4** spread watermark: cross-key p < 0.001 for E0, E7 and E8; p > 0.001 for E2 and E3, as the jitter model predicts.
# - **B5** visibility: the coin is readable from the last inter-chunk gap with the hold-back (AUC ≥ 0.9, phase A) but not with the spread delay (AUC ≤ 0.6, phase B).
# - **B6** pinned keys (E9) and the steady open loop (E10): class none, cross-key echo p > 0.01.
# - **B7** the bursty open loop (E11) is flagged as bunching (z_R < −3).
#
# **Run settings.** Accelerator **GPU T4 ×1**, Internet **On**. Paste back the PASTE block and upload `nb8_requests_A.csv` and `nb8_requests_B.csv`.

# %% [code]
# Cell 1 — settings
import os, sys, json, time, subprocess, signal, glob, threading, urllib.request
T_START = time.time()
NH = int(os.environ.get("NB8_HUMANS", 400))
HUM = lambda: {"hum_a": dict(kind="human", n=NH // 2, key_offset=0), "hum_b": dict(kind="human", n=NH - NH // 2, key_offset=NH // 2),
               "bat": dict(kind="batch", n=4)}
CFG = dict(
    backend=os.environ.get("NB8_BACKEND", "vllm"), model="Qwen/Qwen2.5-0.5B-Instruct", seed=8,
    run_s={"A": int(os.environ.get("NB8_RUN_A", 3600)), "B": int(os.environ.get("NB8_RUN_B", 2400))},
    warmup_s=int(os.environ.get("NB8_WARMUP", 240)),
    tok_med=96, tok_sig=0.6, client_gap_med=0.2, max_out=1024, max_prompt=1500,
    human=dict(gap_s=float(os.environ.get("NB8_HUMAN_GAP", 360)), mean_req=6, think_med_s=6.0, think_sig=0.9),
    agent=dict(mean_steps=8, tool_med_s=2.0, tok_med=64),
    lengths={"A": "wildchat", "B": "fixed"}, humans="wildchat",
    wm_mode={"A": "holdback", "B": "spread"}, wm_delta=0.15,
    vllm_extra={"A": [["--num-gpu-blocks-override", "1600"], ["--gpu-memory-utilization", "0.12"]], "B": [[]]},
    pops={
        "A": dict(HUM(),
                  P1_shuffle=dict(kind="litellm", strategy="simple-shuffle", K=6, M=40),
                  P2_least=dict(kind="litellm", strategy="least-busy", K=10, M=40),
                  P3_rrshared=dict(kind="rr_shared", K=12, M=40), P4_rrworker=dict(kind="rr_worker", K=8, M=40),
                  P5_rrjitter=dict(kind="rr_shared", K=8, M=40, jitter=6.0), P6_agents=dict(kind="agents", K=6, M=32),
                  P7_relay_random=dict(kind="open_loop", lam=1.0, policy="random", M=40),
                  P8_relay_rr=dict(kind="open_loop", lam=1.0, policy="rr", M=40),
                  C_company=dict(kind="litellm", strategy="simple-shuffle", K=4, M=10), P9_pinned=dict(kind="pinned", K=8, M=8)),
        "B": dict(HUM(),
                  E0_rr_J0=dict(kind="rr_shared", K=8, M=20), E1_rr_J2=dict(kind="rr_shared", K=8, M=20, jitter=2.0),
                  E2_rr_J6=dict(kind="rr_shared", K=8, M=20, jitter=6.0), E3_rr_J15=dict(kind="rr_shared", K=8, M=20, jitter=15.0),
                  E4_rr_J15_K24=dict(kind="rr_shared", K=24, M=20, jitter=15.0),
                  E5_hold5=dict(kind="rr_hold", K=8, M=20, H=5), E6_hold20=dict(kind="rr_hold", K=8, M=20, H=20),
                  E7_shuffle_K8=dict(kind="random", K=8, M=20), E8_shuffle_K18=dict(kind="random", K=18, M=20),
                  E9_pinned=dict(kind="pinned", K=8, M=8),
                  E10_open_steady=dict(kind="open_loop", lam=3.0, policy="random", M=20),
                  E11_open_bursty=dict(kind="open_loop", lam=6.0, policy="random", M=20, on_s=30.0, off_s=60.0)),
    },
    expected={
        "A": {"hum_random40": "none", "benign_36hum_4bat": "none", "P1_shuffle": "random / shuffle", "P2_least": "sticky / episodic",
              "P3_rrshared": "shared round-robin (closed loop)", "P4_rrworker": "per-worker round-robin",
              "P5_rrjitter": "shared round-robin (closed loop)", "P6_agents": "sticky / episodic", "P7_relay_random": "none",
              "P8_relay_rr": "shared round-robin (open-loop relay)", "C_company": "random / shuffle", "P9_pinned": "none"},
        "B": {"hum_random40": "none", "E0_rr_J0": "shared round-robin (closed loop)", "E1_rr_J2": "shared round-robin (closed loop)",
              "E2_rr_J6": "shared round-robin (closed loop)", "E3_rr_J15": "shared round-robin (closed loop)",
              "E4_rr_J15_K24": "shared round-robin (closed loop) | shared round-robin (open-loop relay)",
              "E5_hold5": "shared round-robin (closed loop) | sticky / episodic", "E6_hold20": "shared round-robin (closed loop) | sticky / episodic",
              "E7_shuffle_K8": "random / shuffle", "E8_shuffle_K18": "random / shuffle", "E9_pinned": "none", "E10_open_steady": "none",
              "E11_open_bursty": "bunching"},
    },
    gateway="http://127.0.0.1:9000",
)
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
GATEWAY8_PLACEHOLDER = True

# %% [code]
DRIVER8_PLACEHOLDER = True

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

def start_backend(phase):
    """returns (proc, kind, args used). Tries the phase's argument sets in order, then eager mode."""
    if BACKEND["python"] is None:
        p = subprocess.Popen([sys.executable, "simserver.py", "--port", "8000"], stdout=open(f"sim_{phase}.log", "w"), stderr=subprocess.STDOUT)
        assert wait_http("http://127.0.0.1:8000/health", 60, p); return p, "sim", []
    base = [BACKEND["python"], "-m", "vllm.entrypoints.openai.api_server", "--model", CFG["model"], "--dtype", "half",
            "--max-model-len", "2048", "--max-num-seqs", "128", "--port", "8000"]
    for extra in CFG["vllm_extra"][phase] + [["--enforce-eager"]]:
        args = list(extra)
        if "--gpu-memory-utilization" not in args: args += ["--gpu-memory-utilization", "0.85"]
        p = subprocess.Popen(base + args, stdout=open(f"vllm_{phase}.log", "w"), stderr=subprocess.STDOUT)
        if wait_http("http://127.0.0.1:8000/v1/models", 900, p):
            return p, "vllm", args
        print("vLLM failed with", args, "\n", tail(f"vllm_{phase}.log", 12)); p.kill(); time.sleep(5)
    raise RuntimeError("vLLM could not start for phase " + phase)

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
# Cell 6 — run one phase: model server, watermarking gateway, every population in its own process
def run_phase(phase):
    rd = os.path.abspath(f"run_{phase}"); os.makedirs(rd, exist_ok=True)
    for f in glob.glob(os.path.join(rd, "*")): os.remove(f)
    proc, kind, args = start_backend(phase)
    gw = subprocess.Popen([sys.executable, os.path.abspath("gateway8.py"), "--port", "9000", "--backend", "http://127.0.0.1:8000",
                           "--log", os.path.join(rd, "gateway_log.csv"), "--delta", str(CFG["wm_delta"]), "--seed", str(4242 + ord(phase)),
                           "--mode", CFG["wm_mode"][phase]], stdout=open(os.path.join(rd, "gateway.out"), "w"), stderr=subprocess.STDOUT)
    assert wait_http(CFG["gateway"] + "/v1/models", 60, gw), tail(os.path.join(rd, "gateway.out"))
    model = CFG["model"] if kind == "vllm" else "sim"
    dcfg = dict(CFG, pops=CFG["pops"][phase], run_s=CFG["run_s"][phase], lengths=CFG["lengths"][phase], model=model,
                wc_file=os.path.abspath("wc.json"), t_start=time.time() + 40)
    json.dump(dcfg, open(os.path.join(rd, "driver_cfg.json"), "w"))
    stop, mets = threading.Event(), []
    th = threading.Thread(target=poll_metrics, args=(stop, mets), daemon=True); th.start()
    procs = {p: subprocess.Popen([sys.executable, os.path.abspath("driver8.py"), "driver_cfg.json", p], cwd=rd,
                                 stdout=open(os.path.join(rd, f"drv_{p}.out"), "w"), stderr=subprocess.STDOUT) for p in dcfg["pops"]}
    t_run = time.time()
    while any(pr.poll() is None for pr in procs.values()):
        time.sleep(120)
        n = sum(1 for _ in open(os.path.join(rd, "gateway_log.csv"))) - 1
        last = mets[-1] if mets else {}
        print(f"[{phase}] t={time.time() - dcfg['t_start']:.0f}s requests={n} running={sum(pr.poll() is None for pr in procs.values())} "
              f"metrics={ {k.split(':')[-1]: round(v, 3) for k, v in last.items() if k != 't'} }", flush=True)
    stop.set(); gw.terminate(); proc.terminate(); time.sleep(10)
    try: proc.kill()
    except Exception: pass
    tails = {}
    for p in procs:
        txt = open(os.path.join(rd, f"drv_{p}.out")).read().strip().splitlines()
        tails[p] = txt[-1][:200] if txt else ""
    m = pd.DataFrame(mets)
    msum = {}
    for c in m.columns:
        if c == "t": continue
        v = m[c].dropna()
        if len(v): msum[c.split(":")[-1]] = dict(max=round(float(v.max()), 3), p90=round(float(v.quantile(.9)), 3), last=round(float(v.iloc[-1]), 3))
    cfg_phase = dict(dcfg, expected=CFG["expected"], wm_mode=CFG["wm_mode"], warmup_s=CFG["warmup_s"], pops=CFG["pops"][phase])
    return dict(backend=kind, vllm_args=args, wall_min=round((time.time() - t_run) / 60, 1), driver_tail=tails, metrics=msum), cfg_phase, rd

# %% [code]
# Cell 7 — Phase A: stress run and frozen analysis
import importlib, occlib, fplib, nb8an
importlib.reload(occlib); importlib.reload(fplib); importlib.reload(nb8an)
print("fplib", fplib.__version__)
RES = {"wildchat": WC_INFO, "fplib": fplib.__version__}
runA, cfgA, rdA = run_phase("A")
RES["A_run"] = runA
RES["A"] = nb8an.analyze(rdA, cfgA, np.random.default_rng(CFG["seed"]), "A")
pd.read_csv(os.path.join(rdA, "gateway_log.csv")).to_csv("nb8_requests_A.csv", index=False)
print(json.dumps({k: RES["A"][k] for k in ("site", "watermark_human_null")}, indent=1, default=str))

# %% [code]
# Cell 8 — Phase B: evasion gradient
runB, cfgB, rdB = run_phase("B")
RES["B_run"] = runB
RES["B"] = nb8an.analyze(rdB, cfgB, np.random.default_rng(CFG["seed"] + 1), "B")
pd.read_csv(os.path.join(rdB, "gateway_log.csv")).to_csv("nb8_requests_B.csv", index=False)

# %% [code]
# Cell 9 — scorecard against the locked predictions, then the summary to paste
A, B = RES["A"], RES["B"]
P = lambda R, p, k="pool_w1": R["watermark"].get(p, {}).get(k, {}).get("p", float("nan"))
mA = runA["metrics"]
kv = max([v["max"] for k, v in mA.items() if "cache_usage" in k] or [float("nan")])
pre = max([v["max"] for k, v in mA.items() if "preempt" in k] or [0.0])
ok_k = lambda R, p: bool(R["pools"].get(p, {}).get("K_little") and R["pools"][p]["K_true"] and abs(R["pools"][p]["K_little"] / R["pools"][p]["K_true"] - 1) <= 0.25)
rec = lambda R, det, p, x: (R["site"][det]["recovery"].get(p) or {}).get(x) or 0
RES["scorecard"] = dict(
    A0_stress=dict(kv_cache_usage_max=kv, preemptions=pre, pass_=bool((kv >= 0.85) or (pre > 0))),
    A1_classification=f"{sum(r['correct'] for r in A['pools'].values())}/{len(A['pools'])} (pass if >= 10)",
    A1b_classification_v06=f"{sum(r['correct_v06'] for r in A['pools'].values())}/{len(A['pools'])} (pass if >= 10)",
    ring_periods_A={p: A["pools"][p]["ring_period_s"] for p in ("P3_rrshared", "P5_rrjitter", "P8_relay_rr") if p in A["pools"]},
    A2b_spectral_v06=dict(**{p: A["site"]["spectral_v06"]["recovery"].get(p) for p in ("P3_rrshared", "P5_rrjitter", "P8_relay_rr")},
                          humans=A["site"]["spectral_v06"]["humans"],
                          pass_=all(rec(A, "spectral_v06", p, "recall") >= 0.9 for p in ("P3_rrshared", "P5_rrjitter", "P8_relay_rr")) and A["site"]["spectral_v06"]["humans"] <= 2),
    A2_spectral=dict(**{p: A["site"]["spectral"]["recovery"].get(p) for p in ("P3_rrshared", "P5_rrjitter", "P8_relay_rr")},
                     humans=A["site"]["spectral"]["humans"],
                     pass_=all(rec(A, "spectral", p, "recall") >= 0.9 and rec(A, "spectral", p, "purity") >= 0.9 for p in ("P3_rrshared", "P5_rrjitter", "P8_relay_rr"))
                     and A["site"]["spectral"]["humans"] <= 2),
    A3_sideband=dict(P3=rec(A, "sideband_graph", "P3_rrshared", "recall"), P4=rec(A, "sideband_graph", "P4_rrworker", "recall"), humans=A["site"]["sideband_graph"]["humans"],
                     pass_=rec(A, "sideband_graph", "P3_rrshared", "recall") >= 0.9 and rec(A, "sideband_graph", "P4_rrworker", "recall") >= 0.9 and A["site"]["sideband_graph"]["humans"] <= 2),
    A4_random_relay_p=P(A, "P7_relay_random"), A4_pass=bool(P(A, "P7_relay_random") > 0.01),
    A5_nulls=dict(watermark=A["watermark_human_null"], R_frac_abs_z_gt3=A["null_humans_40"]["R"]["frac_abs_z_gt3"],
                  pass_=bool(A["watermark_human_null"]["n_p_below_0_01"] <= 2 and A["null_humans_40"]["R"]["frac_abs_z_gt3"] <= 0.05)),
    A6_K={p: ok_k(A, p) for p in ("P1_shuffle", "P2_least", "P3_rrshared", "P4_rrworker", "C_company")},
    B1_ring_z={p: (B["pools"][p]["ring_z"], B["pools"][p]["ring_z_multiscale"]) for p in ("E0_rr_J0", "E1_rr_J2", "E2_rr_J6", "E3_rr_J15", "E4_rr_J15_K24") if p in B["pools"]},
    B2_hold={p: dict(ring_z=B["pools"][p]["ring_z_multiscale"], z_R=B["pools"][p]["z_R"]) for p in ("E5_hold5", "E6_hold20") if p in B["pools"]},
    B3_classification_v06=f"{sum(r['correct_v06'] for r in B['pools'].values())}/{len(B['pools'])} (pass if >= 10)",
    B4_spread_watermark={p: P(B, p) for p in ("E0_rr_J0", "E7_shuffle_K8", "E8_shuffle_K18", "E2_rr_J6", "E3_rr_J15")},
    B5_visibility=dict(holdback=A.get("client_visibility_ms"), spread=B.get("client_visibility_ms")),
    B6_pinned_open=dict(E9=(B["pools"].get("E9_pinned", {}).get("predicted"), P(B, "E9_pinned")),
                        E10=(B["pools"].get("E10_open_steady", {}).get("predicted"), P(B, "E10_open_steady"))),
    B7_bursty=dict(predicted=B["pools"].get("E11_open_bursty", {}).get("predicted"), z_R=B["pools"].get("E11_open_bursty", {}).get("z_R")),
    cost_curve={p: dict(req_s_per_worker=B["pools"][p]["req_per_s_per_worker"], req_s=B["pools"][p]["req_per_s"], z_R=B["pools"][p]["z_R"],
                        ring_z=B["pools"][p]["ring_z_multiscale"], wm_p=P(B, p), cls=B["pools"][p]["predicted_v06"]) for p in B["pools"] if p.startswith("E")},
)
RES["runtime_min"] = round((time.time() - T_START) / 60, 1)
json.dump(RES, open("nb8_results.json", "w"), indent=1, default=str)
print(json.dumps(RES["scorecard"], indent=1, default=str))
print("===== PASTE FROM HERE =====")
print(json.dumps(RES, separators=(",", ":"), default=str))
print("===== PASTE TO HERE =====")
