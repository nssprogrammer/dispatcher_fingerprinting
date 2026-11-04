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
