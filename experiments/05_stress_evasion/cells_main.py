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
