# %% [markdown]
# # Notebook 7 — Active timing watermark + frozen re-test of the site detectors
# **Two questions.**
# 1. **Watermark.** If the provider secretly delays the end of each response by 0 or 150 ms (a fair coin per request), do the *next* requests on the *other* keys of a pool shift by the same amount? A shift proves that one client waits for these responses before dispatching on other keys. The p-value is exact, because the coins were drawn by the provider (Fisher randomisation; the timing-scale version of Proposition 5 in *The Limits of Query Monitoring*).
# 2. **Frozen re-test.** The classifier from NB6 and the two site-wide detectors fixed after NB6 (hand-off graph with a ±60 s sideband null; hollowed spectral ring discovery) run with every setting locked, on a fresh run with a new seed.
#
# **New population.** `P9_pinned`: 8 workers, each permanently on its own key, the case where timing alone provably cannot link keys.
#
# **Locked predictions** (written before this run; scored at the end):
# - **W1** whole-pool echo p < 0.001 for P1 shuffle, P2 least-busy, P3 shared ring, P4 per-worker ring, C company.
# - **W2** P9 pinned: no cross-key echo (p > 0.01), but same-key echo p < 0.001 (automation visible per key, not linkable).
# - **W3** open-loop relays P7 and P8: no echo (p > 0.01).
# - **W4** humans: at most 2 of 50 random 40-key pools reach p < 0.01.
# - **W5** half-to-half linking (echo from one random half of the keys onto the other half) p < 0.001 for P1, P3, P4.
# - **F1** classification ≥ 10 of 12 pools correct (expected class for P2 updated after NB6 to sticky / episodic).
# - **F2** spectral rings: P3, P5, P8 each recall ≥ 0.9 and purity ≥ 0.9; at most 2 humans selected.
# - **F3** sideband hand-off graph: P3, P4 recall ≥ 0.9; at most 2 humans in components.
#
# **Run settings.** Accelerator **GPU T4 ×1**, Internet **On**. Runtime ≈ 90 min. Paste back the block between the PASTE markers, and keep `nb7_requests.csv` from the output folder.

# %% [code]
# Cell 1 — settings
import os, sys, json, time, subprocess, signal, glob
T_START = time.time()
NH = int(os.environ.get("NB7_HUMANS", 400))
CFG = dict(
    backend=os.environ.get("NB7_BACKEND", "vllm"),   # "vllm" = stop if vLLM cannot start; "auto" = fall back to the simulated server; "sim"
    model="Qwen/Qwen2.5-0.5B-Instruct",
    run_s=int(os.environ.get("NB7_RUN_S", 2400)), warmup_s=int(os.environ.get("NB7_WARMUP", 180)), seed=7, wm_delta=0.15,
    tok_med=96, tok_sig=0.6, client_gap_med=0.2,
    human=dict(gap_s=float(os.environ.get("NB7_HUMAN_GAP", 360)), mean_req=6, think_med_s=6.0, think_sig=0.9),
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
        "P9_pinned": dict(kind="pinned", K=8, M=8),
    },
    expected={
        "hum_random40": "none", "benign_36hum_4bat": "none",
        "P1_shuffle": "random / shuffle", "P2_least": "sticky / episodic",   # updated after NB6: on vLLM least-busy workers swap keys every other request
        "P3_rrshared": "shared round-robin (closed loop)", "P4_rrworker": "per-worker round-robin",
        "P5_rrjitter": "shared round-robin (closed loop)", "P6_agents": "sticky / episodic",
        "P7_relay_random": "none", "P8_relay_rr": "shared round-robin (open-loop relay)", "C_company": "random / shuffle", "P9_pinned": "none",
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
GATEWAY7_PLACEHOLDER = True

# %% [code]
DRIVER7_PLACEHOLDER = True
