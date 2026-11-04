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
