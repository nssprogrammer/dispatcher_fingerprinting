"""Workload driver: runs every client population at once against the gateway, each on its own API keys.
Ground truth (key -> workload) is written to labels.json. Client code mirrors how such tools are really written:
the OpenAI SDK with key rotation, LiteLLM's Router over many keys, agent loops, an open-loop relay."""
import asyncio, json, sys, time, itertools, random, math, logging
import numpy as np
from openai import AsyncOpenAI

cfg = json.load(open(sys.argv[1]))
rng = np.random.default_rng(cfg["seed"]); random.seed(cfg["seed"])
BASE = cfg["gateway"] + "/v1"; MODEL = cfg["model"]
T0 = time.time(); T_END = T0 + cfg["run_s"]
labels = {}
stats = {"ok": 0, "err": 0}
err_types = {}
_clients = {}

def keys(prefix, n):
    ks = [f"{prefix}-{i:03d}" for i in range(n)]
    for k in ks: labels[k] = prefix
    return ks

def client(key):
    if key not in _clients:
        _clients[key] = AsyncOpenAI(api_key=key, base_url=BASE, max_retries=0, timeout=300)
    return _clients[key]

def ntok(med=None):
    med = med or cfg["tok_med"]
    return int(np.clip(rng.lognormal(math.log(med), cfg["tok_sig"]), 16, 512))

def prompt():
    return [{"role": "user", "content": f"Write a short note about topic {random.randint(0, 10**6)}."}]

async def call(key, n):
    try:
        s = await client(key).chat.completions.create(model=MODEL, messages=prompt(), max_tokens=n, stream=True,
                                                      extra_body={"ignore_eos": True})
        async for _ in s:
            pass
        stats["ok"] += 1
    except Exception as ex:
        stats["err"] += 1
        k = f"{labels.get(key, '?')}:{type(ex).__name__}:{str(ex)[:80]}"
        err_types[k] = err_types.get(k, 0) + 1
        await asyncio.sleep(0.5)

def alive():
    return time.time() < T_END

# ---------------------------------------------------------------- benign populations
async def human(key):
    h = cfg["human"]
    await asyncio.sleep(rng.exponential(h["gap_s"]))
    while alive():
        for _ in range(rng.geometric(1 / h["mean_req"])):
            if not alive(): return
            await call(key, ntok())
            await asyncio.sleep(rng.lognormal(math.log(h["think_med_s"]), h["think_sig"]))
        await asyncio.sleep(rng.exponential(h["gap_s"]))

async def batch_client(key):          # legitimate single-key closed-loop job
    await asyncio.sleep(rng.uniform(0, 5))
    while alive():
        await call(key, ntok()); await asyncio.sleep(rng.lognormal(math.log(0.2), 0.5))

# ---------------------------------------------------------------- campaign tools
async def rotating_script(ks, K, jitter=0.0):
    """Typical distillation script: K async workers sharing a round-robin key cycle."""
    cyc = itertools.cycle(ks)
    async def worker():
        await asyncio.sleep(rng.uniform(0, 2))
        while alive():
            await call(next(cyc), ntok())
            await asyncio.sleep(rng.lognormal(math.log(0.2), 0.5) + (rng.uniform(0, jitter) if jitter else 0.0))
    await asyncio.gather(*[worker() for _ in range(K)])

async def litellm_router(ks, K, strategy):
    import litellm
    from litellm import Router
    litellm.suppress_debug_info = True
    if hasattr(litellm, "REPEATED_STREAMING_CHUNK_LIMIT"):
        litellm.REPEATED_STREAMING_CHUNK_LIMIT = 10 ** 9   # forced-length generations can repeat; do not abort them
    logging.getLogger("LiteLLM").setLevel(logging.ERROR); logging.getLogger("LiteLLM Router").setLevel(logging.ERROR)
    router = Router(model_list=[{"model_name": "teacher", "litellm_params": {"model": f"openai/{MODEL}", "api_base": BASE, "api_key": k}}
                                for k in ks], routing_strategy=strategy, num_retries=0)
    async def worker():
        await asyncio.sleep(rng.uniform(0, 2))
        while alive():
            try:
                r = await router.acompletion(model="teacher", messages=prompt(), max_tokens=ntok(), stream=True,
                                             extra_body={"ignore_eos": True})
                async for _ in r:
                    pass
                stats["ok"] += 1
            except Exception as ex:
                stats["err"] += 1
                k = f"A3llm:{type(ex).__name__}:{str(ex)[:80]}"
                err_types[k] = err_types.get(k, 0) + 1
                await asyncio.sleep(0.5)
            await asyncio.sleep(rng.lognormal(math.log(0.2), 0.5))
    await asyncio.gather(*[worker() for _ in range(K)])

async def agent_fleet(ks, n_agents):
    """Agents: each episode holds one key; LLM call -> tool execution -> next call."""
    a = cfg["agent"]
    async def agent():
        await asyncio.sleep(rng.uniform(0, 5))
        while alive():
            key = ks[rng.integers(len(ks))]
            for _ in range(rng.geometric(1 / a["mean_steps"])):
                if not alive(): return
                await call(key, ntok(a["tok_med"]))
                await asyncio.sleep(rng.lognormal(math.log(a["tool_med_s"]), 0.6))
            await asyncio.sleep(rng.lognormal(math.log(1.0), 0.5))
    await asyncio.gather(*[agent() for _ in range(n_agents)])

async def open_loop_relay(ks, lam):
    """Relay forwarding independent end-user requests: arrivals do not wait for completions."""
    tasks = set()
    while alive():
        await asyncio.sleep(rng.exponential(1 / lam))
        t = asyncio.create_task(call(ks[rng.integers(len(ks))], ntok()))
        tasks.add(t); t.add_done_callback(tasks.discard)
    if tasks:
        await asyncio.wait(tasks, timeout=120)

async def main():
    w = cfg["workloads"]; jobs = []
    jobs += [human(k) for k in keys("hum", w["humans"])]
    jobs += [batch_client(k) for k in keys("bat", w["batch_clients"])]
    jobs.append(rotating_script(keys("A1sem", 40), w["A1_K"]))
    jobs.append(rotating_script(keys("A2jit", 40), w["A2_K"], jitter=w["A2_jitter_s"]))
    jobs.append(litellm_router(keys("A3llm", 40), w["A3_K"], "simple-shuffle"))
    jobs.append(agent_fleet(keys("A4agt", 40), w["A4_agents"]))
    jobs.append(open_loop_relay(keys("A5rel", 40), w["A5_rate"]))
    jobs.append(rotating_script(keys("Ccomp", 10), w["C_K"]))     # legitimate company: one queue over 10 keys
    json.dump(labels, open("labels.json", "w"))

    async def progress():
        while alive():
            await asyncio.sleep(60)
            print(f"[driver] t={time.time() - T0:.0f}s ok={stats['ok']} err={stats['err']}", flush=True)
    await asyncio.gather(*jobs, progress())
    print(f"[driver] done ok={stats['ok']} err={stats['err']}", flush=True)
    top = sorted(err_types.items(), key=lambda kv: -kv[1])[:8]
    print("[driver] errors " + json.dumps(dict(top)), flush=True)
    json.dump(err_types, open("driver_errors.json", "w"))

asyncio.run(main())
