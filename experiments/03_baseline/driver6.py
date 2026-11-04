"""One client population per process (no shared event loop between populations).
usage: python driver6.py cfg.json POP
Writes labels_POP.json (key -> POP, ring position) and clientlog_POP.csv (rid, key, t_sched, t_send, t_done, ok).
Every request carries an X-Req-Id header so client-side and gateway-side timestamps can be joined."""
import asyncio, json, sys, time, itertools, random, math, logging, uuid
import numpy as np
from openai import AsyncOpenAI

cfg = json.load(open(sys.argv[1])); POP = sys.argv[2]; P = cfg["pops"][POP]
seed = cfg["seed"] * 1000 + sorted(cfg["pops"]).index(POP)
rng = np.random.default_rng(seed); random.seed(seed)
BASE = cfg["gateway"] + "/v1"; MODEL = cfg["model"]
T0 = cfg["t_start"]; T_END = T0 + cfg["run_s"]
labels = {}; stats = {"ok": 0, "err": 0}; err_types = {}; _clients = {}
clog = open(f"clientlog_{POP}.csv", "w", buffering=1); clog.write("rid,key,t_sched,t_send,t_done,ok\n")

def keys(n, prefix=None):
    prefix = prefix or POP
    ks = [f"{prefix}-{i:03d}" for i in range(P.get("key_offset", 0), P.get("key_offset", 0) + n)]
    for i, k in enumerate(ks): labels[k] = dict(pop=POP, pos=i)
    return ks

def client(key):
    if key not in _clients:
        _clients[key] = AsyncOpenAI(api_key=key, base_url=BASE, max_retries=0, timeout=300)
    return _clients[key]

def ntok(med=None):
    return int(np.clip(rng.lognormal(math.log(med or cfg["tok_med"]), cfg["tok_sig"]), 16, 512))

def prompt():
    return [{"role": "user", "content": f"Write a short note about topic {random.randint(0, 10**6)}."}]

def note_err(ex):
    stats["err"] += 1
    k = f"{POP}:{type(ex).__name__}:{str(ex)[:80]}"; err_types[k] = err_types.get(k, 0) + 1

async def call(key, n, t_sched=None):
    rid = uuid.uuid4().hex[:16]; t_send = time.time(); ok = 0
    try:
        s = await client(key).chat.completions.create(model=MODEL, messages=prompt(), max_tokens=n, stream=True,
                                                      extra_body={"ignore_eos": True}, extra_headers={"X-Req-Id": rid})
        async for _ in s:
            pass
        stats["ok"] += 1; ok = 1
    except Exception as ex:
        note_err(ex); await asyncio.sleep(0.5)
    clog.write(f"{rid},{key},{t_sched or t_send:.6f},{t_send:.6f},{time.time():.6f},{ok}\n")

def alive():
    return time.time() < T_END

async def start_delay(hi):
    await asyncio.sleep(max(0.0, T0 - time.time()) + rng.uniform(0, hi))

def gap():
    return rng.lognormal(math.log(cfg["client_gap_med"]), 0.5)

# ---------------------------------------------------------------- benign
async def human(key):
    h = cfg["human"]
    await start_delay(0); await asyncio.sleep(rng.exponential(h["gap_s"]))
    while alive():
        for _ in range(rng.geometric(1 / h["mean_req"])):
            if not alive(): return
            await call(key, ntok())
            await asyncio.sleep(rng.lognormal(math.log(h["think_med_s"]), h["think_sig"]))
        await asyncio.sleep(rng.exponential(h["gap_s"]))

async def batch_client(key):
    await start_delay(5)
    while alive():
        await call(key, ntok()); await asyncio.sleep(gap())

# ---------------------------------------------------------------- dispatchers
async def rr_shared(ks, K, jitter=0.0):
    """K workers share ONE itertools.cycle over the keys (the common 'rotate keys' snippet)"""
    cyc = itertools.cycle(ks)
    async def worker():
        await start_delay(2)
        while alive():
            await call(next(cyc), ntok()); await asyncio.sleep(gap() + (rng.uniform(0, jitter) if jitter else 0.0))
    await asyncio.gather(*[worker() for _ in range(K)])

async def rr_per_worker(ks, K):
    """each worker has its OWN cycle, started at a random key"""
    async def worker():
        i = int(rng.integers(len(ks)))
        await start_delay(2)
        while alive():
            await call(ks[i % len(ks)], ntok()); i += 1; await asyncio.sleep(gap())
    await asyncio.gather(*[worker() for _ in range(K)])

async def litellm_router(ks, K, strategy):
    import litellm
    from litellm import Router
    litellm.suppress_debug_info = True
    if hasattr(litellm, "REPEATED_STREAMING_CHUNK_LIMIT"):
        litellm.REPEATED_STREAMING_CHUNK_LIMIT = 10 ** 9
    for n in ("LiteLLM", "LiteLLM Router", "LiteLLM Proxy"): logging.getLogger(n).setLevel(logging.ERROR)
    router = Router(model_list=[{"model_name": "teacher", "litellm_params": {"model": f"openai/{MODEL}", "api_base": BASE, "api_key": k}}
                                for k in ks], routing_strategy=strategy, num_retries=0)
    async def worker():
        await start_delay(2)
        while alive():
            rid = uuid.uuid4().hex[:16]; t_send = time.time(); ok = 0
            try:
                r = await router.acompletion(model="teacher", messages=prompt(), max_tokens=ntok(), stream=True,
                                             extra_body={"ignore_eos": True}, extra_headers={"X-Req-Id": rid})
                async for _ in r:
                    pass
                stats["ok"] += 1; ok = 1
            except Exception as ex:
                note_err(ex); await asyncio.sleep(0.5)
            clog.write(f"{rid},,{t_send:.6f},{t_send:.6f},{time.time():.6f},{ok}\n")
            await asyncio.sleep(gap())
    await asyncio.gather(*[worker() for _ in range(K)])

async def agent_fleet(ks, n_agents):
    a = cfg["agent"]
    async def agent():
        await start_delay(5)
        while alive():
            key = ks[rng.integers(len(ks))]
            for _ in range(rng.geometric(1 / a["mean_steps"])):
                if not alive(): return
                await call(key, ntok(a["tok_med"]))
                await asyncio.sleep(rng.lognormal(math.log(a["tool_med_s"]), 0.6))
            await asyncio.sleep(rng.lognormal(math.log(1.0), 0.5))
    await asyncio.gather(*[agent() for _ in range(n_agents)])

async def open_loop(ks, lam, policy):
    """Relay of independent end users: Poisson arrivals on an ABSOLUTE schedule (no drift from event-loop lag)."""
    tasks = set(); t = T0; rr = 0
    while True:
        t += rng.exponential(1 / lam)
        if t >= T_END: break
        await asyncio.sleep(max(0.0, t - time.time()))
        if policy == "rr": k = ks[rr % len(ks)]; rr += 1
        else: k = ks[rng.integers(len(ks))]
        task = asyncio.create_task(call(k, ntok(), t_sched=t)); tasks.add(task); task.add_done_callback(tasks.discard)
    if tasks:
        await asyncio.wait(tasks, timeout=120)

async def main():
    kind = P["kind"]
    if kind == "human":
        jobs = [human(k) for k in keys(P["n"], prefix="hum")]
    elif kind == "batch":
        jobs = [batch_client(k) for k in keys(P["n"], prefix="bat")]
    elif kind == "rr_shared":
        jobs = [rr_shared(keys(P["M"]), P["K"], P.get("jitter", 0.0))]
    elif kind == "rr_worker":
        jobs = [rr_per_worker(keys(P["M"]), P["K"])]
    elif kind == "litellm":
        jobs = [litellm_router(keys(P["M"]), P["K"], P["strategy"])]
    elif kind == "agents":
        jobs = [agent_fleet(keys(P["M"]), P["K"])]
    elif kind == "open_loop":
        jobs = [open_loop(keys(P["M"]), P["lam"], P["policy"])]
    else:
        raise ValueError(kind)
    json.dump(labels, open(f"labels_{POP}.json", "w"))
    async def progress():
        while alive():
            await asyncio.sleep(120)
            print(f"[{POP}] t={time.time() - T0:.0f}s ok={stats['ok']} err={stats['err']}", flush=True)
    await asyncio.gather(*jobs, progress())
    print(f"[{POP}] done ok={stats['ok']} err={stats['err']} errors={json.dumps(dict(sorted(err_types.items(), key=lambda kv: -kv[1])[:4]))}", flush=True)
    json.dump(dict(stats=stats, errors=err_types), open(f"driverstats_{POP}.json", "w"))

asyncio.run(main())
