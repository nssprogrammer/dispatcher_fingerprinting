"""Gateway with several model replicas behind a provider-side balancer and an enforced per-key request quota.
--lb least     : each request goes to the replica with the fewest requests in flight (provider least-outstanding)
--lb affinity  : each key is pinned to replica hash(key) % R (session/prefix-cache affinity)
--rpm Q        : at most Q requests per key in any trailing 60 s; excess requests get HTTP 429 immediately
Logs: key, rid, t_arrival, t_first, t_complete, t_end, status, nbytes, wm_bit, replica"""
import asyncio, time, argparse, zlib, collections, json
import aiohttp
from aiohttp import web

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=9000)
ap.add_argument("--backends", default="http://127.0.0.1:8000,http://127.0.0.1:8001")
ap.add_argument("--lb", default="least", choices=["least", "affinity"])
ap.add_argument("--rpm", type=float, default=0)
ap.add_argument("--log", default="gateway_log.csv")
args = ap.parse_args()
B = args.backends.split(","); inflight = [0] * len(B)
hist = collections.defaultdict(collections.deque)
logf = open(args.log, "a", buffering=1)
if logf.tell() == 0:
    logf.write("key,rid,t_arrival,t_first,t_complete,t_end,status,nbytes,wm_bit,replica\n")
session = None

async def on_start(app):
    global session
    session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0), timeout=aiohttp.ClientTimeout(total=None))

async def on_stop(app):
    await session.close(); logf.close()

def pick(key):
    if args.lb == "affinity":
        return zlib.crc32(key.encode()) % len(B)
    m = min(inflight); return inflight.index(m)

async def proxy(request):
    key = request.headers.get("Authorization", "").replace("Bearer ", "").strip() or "anon"
    rid = request.headers.get("X-Req-Id", "")
    t_arr = time.time(); t_first = None; nbytes = 0; status = 0; rep = -1
    body = await request.read()
    if args.rpm > 0:
        h = hist[key]
        while h and h[0] < t_arr - 60: h.popleft()
        if len(h) >= args.rpm:
            logf.write(f"{key},{rid},{t_arr:.6f},0,{time.time():.6f},{time.time():.6f},429,0,0,-1\n")
            return web.json_response({"error": {"message": "rate limit", "type": "rate_limit_error"}}, status=429)
        h.append(t_arr)
    rep = pick(key); inflight[rep] += 1
    try:
        async with session.post(B[rep] + request.path, data=body, headers={"Content-Type": "application/json"}) as r:
            status = r.status
            resp = web.StreamResponse(status=r.status, headers={"Content-Type": r.headers.get("Content-Type", "text/event-stream")})
            await resp.prepare(request)
            async for chunk in r.content.iter_any():
                if t_first is None: t_first = time.time()
                await resp.write(chunk); nbytes += len(chunk)
            await resp.write_eof()
    except (ConnectionResetError, aiohttp.ClientError, asyncio.CancelledError, RuntimeError):
        status = 499; resp = web.Response(status=499)
    finally:
        inflight[rep] -= 1; t_end = time.time()
        logf.write(f"{key},{rid},{t_arr:.6f},{(t_first or 0):.6f},{t_end:.6f},{t_end:.6f},{status},{nbytes},0,{rep}\n")
    return resp

async def passthrough_get(request):
    async with session.get(B[0] + request.path) as r:
        return web.Response(status=r.status, body=await r.read(), content_type="application/json")

app = web.Application(client_max_size=10 * 1024 ** 2)
app.on_startup.append(on_start); app.on_cleanup.append(on_stop)
app.router.add_post("/v1/chat/completions", proxy)
app.router.add_get("/v1/models", passthrough_get)
web.run_app(app, port=args.port, print=None, access_log=None)
