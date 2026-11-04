"""Logging gateway with an active timing watermark.
For every request it draws a secret fair coin b and, after the model has finished, holds back the last streamed chunk
(which carries [DONE]) for b * delta seconds. It logs: key, request id, arrival, first byte, true completion (backend
finished), end (last byte sent), status, bytes, b. Only the provider knows b and the true completion time."""
import asyncio, time, argparse, random, json
import aiohttp
from aiohttp import web

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=9000)
ap.add_argument("--backend", default="http://127.0.0.1:8000")
ap.add_argument("--log", default="gateway_log.csv")
ap.add_argument("--delta", type=float, default=0.15)
ap.add_argument("--seed", type=int, default=12345)
ap.add_argument("--mode", default="holdback", choices=["holdback", "spread"])
args = ap.parse_args()
coin = random.Random(args.seed)
logf = open(args.log, "a", buffering=1)
if logf.tell() == 0:
    logf.write("key,rid,t_arrival,t_first,t_complete,t_end,status,nbytes,wm_bit\n")
session = None

async def on_start(app):
    global session
    session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0), timeout=aiohttp.ClientTimeout(total=None))

async def on_stop(app):
    await session.close(); logf.close()

async def proxy(request):
    key = request.headers.get("Authorization", "").replace("Bearer ", "").strip() or "anon"
    rid = request.headers.get("X-Req-Id", "")
    bit = coin.getrandbits(1)
    t_arr = time.time(); t_first = None; t_comp = None; nbytes = 0; status = 0; slept = 0.0
    body = await request.read()
    try:
        max_tok = max(1, int(json.loads(body).get("max_tokens") or 64))
    except Exception:
        max_tok = 64
    try:
        async with session.post(args.backend + request.path, data=body, headers={"Content-Type": "application/json"}) as r:
            status = r.status
            resp = web.StreamResponse(status=r.status, headers={"Content-Type": r.headers.get("Content-Type", "text/event-stream")})
            await resp.prepare(request)
            held = None
            async for chunk in r.content.iter_any():
                if t_first is None:
                    t_first = time.time()
                if args.mode == "spread":
                    # spread the delay over the stream: each token is held back delta / max_tokens
                    if bit and args.delta > 0:
                        dl = args.delta * chunk.count(b"data:") / max_tok
                        if dl > 0:
                            await asyncio.sleep(dl); slept += dl
                    await resp.write(chunk); nbytes += len(chunk)
                else:
                    if held is not None:
                        await resp.write(held); nbytes += len(held)
                    held = chunk
            if args.mode == "holdback":
                t_comp = time.time()
                if bit and args.delta > 0:
                    await asyncio.sleep(args.delta)
                if held is not None:
                    await resp.write(held); nbytes += len(held)
            await resp.write_eof()
            if args.mode == "spread":
                t_comp = time.time() - slept          # when the stream would have ended without the delay
    except (ConnectionResetError, aiohttp.ClientError, asyncio.CancelledError, RuntimeError):
        status = 499
        resp = web.Response(status=499)
    finally:
        t_end = time.time()
        logf.write(f"{key},{rid},{t_arr:.6f},{(t_first or 0):.6f},{(t_comp or t_end):.6f},{t_end:.6f},{status},{nbytes},{bit}\n")
    return resp

async def passthrough_get(request):
    async with session.get(args.backend + request.path) as r:
        return web.Response(status=r.status, body=await r.read(), content_type="application/json")

app = web.Application(client_max_size=10 * 1024 ** 2)
app.on_startup.append(on_start); app.on_cleanup.append(on_stop)
app.router.add_post("/v1/chat/completions", proxy)
app.router.add_get("/v1/models", passthrough_get)
web.run_app(app, port=args.port, print=None, access_log=None)
