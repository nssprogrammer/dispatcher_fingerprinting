"""Logging API gateway: authenticates by API key, forwards to the model server, streams the answer back,
and logs one line per request: key, arrival time, first byte, stream end, status, bytes.
This is what a provider sees: arrival -> end of stream at its own edge (queueing included)."""
import asyncio, time, argparse
import aiohttp
from aiohttp import web

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=9000)
ap.add_argument("--backend", default="http://127.0.0.1:8000")
ap.add_argument("--log", default="gateway_log.csv")
args = ap.parse_args()
logf = open(args.log, "a", buffering=1)
if logf.tell() == 0:
    logf.write("key,rid,t_arrival,t_first,t_end,status,nbytes\n")
session = None

async def on_start(app):
    global session
    session = aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0), timeout=aiohttp.ClientTimeout(total=None))

async def on_stop(app):
    await session.close(); logf.close()

async def proxy(request):
    key = request.headers.get("Authorization", "").replace("Bearer ", "").strip() or "anon"
    rid = request.headers.get("X-Req-Id", "")
    t_arr = time.time(); t_first = None; nbytes = 0; status = 0
    body = await request.read()
    try:
        async with session.post(args.backend + request.path, data=body, headers={"Content-Type": "application/json"}) as r:
            status = r.status
            resp = web.StreamResponse(status=r.status, headers={"Content-Type": r.headers.get("Content-Type", "text/event-stream")})
            await resp.prepare(request)
            async for chunk in r.content.iter_any():
                if t_first is None:
                    t_first = time.time()
                await resp.write(chunk); nbytes += len(chunk)
            await resp.write_eof()
    except (ConnectionResetError, aiohttp.ClientError, asyncio.CancelledError, RuntimeError):
        status = 499                                   # client went away or backend failed mid-stream
        resp = web.Response(status=499)
    finally:
        logf.write(f"{key},{rid},{t_arr:.6f},{(t_first or 0):.6f},{time.time():.6f},{status},{nbytes}\n")
    return resp

async def passthrough_get(request):
    async with session.get(args.backend + request.path) as r:
        return web.Response(status=r.status, body=await r.read(), content_type="application/json")

app = web.Application(client_max_size=10 * 1024 ** 2)
app.on_startup.append(on_start); app.on_cleanup.append(on_stop)
app.router.add_post("/v1/chat/completions", proxy)
app.router.add_get("/v1/models", passthrough_get)
web.run_app(app, port=args.port, print=None, access_log=None)
