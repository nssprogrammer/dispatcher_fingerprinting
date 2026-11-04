"""Simulated OpenAI-compatible streaming server (fallback when no GPU/vLLM).
Latency model: requests beyond `capacity` wait in a FIFO queue; per-token time grows with the number of active
sequences (batching interference): tpot = base_tpot * (1 + slowdown * active / capacity)."""
import asyncio, json, time, argparse, random
from aiohttp import web

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8000)
ap.add_argument("--capacity", type=int, default=128)
ap.add_argument("--base_tpot", type=float, default=0.02)
ap.add_argument("--slowdown", type=float, default=1.0)
ap.add_argument("--ttft", type=float, default=0.08)
args = ap.parse_args()
sem = asyncio.Semaphore(args.capacity)
active = 0

async def chat(request):
    global active
    body = await request.json()
    n = int(body.get("max_tokens") or 64)
    resp = web.StreamResponse(status=200, headers={"Content-Type": "text/event-stream"})
    await resp.prepare(request)
    async with sem:
        active += 1
        try:
            await asyncio.sleep(args.ttft * random.uniform(0.8, 1.2))
            for i in range(n):
                chunk = {"id": "x", "object": "chat.completion.chunk", "created": int(time.time()), "model": body.get("model", "sim"),
                         "choices": [{"index": 0, "delta": {"content": f"w{random.randint(0, 9999)} "}, "finish_reason": None}]}
                await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
                await asyncio.sleep(args.base_tpot * (1 + args.slowdown * active / args.capacity))
            fin = {"id": "x", "object": "chat.completion.chunk", "created": int(time.time()), "model": body.get("model", "sim"),
                   "choices": [{"index": 0, "delta": {}, "finish_reason": "length"}]}
            await resp.write(f"data: {json.dumps(fin)}\n\n".encode())
            await resp.write(b"data: [DONE]\n\n")
        finally:
            active -= 1
    await resp.write_eof()
    return resp

async def models(request):
    return web.json_response({"object": "list", "data": [{"id": "sim", "object": "model"}]})

async def health(request):
    return web.Response(text="ok")

app = web.Application()
app.router.add_post("/v1/chat/completions", chat)
app.router.add_get("/v1/models", models)
app.router.add_get("/health", health)
web.run_app(app, port=args.port, print=None, access_log=None)
