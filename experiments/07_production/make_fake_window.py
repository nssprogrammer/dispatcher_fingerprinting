import numpy as np, pandas as pd
rng = np.random.default_rng(5); W = 3600; rows = []
NH, NB = 2600, 400
for u in range(NH):                       # humans: sessions with think times
    t = rng.uniform(-1800, W)
    for _ in range(rng.geometric(0.15)):
        d = rng.lognormal(1.6, 1.0); rows.append((u, t, t + d)); t += d + rng.lognormal(np.log(40), 1.0)
        if t > W: break
for b in range(NB):                       # bots: independent Poisson streams, some concurrent
    lam = rng.lognormal(np.log(0.05), 1.0); t = -1800
    while t < W:
        t += rng.exponential(1 / lam); d = rng.lognormal(1.8, 1.1); rows.append((NH + b, t, t + d))
r = np.array(rows); n = len(r)
inst = np.where(rng.random(n) < 0.5, (r[:, 0] * 7919) % 120, rng.integers(0, 120, n))   # half of traffic pinned to a home instance
df = pd.DataFrame({"started": r[:, 1].astype(np.float32), "completed": r[:, 2].astype(np.float32), "acct": r[:, 0].astype(np.int32),
                   "inst": inst.astype(np.int32), "chute": rng.integers(0, 40, n).astype(np.int32),
                   "ttft": rng.lognormal(-0.5, 0.5, n).astype(np.float32), "ot": rng.integers(5, 2000, n).astype(np.float32)})
df.sort_values("started").to_parquet("fake/fake_window.parquet", compression="zstd", index=False)
print(len(df), "rows")
