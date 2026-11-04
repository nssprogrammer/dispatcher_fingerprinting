import numpy as np, pyarrow as pa, pyarrow.parquet as pq
rng = np.random.default_rng(0)
for k, t0 in enumerate([0, 10*86400, 20*86400]):
    n = 900_000; T = 3*86400
    s = np.sort(t0 + rng.uniform(0, T, n)); dur = rng.lognormal(1.2, 0.9, n)
    users = rng.zipf(1.6, n) % 5000
    tab = pa.table({"invocation_id": pa.array(np.arange(n), pa.int64()),
        "function_name": pa.array(np.where(rng.random(n) < 0.9, "chat", "completion")),
        "chute_id": pa.array([f"c{x}" for x in rng.integers(0, 50, n)]),
        "user_id": pa.array([f"u1_{x:07d}" for x in users]), "rehash_round": pa.array(np.full(n, k + 1, np.int32)),
        "instance_id": pa.array([f"i{x:07d}" for x in rng.integers(0, 300, n)]),
        "started_at": pa.array((s * 1e6).astype(np.int64)).cast(pa.timestamp("us")),
        "completed_at": pa.array(((s + dur) * 1e6).astype(np.int64)).cast(pa.timestamp("us")),
        "it": pa.array(rng.integers(10, 3000, n).astype(np.float64)), "ot": pa.array(rng.integers(5, 800, n).astype(np.float64)),
        "ct": pa.array(np.zeros(n)), "ttft": pa.array(rng.lognormal(-0.5, 0.5, n))})
    pq.write_table(tab, f"fake/shard_{k:02d}.parquet", row_group_size=100_000)
print("ok")
