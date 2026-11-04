"""Local-only: write mock WildChat-1M shards with the real nested schema, to test Notebook 2 end to end.
Users: closed-loop humans keyed by hashed_ip; plus one hidden 'operator' pool of 4 slots rotating over 12 IPs
(same UA+country) so the discovery scan has something to find."""
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, os, sys
out = sys.argv[1]; with_user_ts = sys.argv[2] == "1"
os.makedirs(out, exist_ok=True)
rng = np.random.default_rng(5)
T0 = 1_680_307_200  # 2023-04-01
DAYS = 20
uas = [f"Mozilla/5.0 UA{i}" for i in range(30)]
countries = ["United States", "China", "Russia", "India", "Germany", "Brazil"]
convs = []
# humans
for u in range(1500):
    ip = f"ip{u:05d}"; ua = uas[rng.integers(30)]; ct = countries[rng.integers(6)]
    t = T0 + rng.uniform(0, DAYS * 86400)
    nconv = rng.geometric(1 / 3)
    for _ in range(nconv):
        turns = []; tt = t
        for k in range(rng.geometric(1 / 3)):
            out_chars = int(rng.lognormal(np.log(1200), 0.7))
            dur = 0.8 + out_chars / 4 / 50 * rng.lognormal(0, 0.2)
            turns.append(("user", tt, 80)); turns.append(("assistant", tt + dur, out_chars))
            tt += dur + rng.lognormal(np.log(30), 0.9)
        convs.append((ip, ua, ct, "gpt-3.5-turbo-0613", turns))
        t = tt + rng.exponential(86400 * 2)
# hidden operator: 4 closed-loop slots rotating over 12 IPs, same UA + country, for 2 days
op_ips = [f"op{i:02d}" for i in range(12)]
for w in range(4):
    t = T0 + 5 * 86400 + rng.uniform(0, 5)
    while t < T0 + 7 * 86400:
        out_chars = int(rng.lognormal(np.log(1200), 0.5)); dur = 0.8 + out_chars / 4 / 50
        ip = op_ips[rng.integers(12)]
        convs.append((ip, "python-requests/2.31", "Singapore", "gpt-3.5-turbo-0613", [("user", t, 300), ("assistant", t + dur, out_chars)]))
        t += dur + rng.lognormal(np.log(0.5), 0.5)
rng.shuffle(convs)
hdr_t = pa.struct([("accept-language", pa.string()), ("user-agent", pa.string())])
turn_t = pa.struct([("content", pa.string()), ("country", pa.string()), ("hashed_ip", pa.string()), ("header", hdr_t),
                    ("language", pa.string()), ("redacted", pa.bool_()), ("role", pa.string()), ("state", pa.string()),
                    ("timestamp", pa.timestamp("us", tz="UTC")), ("toxic", pa.bool_()), ("turn_identifier", pa.int64())])
schema = pa.schema([("conversation_hash", pa.string()), ("model", pa.string()), ("timestamp", pa.timestamp("us", tz="UTC")),
                    ("conversation", pa.list_(turn_t)), ("turn", pa.int64()), ("language", pa.string()),
                    ("toxic", pa.bool_()), ("redacted", pa.bool_()), ("state", pa.string()), ("country", pa.string()),
                    ("hashed_ip", pa.string()), ("header", hdr_t)])
shards = np.array_split(np.arange(len(convs)), 3)
tid = 0
for si, idx in enumerate(shards):
    rows = []
    for i in idx:
        ip, ua, ct, model, turns = convs[i]
        conv = []
        for role, ts, nch in turns:
            tid += 1
            ts_val = None if (role == "user" and not with_user_ts) else int(ts * 1e6)
            conv.append(dict(content="x" * nch, country=ct if role == "user" else None, hashed_ip=ip if role == "user" else None,
                             header={"accept-language": "en-US", "user-agent": ua} if role == "user" else {"accept-language": None, "user-agent": None},
                             language="English", redacted=False, role=role, state=None, timestamp=ts_val, toxic=False, turn_identifier=tid))
        rows.append(dict(conversation_hash=f"h{i}", model=model, timestamp=int(turns[-1][1] * 1e6), conversation=conv,
                         turn=len(turns) // 2, language="English", toxic=False, redacted=False, state=None, country=ct,
                         hashed_ip=ip, header={"accept-language": "en-US", "user-agent": ua}))
    tbl = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(tbl, os.path.join(out, f"train-{si:05d}-of-00003.parquet"))
print("mock conversations:", len(convs))
