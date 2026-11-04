# %% [markdown]
# # Notebook 1 — Foundations: the occupancy library, validated
# **Goal.** Build `occlib` (exact occupancy statistics, high-pass ratio, surrogates, pool-growth score, simulators) and check it three ways before any real data touches it:
# 1. **Unit tests** against brute-force time-grid computations.
# 2. **Theory checks**: Proposition 2 (hidden pool), Proposition 3 (slot-count modulation), open-loop relays, pinned workers.
# 3. **Calibration and power** on simulated benign pools: null spread vs window, correlation time κ, power vs attacker jitter, simple baselines, hidden-worker estimate, pool-growth AUC.
#
# **Run settings.** Accelerator: *None* (CPU is enough; save your GPU quota). Internet: *On* (only used by the environment probe). Expected runtime: 15–30 min.
#
# **What to paste back.** Only the block printed by the last cell, between `===== PASTE FROM HERE =====` and `===== PASTE TO HERE =====`.

# %% [code]
# Cell 1 — environment probe (information for planning later notebooks)
import sys, os, platform, subprocess, shutil, json, time
env = {"python": sys.version.split()[0], "platform": platform.platform()}
try:
    import numpy, scipy, sklearn
    env.update(numpy=numpy.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__)
except Exception as ex:
    env["import_error"] = repr(ex)
try:
    import torch
    env["torch"] = torch.__version__
    env["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        env["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        env["gpu_capability"] = [".".join(map(str, torch.cuda.get_device_capability(i))) for i in range(torch.cuda.device_count())]
except Exception as ex:
    env["torch"] = f"not importable: {ex!r}"
env["cpu_count"] = os.cpu_count()
try:
    env["ram_gb"] = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9, 1)
except Exception:
    pass
env["disk_free_gb_working"] = round(shutil.disk_usage("/kaggle/working" if os.path.exists("/kaggle/working") else ".").free / 1e9, 1)
try:
    import urllib.request
    urllib.request.urlopen("https://huggingface.co", timeout=10)
    env["internet"] = True
except Exception as ex:
    env["internet"] = f"no ({type(ex).__name__})"
try:
    from kaggle_secrets import UserSecretsClient
    try:
        UserSecretsClient().get_secret("HF_TOKEN"); env["hf_token_secret"] = True
    except Exception:
        env["hf_token_secret"] = False
except Exception:
    env["hf_token_secret"] = "not on kaggle"
print(json.dumps(env, indent=1))
T_START = time.time()

# %% [code]
# Cell 1b — data-access probe for Notebook 2 (WildChat: real human request timing). Downloads ~one shard at most.
probe = {}
try:
    from huggingface_hub import HfApi, hf_hub_download
    tok = None
    try:
        from kaggle_secrets import UserSecretsClient
        tok = UserSecretsClient().get_secret("HF_TOKEN")
    except Exception:
        pass
    api = HfApi()
    for repo in ["allenai/WildChat-1M", "allenai/WildChat-4.8M", "allenai/WildChat"]:
        try:
            files = [f for f in api.list_repo_tree(repo, repo_type="dataset", recursive=True, token=tok)
                     if getattr(f, "path", "").endswith(".parquet")]
            probe[repo] = dict(n_parquet=len(files), total_gb=round(sum((f.size or 0) for f in files) / 1e9, 2),
                               first=files[0].path if files else None)
        except Exception as ex:
            probe[repo] = f"error: {type(ex).__name__}: {str(ex)[:120]}"
    # try one shard of WildChat-1M and report its schema
    r = probe.get("allenai/WildChat-1M")
    if isinstance(r, dict) and r["first"]:
        import pyarrow.parquet as pq
        p = hf_hub_download("allenai/WildChat-1M", r["first"], repo_type="dataset", token=tok)
        pf = pq.ParquetFile(p)
        probe["wildchat1m_shard_rows"] = pf.metadata.num_rows
        probe["wildchat1m_schema"] = str(pf.schema_arrow)[:1500]
except Exception as ex:
    probe["error"] = f"{type(ex).__name__}: {str(ex)[:200]}"
print(json.dumps(probe, indent=1)[:4000])
env["data_probe"] = {k: (v if k != "wildchat1m_schema" else v[:800]) for k, v in probe.items()}

# %% [code]
# Cell 2 — write the library to disk (later notebooks reuse this exact file)
LIB = r'''
__LIBRARY__
'''
with open("occlib.py", "w") as f:
    f.write(LIB)
import importlib, occlib
importlib.reload(occlib)
print("occlib", occlib.__version__, "written to", os.path.abspath("occlib.py"))

# %% [code]
# Cell 3 — unit tests against brute force on a fine time grid
import numpy as np
from occlib import *
RES = {"env": env}
rng = np.random.default_rng(0)

def brute(acct, s, e, t0, t1, tau=None, dt=0.002):
    tau = (t1 - t0) if tau is None else tau
    nb = int(np.floor((t1 - t0) / tau + 1e-9)); t1 = t0 + nb * tau
    grid = np.arange(t0, t1, dt) + dt / 2
    A = acct.max() + 1
    X = np.zeros((A, len(grid)))
    for a, ss, ee in zip(acct, s, e):
        X[a] += (grid >= ss) & (grid < ee)
    N = X.sum(0)
    blk = ((grid - t0) // tau).astype(int)
    def wv(y):
        return np.mean([y[blk == b].var() for b in range(nb)])
    VN = wv(N); Vi = np.array([wv(X[a]) for a in range(A)])
    return VN / Vi.sum(), VN, Vi

tests = {}
# random small log with overlapping requests inside accounts
A = 7; n = 120
acct = rng.integers(0, A, n); s = rng.uniform(0, 300, n); e = s + rng.lognormal(1.5, 0.8, n)
for tau in [None, 37.0, 60.0]:
    ex = occupancy(acct, s, e, 10.0, 290.0, tau)
    rb, vnb, vib = brute(acct, s, e, 10.0, 290.0, tau)
    tests[f"exact_vs_grid_tau={tau}"] = float(max(abs(ex["R"] - rb), np.max(np.abs(ex["V_i"] - vib)), abs(ex["V_N"] - vnb)))
# identity: tau = whole window equals full-window
tests["Rtau_equals_R_when_tau_is_window"] = float(abs(R_hp(acct, s, e, 10, 290, 280.0) - R_full(acct, s, e, 10, 290)))
# surrogate keeps per-account variances
a2, s2, e2 = circular_shift(acct, s, e, 10, 290, rng)
tests["surrogate_keeps_V_i"] = float(np.max(np.abs(occupancy(a2, s2, e2, 10, 290)["V_i"] - occupancy(acct, s, e, 10, 290)["V_i"])))
# pool correlation against brute force
pool = (acct[acct < 4], s[acct < 4], e[acct < 4])
cm = acct >= 4
cand = (acct[cm] - 4, s[cm], e[cm])
c_ex = corr_with_pool(pool, cand, 10.0, 290.0)
grid = np.arange(10, 290, 0.002) + 0.001
Np = sum(((grid >= a) & (grid < b)).astype(float) for a, b in zip(*pool[1:]))
c_br = []
for j in range(3):
    m = cand[0] == j
    Xj = sum(((grid >= a) & (grid < b)).astype(float) for a, b in zip(cand[1][m], cand[2][m]))
    c_br.append(np.corrcoef(Xj, Np)[0, 1])
tests["corr_with_pool_vs_grid"] = float(np.max(np.abs(c_ex - np.array(c_br))))
RES["unit_tests_max_abs_error"] = tests
print(json.dumps(tests, indent=1))
assert all(v < 5e-3 for v in tests.values()), "unit test failed"
print("all unit tests passed")

# %% [code]
# Cell 4 — theory checks (Proposition 2, open loop, pinning, Proposition 3 modulation)
T0, T6 = 300.0, 6 * 3600.0
rows = []
for K in [2, 4, 8, 16, 34]:
    for J in [0, 10, 30, 60, 120]:
        for M in [10, 40]:
            Rs = [R_full(*closed_loop_pool(rng, T6, K, M, J), T0, T6) for _ in range(4)]
            rows.append(dict(K=K, J=J, M=M, R_sim=float(np.mean(Rs)), R_th=float(theory_R(duty(J), M))))
err = [abs(r["R_sim"] - r["R_th"]) for r in rows]
RES["prop2"] = dict(n_settings=len(rows), max_abs_err=round(max(err), 4), mean_abs_err=round(float(np.mean(err)), 4))
print("Prop 2:", RES["prop2"])

edge = {}
edge["pinned_40_slots_40_accts_R"] = float(np.mean([R_full(*closed_loop_pool(rng, T6, 40, 40, 0, assign="pinned"), T0, T6) for _ in range(2)]))
for pol in ["random", "rr", "least"]:
    edge[f"open_loop_relay_{pol}_R"] = float(np.mean([R_full(*open_loop_relay(rng, T6, 40, 0.2, pol), T0, T6) for _ in range(2)]))
RES["edge_cases"] = {k: round(v, 3) for k, v in edge.items()}
print("edge cases:", RES["edge_cases"])

mod = []
for label, kw in [("none", dict(K=10)), ("poisson10_every300s", dict(K=40, K_bar=10, epoch=300)),
                  ("poisson10_every60s", dict(K=40, K_bar=10, epoch=60)), ("poisson10_every20s", dict(K=40, K_bar=10, epoch=20)),
                  ("every300s_plus_30min_stints", dict(K=40, K_bar=10, epoch=300, assign="stint", stint=1800))]:
    v = []
    for _ in range(4):
        log = closed_loop_pool(rng, T6, M=40, **kw)
        v.append([R_full(*log, T0, T6), R_hp(*log, T0, T6, 60.0), R_hp(*log, T0, T6, 20.0)])
    v = np.mean(v, 0)
    mod.append(dict(case=label, R=round(float(v[0]), 3), R60=round(float(v[1]), 3), R20=round(float(v[2]), 3)))
RES["modulation"] = mod
print("Prop 3 / modulation:"); [print(" ", m) for m in mod]

# %% [code]
# Cell 5 — null calibration on simulated benign pools (40 accounts)
null = {}
for hours, npools in [(1, 400), (6, 400), (24, 120)]:
    T = hours * 3600.0
    R1, R2 = [], []
    for _ in range(npools):
        log = benign_humans(rng, 40, T)
        R1.append(R_full(*log, T0, T)); R2.append(R_hp(*log, T0, T, 60.0))
    R1, R2 = np.array(R1), np.array(R2)
    null[f"{hours}h"] = dict(n=npools, R_mean=round(float(R1.mean()), 4), R_sd=round(float(R1.std()), 4),
                             R60_mean=round(float(R2.mean()), 4), R60_sd=round(float(R2.std()), 4),
                             R60_q01=round(float(np.quantile(R2, 0.01)), 4))
L = np.array([h * 3600 - T0 for h in (1, 6, 24)])
null["kappa_R_s"] = round(float(np.mean([null[f"{h}h"]["R_sd"] ** 2 * l / 2 for h, l in zip((1, 6, 24), L)])), 2)
null["kappa_R60_s"] = round(float(np.mean([null[f"{h}h"]["R60_sd"] ** 2 * l / 2 for h, l in zip((1, 6, 24), L)])), 2)
# surrogate null vs reference null on single pools (6 h)
sur = []
for _ in range(5):
    log = benign_humans(rng, 40, T6)
    sur.append(float(surrogate_null(*log, T0, T6, 60.0, 40, rng).std()))
null["surrogate_sd_R60_6h_mean"] = round(float(np.mean(sur)), 4)
# common-mode daily cycle
Rd = [R_full(*benign_humans(rng, 40, 86400.0, diurnal=0.8), T0, 86400.0) for _ in range(20)]
null["diurnal_24h_R_mean"] = round(float(np.mean(Rd)), 3)
RES["null"] = null
print(json.dumps(null, indent=1))

# %% [code]
# Cell 6 — power vs attacker jitter at fixed throughput, baselines, hidden-worker estimate, pool growth
from sklearn.metrics import roc_auc_score
power = []
for hours in [1, 6]:
    T = hours * 3600.0; nn = null[f"{hours}h"]
    for J, K in [(0, 4), (10, 6), (30, 10), (60, 16), (120, 28), (240, 52)]:
        v = []; thr = []
        for _ in range(6):
            log = closed_loop_pool(rng, T, K, 40, J)
            v.append(R_hp(*log, T0, T, 60.0)); thr.append(baselines(*log, T0, T)["req_per_h"])
        power.append(dict(h=hours, J=J, K=K, req_h=int(np.mean(thr)), R60=round(float(np.mean(v)), 3),
                          z=round(float((nn["R60_mean"] - np.mean(v)) / nn["R60_sd"]), 1)))
RES["power"] = power
print("power:"); [print(" ", p) for p in power]

def heavy_benign():
    return concat(benign_humans(rng, 36, T6), closed_loop_pool(rng, T6, 4, 4, 0, assign="pinned", offset=36))
cases = [("benign_40_humans", lambda: benign_humans(rng, 40, T6)),
         ("benign_36_humans_plus_4_batch_clients", heavy_benign),
         ("attack_4_slots_40_accts", lambda: closed_loop_pool(rng, T6, 4, 40, 0)),
         ("attack_10_slots_J30", lambda: closed_loop_pool(rng, T6, 10, 40, 30)),
         ("legit_company_4_slots_40_keys", lambda: closed_loop_pool(rng, T6, 4, 40, 0))]
base = []
for name, gen in cases:
    log = gen(); b = baselines(*log, T0, T6)
    base.append(dict(pool=name, req_h=int(b["req_per_h"]), peak=int(b["peak_N"]), fano=round(float(b["fano"]), 3),
                     R=round(float(R_full(*log, T0, T6)), 3), R60=round(float(R_hp(*log, T0, T6, 60.0)), 3)))
RES["baselines"] = base
print("baselines:"); [print(" ", b) for b in base]

kh = []
for J, K in [(0, 4), (10, 6), (30, 10), (60, 18)]:
    kk = [fano_khat(*closed_loop_pool(rng, T6, K, 40, J), T0, T6)["K_hat"] for _ in range(3)]
    kh.append(dict(K=K, J=J, K_hat=round(float(np.mean(kk)), 2)))
RES["K_hat"] = kh
print("K_hat:", kh)

growth = []
for J, K in [(0, 4), (30, 10), (120, 28)]:
    aucs = []
    for _ in range(6):
        att = closed_loop_pool(rng, T6, K, 40, J)
        ben = benign_humans(rng, 220, T6, offset=40)
        log = concat(att, ben)
        a = log[0]
        seed_ids = np.r_[0:20, 40:60]                         # half the campaign + 20 benign wrongly linked
        cand_ids = np.r_[20:40, 60:260]
        ms = np.isin(a, seed_ids); mc = np.isin(a, cand_ids)
        remap = -np.ones(260, int); remap[cand_ids] = np.arange(len(cand_ids))
        sc = corr_with_pool((a[ms], log[1][ms], log[2][ms]), (remap[a[mc]], log[1][mc], log[2][mc]), T0, T6, tau=60.0, n_cand=len(cand_ids))
        lab = np.r_[np.ones(20), np.zeros(200)]
        sc = np.nan_to_num(-sc, nan=0.0)
        aucs.append(roc_auc_score(lab, sc))
    growth.append(dict(J=J, K=K, auc=round(float(np.mean(aucs)), 3)))
RES["pool_growth_auc"] = growth
print("pool growth:", growth)

# %% [code]
# Cell 7 — summary to paste back
RES["runtime_min"] = round((time.time() - T_START) / 60, 1)
with open("nb1_results.json", "w") as f:
    json.dump(RES, f, indent=1)
print("===== PASTE FROM HERE =====")
print(json.dumps(RES, separators=(",", ":")))
print("===== PASTE TO HERE =====")
