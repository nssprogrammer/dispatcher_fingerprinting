# %% [code]
# Cell 2 — library
LIB_PLACEHOLDER = True

# %% [code]
# Cell 3 — download
import numpy as np, pandas as pd
import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from occlib import *
RES = {"cfg": CFG}
if CFG["data_dir"]:
    files = sorted(glob.glob(os.path.join(CFG["data_dir"], "*.parquet")))[: CFG["n_shards"]]
else:
    from huggingface_hub import HfApi, hf_hub_download
    names = sorted(f.path for f in HfApi().list_repo_tree(CFG["repo"], repo_type="dataset", recursive=True)
                   if getattr(f, "path", "").endswith(".parquet"))[: CFG["n_shards"]]
    files = [hf_hub_download(CFG["repo"], n, repo_type="dataset") for n in names]
print(len(files), "shards")

# %% [code]
# Cell 4 — parse: one row per turn, keeping each user turn's own IP and a normalised prompt prefix
def norm_prefix(content, n):
    x = pc.utf8_slice_codeunits(content, 0, 300)
    x = pc.utf8_lower(x)
    x = pc.replace_substring_regex(x, r"[0-9]+", "0")
    x = pc.replace_substring_regex(x, r"\s+", " ")
    x = pc.utf8_trim_whitespace(x)
    return pc.utf8_slice_codeunits(x, 0, n)

def parse_shard(path):
    names = pq.ParquetFile(path).schema_arrow.names
    t = pq.read_table(path, columns=[c for c in ["model", "conversation", "language", "country", "hashed_ip", "header"] if c in names])
    conv = t.column("conversation").combine_chunks()
    flat = pc.list_flatten(conv)
    parent = pc.list_parent_indices(conv).to_numpy()
    pos = np.arange(len(flat)) - conv.offsets.to_numpy()[parent]
    content = flat.field("content")
    turns = pd.DataFrame(dict(conv=parent, pos=pos, role=flat.field("role").to_numpy(zero_copy_only=False),
                              ts=pc.cast(flat.field("timestamp"), pa.int64()).to_numpy(zero_copy_only=False),
                              clen=pc.utf8_length(content).to_numpy(zero_copy_only=False),
                              tip=flat.field("hashed_ip").to_numpy(zero_copy_only=False),
                              p64=norm_prefix(content, 64).to_numpy(zero_copy_only=False)))
    hdr = t.column("header").combine_chunks()
    convs = pd.DataFrame(dict(model=t.column("model").to_numpy(), language=t.column("language").to_numpy(),
                              country=t.column("country").to_numpy(), ip=t.column("hashed_ip").to_numpy(),
                              ua=hdr.field("user-agent").to_numpy(zero_copy_only=False)))
    return turns, convs

tp, cp, base = [], [], 0
for f in files:
    tu, cv = parse_shard(f); tu["conv"] += base; cv.index = np.arange(base, base + len(cv)); base += len(cv)
    tp.append(tu); cp.append(cv)
turns = pd.concat(tp, ignore_index=True); convs = pd.concat(cp); del tp, cp
turns = turns.sort_values(["conv", "pos"], kind="stable").reset_index(drop=True)
print(len(convs), "conversations,", len(turns), "turns")

# %% [code]
# Cell 5 — requests (end = assistant timestamp; start imputed from response length, as in Notebook 2)
prev = lambda col: np.r_[[None], turns[col].values[:-1]]
a = turns[turns.role.values == "assistant"].copy()
pr, pc_, ptip, pp = prev("role")[a.index], np.r_[-1, turns.conv.values[:-1]][a.index], prev("tip")[a.index], prev("p64")[a.index]
okp = (pr == "user") & (pc_ == a.conv.values)
a["p64"] = np.where(okp, pp, None); a["tip"] = np.where(okp, ptip, None)
a["e"] = a.ts.astype(float) / 1e6
a = a[np.isfinite(a.e.values)].join(convs, on="conv")
a = a[~pd.isna(a.ip.values)]
cjk = a.language.isin(["Chinese", "Japanese", "Korean"]).values
tok = np.where(cjk, a.clen.values / 1.1, a.clen.values / 4.0)
g4 = a.model.astype(str).str.contains("gpt-4").values
a["s"] = a.e.values - np.clip(np.where(g4, 1.0, 0.8) + tok / np.where(g4, 20.0, 50.0), 0.3, CFG["max_dur_s"])
ip_codes, ip_names = pd.factorize(a.ip.values)
a["acct"] = ip_codes
req = a[["acct", "conv", "s", "e", "country", "ua", "p64", "tip", "ip"]].sort_values("s").reset_index(drop=True)
print(len(req), "requests,", req.acct.nunique(), "IP accounts")

# %% [code]
# Cell 6 — H1: identity resolution. Link IPs that appear in the same conversation; components = 'persons'
ip_index = pd.Index(ip_names)
u = turns[(turns.role.values == "user") & ~pd.isna(turns.tip.values)][["conv", "tip"]]
u = u[u.conv.isin(req.conv.unique())]
u["conv_ip"] = convs.ip.reindex(u.conv.values).values
u = u[~pd.isna(u.conv_ip.values)]
i_a = ip_index.get_indexer(u.conv_ip.values); i_b = ip_index.get_indexer(u.tip.values)
ok = (i_a >= 0) & (i_b >= 0)
multi_ip_conv = u.groupby("conv").tip.nunique()
n_ip = len(ip_index)
G = coo_matrix((np.ones(ok.sum()), (i_a[ok], i_b[ok])), shape=(n_ip, n_ip))
ncomp, comp = connected_components(G, directed=False)
csize = np.bincount(comp)
big = csize[comp] > CFG["max_component_ips"]
person = np.where(big, ncomp + np.arange(n_ip), comp)          # IPs in giant components stay separate
person = pd.factorize(person)[0]
req["person"] = person[req.acct.values]
RES["identity"] = dict(conversations_with_multiple_user_ips=round(float((multi_ip_conv > 1).mean()), 4),
                       n_ip_accounts=int(n_ip), n_persons=int(req.person.nunique()),
                       components_with_2plus_ips=int((csize >= 2).sum()), largest_component_ips=int(csize.max()),
                       ips_in_components_over_cap=int(big.sum()),
                       share_requests_from_multi_ip_persons=round(float((np.bincount(person)[req.person.values] > 1).mean()), 4))
print(json.dumps(RES["identity"], indent=1))

