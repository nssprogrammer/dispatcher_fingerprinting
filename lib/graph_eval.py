import numpy as np, pandas as pd
def evalg(fplib, A,S,E,N,t0,t1,grp,label):
    import functools
    for name,fn in [("v0.4",fplib.site_handoff_graph_sideband),("v0.8",fplib.site_handoff_graph_coverage),("v0.9",fplib.site_handoff_graph_v09)]:
        lab,kept=fn(A,S,E,N,t0,t1)
        rec={}
        for g in sorted(set(grp)):
            if g in ('hum','bat'): continue
            mem=(grp==g); best=0
            for c in np.unique(lab[lab>=0]): best=max(best,int(((lab==c)&mem).sum()))
            if best: rec[g]=round(best/mem.sum(),2)
        print(f"{label} {name:9s} edges {len(kept):4d} humans {int(((lab>=0)&(grp=='hum')).sum())} recall {rec}", flush=True)
