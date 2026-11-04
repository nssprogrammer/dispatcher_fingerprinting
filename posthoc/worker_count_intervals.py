import numpy as np, pandas as pd, sys
from scipy import stats
import sys as _s, pathlib as _p; ROOT = next(q for q in _p.Path(__file__).resolve().parents if (q / 'lib' / 'fplib.py').exists()); _s.path.insert(0, str(ROOT / 'lib'))
import fplib
def ci(d, t0, t1, nb=6):
    edges = np.linspace(t0, t1, nb+1); ks=[]
    for a,b in zip(edges[:-1], edges[1:]):
        x = d[(d.t_arrival>=a)&(d.t_arrival<b)]
        ks.append(fplib.K_little(x.t_arrival.values, x.t_end.values, a, b)[0])
    ks=np.array(ks); full=fplib.K_little(d.t_arrival.values,d.t_end.values,t0,t1)[0]
    h = stats.t.ppf(0.975, nb-1)*ks.std(ddof=1)/np.sqrt(nb)
    return full, full-h, full+h
def run(name, d, t0, t1, pools):
    for g,K in pools:
        x=d[d.grp==g]; f,lo,hi=ci(x,t0,t1)
        print(f"{name} {g:18s} K={K:3d} Khat={f:5.2f} 95%CI [{lo:5.2f},{hi:5.2f}] covers={lo<=K<=hi}")
d7=pd.read_csv(ROOT / 'data/live/repeat_requests.csv.gz'); d7=d7[d7.status==200] if 'status' in d7 else d7
T=1792959464.535086; t0=T+180; t1=t0+2190; d7=d7[(d7.t_end>t0)&(d7.t_arrival<t1)]
print(sorted(d7.grp.unique()))
run('NB7',d7,t0,t1,[('P1_shuffle',6),('P2_least',10),('P3_rrshared',12),('P4_rrworker',8),('P5_rr_jitter',8),('C_company',4),('P9_pinned',8)])
dA=pd.read_csv(ROOT / 'data/live/stress_requests.csv.gz'); dA=dA[dA.status==200]; dA['grp']=dA.key.str.rsplit('-',n=1).str[0]
T0=dA.t_arrival.min(); a0=T0+240; a1=T0+3600-30; dA=dA[(dA.t_end>a0)&(dA.t_arrival<a1)]
print(sorted(dA.grp.unique()))
P=[('P1_shuffle',6),('P2_least',10),('P3_rrshared',12),('P4_rrworker',8),('P5_rrjitter',8),('C_company',4),('P9_pinned',8),('P6_agents',6)]
print('---')
run('NB7',d7,t0,t1,P[4:])
run('NB8A',dA,a0,a1,P)
