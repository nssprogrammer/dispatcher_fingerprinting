import sys, json, numpy as np, pandas as pd
import sys as _s, pathlib as _p; ROOT = next(q for q in _p.Path(__file__).resolve().parents if (q / 'lib' / 'fplib.py').exists()); _s.path.insert(0, str(ROOT / 'lib'))
import fplib, replica_an as nb9an
cfg=json.load(open('run_A/driver_cfg.json')); d,labels,cl,t0,t1,ls,r429=nb9an.load('run_A',cfg)
allk=sorted(d.key.unique()); idx={k:i for i,k in enumerate(allk)}; A=d.key.map(idx).values.astype(np.int64); N=len(allk)
grp=np.array([labels[k]["pop"] if labels[k]["pop"] not in ("hum_a","hum_b") else "hum" for k in allk])
lab,kept=fplib.site_handoff_graph_sideband(A,d.t_arrival.values,d.t_end.values,N,t0,t1)
print("edges",len(kept))
for c in np.unique(lab[lab>=0]):
    u,n=np.unique(grp[lab==c],return_counts=True); comp=dict(zip(u.tolist(),n.tolist()))
    if 'hum' in comp: print(comp)
he=[(grp[i],grp[j]) for (i,j,*rest) in kept if 'hum' in (grp[i],grp[j])]
print(pd.Series([tuple(sorted(x)) for x in he]).value_counts())
S=d.t_arrival.values; E=d.t_end.values
for (i,j) in [x[:2] for x in kept if 'hum' in (grp[x[0]],grp[x[1]])][:5]:
    for (u,v) in [(i,j),(j,i)]:
        eu=np.sort(E[A==u]); sv=np.sort(S[A==v])
        lags=[]
        for x in eu:
            k=np.searchsorted(sv,x); 
            if k<len(sv) and sv[k]-x<1: lags.append(round(sv[k]-x,3))
        print(grp[u],'->',grp[v], 'n_end',len(eu),'n_start',len(sv),'handoffs<1s',len(lags), lags[:8])
    hk = allk[i] if grp[i]=='hum' else allk[j]
    x=d[d.key==hk]; print('  human key',hk,'requests',len(x),'429s',int((r429.key==hk).sum()), 'durations', x.dur.round(1).tolist()[:6] if 'dur' in x else '')
