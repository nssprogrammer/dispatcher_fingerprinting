import sys, json, numpy as np, pandas as pd
import sys as _s, pathlib as _p; ROOT = next(q for q in _p.Path(__file__).resolve().parents if (q / 'lib' / 'fplib.py').exists()); _s.path.insert(0, str(ROOT / 'lib'))
import replica_an as nb9an
cfg=json.load(open('run_A/driver_cfg.json')); d,labels,cl,t0,t1,ls,r429=nb9an.load('run_A',cfg)
h=d[d.key=='hum-050'].sort_values('t_arrival')
f=d[d.grp=='F1_rot429_shared']
for _,r in h.iterrows():
    w=f[(f.t_end>r.t_arrival-3)&(f.t_end<r.t_arrival+1)]
    w2=f[(f.t_arrival>r.t_arrival-3)&(f.t_arrival<r.t_arrival+3)]
    print(round(r.t_arrival-t0,2), 'F1 ends in [-3,+1]:', w.key.value_counts().to_dict(), '| F1 starts +-3s:', len(w2), '| rows F1 in run:', )
print(f.key.value_counts().describe())
g=f.sort_values('t_arrival'); ch=(g.key!=g.key.shift()).sum(); print('F1 key changes', ch, 'of', len(g))
print(r429[r429.key.str.startswith('F1')].shape)
