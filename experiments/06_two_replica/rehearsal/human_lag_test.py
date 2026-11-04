import sys, json, numpy as np, pandas as pd
import sys as _s, pathlib as _p; ROOT = next(q for q in _p.Path(__file__).resolve().parents if (q / 'lib' / 'fplib.py').exists()); _s.path.insert(0, str(ROOT / 'lib'))
import replica_an as nb9an
cfg=json.load(open('run_A/driver_cfg.json')); d,labels,cl,t0,t1,ls,r429=nb9an.load('run_A',cfg)
d['g2']=d.grp.where(~d.grp.isin(['hum_a','hum_b']),'hum')
ends_by={g:np.sort(x.t_end.values) for g,x in d.groupby('g2')}
allend=np.sort(np.r_[d.t_end.values, r429.t_end.values])
hs=np.sort(d[d.g2=='hum'].t_arrival.values); rnd=np.sort(np.random.default_rng(0).uniform(t0,t1,20000))
def lagdist(x, ends):
    k=np.searchsorted(ends,x)-1; ok=k>=0; return x[ok]-ends[k[ok]]
for g in ['F1_rot429_shared','F2_rot429_worker','F7_shuffle','F6_rrshared']:
    lh=lagdist(hs,ends_by[g]); lr=lagdist(rnd,ends_by[g])
    print(f"{g:18s} P(lag<0.05) human {np.mean(lh<0.05):.3f} random {np.mean(lr<0.05):.3f} | P(lag<1) human {np.mean(lh<1):.3f} random {np.mean(lr<1):.3f}")
# inter-arrival gaps at gateway, all requests: excess of exact ties?
st=np.sort(np.r_[d.t_arrival.values, r429.t_arrival.values]); g=np.diff(st)
print("gateway arrival gaps: frac < 1ms", np.mean(g<1e-3).round(4), "< 5ms", np.mean(g<5e-3).round(4), "rate/s", len(st)/(t1-t0))
# client vs gateway delay for humans if client logs exist
import glob; print(glob.glob('run_A/clientlog_hum*')[:3])
