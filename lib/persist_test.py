import numpy as np, pandas as pd, sys
def acf(fplib, A,S,E,N,t0,t1,dt=0.5,hp=20.0,maxlag=120.0):
    X = fplib.highpass(fplib.occupancy_matrix(A,S,E,N,t0,t1,dt).astype(np.float64), int(hp/dt))
    X = X - X.mean(1,keepdims=True); sd = X.std(1); sd[sd==0]=1; Z = X/sd[:,None]
    n = Z.shape[1]; F = np.fft.rfft(Z,2*n); return np.fft.irfft(F*np.conj(F),2*n)[:, :int(maxlag/dt)]/n
def persist(ac, dt=0.5):
    neg = ac <= 0; first = np.where(neg.any(1), neg.argmax(1), ac.shape[1])
    mask = np.arange(ac.shape[1])[None,:] < first[:,None]
    return dt*(2*(ac*mask).sum(1)-ac[:,0])
def report(fplib, A,S,E,N,t0,t1,grp,rng,label):
    ac = acf(fplib,A,S,E,N,t0,t1); p = persist(ac)
    act = np.bincount(A,minlength=N)>0
    med = np.median(p[act]); mad = 1.4826*np.median(np.abs(p[act]-med)); keep = act & (p <= med+5*mad)
    print(label, "persist median by group:", pd.Series(p).groupby(grp).median().round(2).to_dict())
    print("  trimmed:", pd.Series(grp[~keep & act]).value_counts().to_dict())
    sel = np.nonzero(keep)[0]; mp = -np.ones(N,int); mp[sel]=np.arange(len(sel)); m = keep[A]
    lab_sub = fplib.spectral_rings_multiscale(mp[A[m]], S[m], E[m], len(sel), t0, t1, rng)
    lab = -np.ones(N,int); lab[sel]=lab_sub
    for c in np.unique(lab[lab>=0]):
        u,n = np.unique(grp[lab==c], return_counts=True); print("  ring:", dict(zip(u.tolist(), n.tolist())))
