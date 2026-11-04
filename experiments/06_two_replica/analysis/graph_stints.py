exec(open('load_balanced.py').read())
for hk, ok_ in [('hum-299','F1_rot429_shared-018'), ('hum-057','F2_rot429_worker-001')]:
    he = np.sort(E[A == kid[hk]]); os_ = np.sort(S[A == kid[ok_]])
    print(hk, '->', ok_)
    for e in he:
        x = os_[(os_ > e) & (os_ <= e + 1)]
        if len(x):
            prev = os_[(os_ > e - 5) & (os_ <= e)]
            print(f"   human end at {e-t0:7.1f}s: {len(x)} starts in next 1 s, spread {1000*(x.max()-x.min()):.0f} ms; starts in 5 s before: {len(prev)}")
# how bursty are rot429 key starts in general?
for g in ['F1_rot429_shared','F2_rot429_worker','F7_shuffle','hum']:
    gaps=[]
    for k in np.nonzero(grp==g)[0]:
        s_=np.sort(S[A==k]); gaps+=list(np.diff(s_))
    gaps=np.array(gaps); print(f"{g:18s} same-key start gaps < 50 ms: {np.mean(gaps<0.05):.2f}")
