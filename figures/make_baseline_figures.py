import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
exec(open('../experiments/03_baseline/analysis/load.py').read())
OUT = str(ROOT / 'figures/out') + '/'; import os; os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, AQUA, GRAY, INK, INK2 = '#2a78d6', '#eb6834', '#1baf7a', '#9a9893', '#0b0b0b', '#52514e'
plt.rcParams.update({'font.family': 'serif', 'font.size': 9.5, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6,
                     'xtick.major.width': 0.6, 'ytick.major.width': 0.6, 'legend.frameon': False})
div = LinearSegmentedColormap.from_list('div', ['#1c5cab', '#86b6ef', '#f0efec', '#f3a27d', '#c4471a'])
def pool_corr(g):
    m = (d.grp == g).values; a = d.pos.values[m].astype(int); s, e = S[m], E[m]
    X = fplib.highpass(fplib.occupancy_matrix(a, s, e, 40, t0, t1, 0.1).astype(float), 200)
    C = fplib.corr_matrix(X); np.fill_diagonal(C, np.nan); return C
rng = np.random.default_rng(4)
C1, C3 = pool_corr('P1_shuffle'), pool_corr('P3_rrshared')
perm = rng.permutation(40); C3p = C3[np.ix_(perm, perm)]
Cz = np.nan_to_num(C3p); order = fplib.circular_seriation(Cz)
fig, ax = plt.subplots(1, 3, figsize=(3.5, 1.75))
for k, (M_, t) in enumerate([(C1, '(a) LiteLLM\nshuffle'), (C3p, '(b) counter,\nkeys unordered'), (C3p[np.ix_(order, order)], '(c) same keys,\nring order')]):
    im = ax[k].imshow(M_, cmap=div, vmin=-0.6, vmax=0.6, interpolation='nearest')
    ax[k].set_title(t, fontsize=8, color=INK); ax[k].set_xticks([]); ax[k].set_yticks([])
    for sp in ax[k].spines.values(): sp.set_visible(False)
cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, ticks=[-0.5, 0, 0.5]); cb.ax.tick_params(labelsize=8); cb.outline.set_visible(False)
fig.savefig(OUT + 'fig_heatmaps.pdf', bbox_inches='tight'); plt.close(fig)

# ring profile observed vs predicted (two-box overlap model)
def overlap(tau, D1, D2): return np.clip(np.minimum(D1, tau + D2) - np.maximum(0, tau), 0, None)
fig, ax = plt.subplots(figsize=(3.4, 1.9))
for g, col, lab in [('P3_rrshared', BLUE, 'shared counter, 12 workers'), ('P8_relay_rr', ORANGE, 'round-robin relay (open loop)'), ('P5_rrjitter', AQUA, 'shared counter + 0-6 s jitter')]:
    m = (d.grp == g).values; a = d.pos.values[m].astype(int); s, e = S[m], E[m]; dur = e - s; M = 40
    lam = len(s) / (t1 - t0); Tc = M / lam; p = dur.mean() / Tc
    C = pool_corr(g); Dm = np.abs(np.subtract.outer(np.arange(M), np.arange(M))); Dm = np.minimum(Dm, M - Dm)
    prof = np.array([np.nanmean(C[Dm == k]) for k in range(1, 21)])
    o = np.argsort(s); ss, aa = s[o], a[o]; pred = []
    for k in range(1, 21):
        lags = []
        for i in range(0, len(ss), 5):
            j = np.searchsorted(ss, ss[i], 'right'); nx = np.nonzero(aa[j:j + 400] == (aa[i] + k) % M)[0]
            if len(nx): lags.append(ss[j + nx[0]] - ss[i])
        lags = np.array(lags); lags = np.where(lags > Tc / 2, lags - Tc, lags)
        both = overlap(lags, rng.choice(dur, len(lags)), rng.choice(dur, len(lags))).mean() / Tc
        pred.append((both - p * p) / (p - p * p))
    ks = np.arange(1, 21)
    ax.plot(ks, pred, color=col, lw=1.4); ax.plot(ks, prof, 'o', color=col, ms=3.2, mec='white', mew=0.5, label=lab)
ax.axhline(0, color=GRAY, lw=0.6)
ax.set_xlabel('distance on the ring $d$ (keys)'); ax.set_ylabel('correlation $c(d)$')
ax.legend(fontsize=8, loc='upper right', handletextpad=0.3, title='dots: observed   lines: predicted', title_fontsize=8)
fig.savefig(OUT + 'fig_profile.pdf', bbox_inches='tight'); plt.close(fig)
print("ok")
