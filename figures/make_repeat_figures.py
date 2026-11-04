import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
exec(open('../experiments/04_repeat/analysis/load7.py').read())
OUT = str(ROOT / 'figures/out') + '/'; import os; os.makedirs(OUT, exist_ok=True)
BLUE, ORANGE, AQUA, GRAY, INK, INK2 = '#2a78d6', '#eb6834', '#1baf7a', '#9a9893', '#0b0b0b', '#52514e'
plt.rcParams.update({'font.family': 'serif', 'font.size': 9.5, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6, 'legend.frameon': False})
rng = np.random.default_rng(0); N = len(keys)
C = np.load(ROOT / 'experiments/04_repeat/analysis/C7.npy'); w = np.linalg.eigvalsh(C)
Cn = fplib.hollow_corr(*fplib.shift_keys(A, S, E, t0, t1, rng, N), N, t0, t1); wn = np.linalg.eigvalsh(Cn)
fig, ax = plt.subplots(figsize=(3.4, 1.8))
bins = np.linspace(-2.5, 8.5, 90)
ax.hist(wn, bins=bins, color=GRAY, alpha=0.55, label='time-shifted null', log=True)
ax.hist(w, bins=bins, histtype='step', color=BLUE, lw=1.0, label='observed (Repeat run, 703 keys)', log=True)
edge = 1.05 * wn[-1]; ax.axvline(edge, color=INK2, lw=0.6, ls='--'); ax.text(edge - 0.1, 150, 'bulk edge', fontsize=8, color=INK2, ha='right')
for x, xt, yt, lab in [(7.8, 7.8, 30, 'shared\ncounter'), (5.75, 5.75, 6, 'relay'), (3.6, 3.6, 30, 'counter\n+ jitter')]:
    ax.annotate(lab, xy=(x, 1.6), xytext=(xt, yt), fontsize=8, color=INK, ha='center', arrowprops=dict(arrowstyle='-', color=INK2, lw=0.5))
ax.set_xlabel('eigenvalue of the hollowed correlation matrix'); ax.set_ylabel('count'); ax.set_ylim(0.7, 400)
ax.legend(fontsize=8, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2)
fig.savefig(OUT + 'fig_spectrum.pdf', bbox_inches='tight'); plt.close(fig)
# watermark predicted vs observed
fig, ax = plt.subplots(figsize=(1.9, 1.8))
pred = [57.0, 32.5, 45.6, 72.7]; obs = [65.9, 35.2, 48.1, 77.5]; labs = ['shuffle', 'shared ctr', 'per-worker', 'company']
ax.plot([0, 90], [0, 90], color=GRAY, lw=0.6)
ax.plot(pred, obs, 'o', color=BLUE, ms=4.5, mec='white', mew=0.6)
for x, y, l in zip(pred, obs, labs): ax.text(x + 2, y - 5, l, fontsize=8, color=INK2)
ax.set_xlim(20, 90); ax.set_ylim(20, 90); ax.set_xlabel('predicted echo (ms)'); ax.set_ylabel('observed echo (ms)')
fig.savefig(OUT + 'fig_watermark.pdf', bbox_inches='tight'); plt.close(fig)
print('outliers', np.round(w[-10:][::-1], 2), 'edge', round(edge, 2))
