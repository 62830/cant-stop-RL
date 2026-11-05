# figures from runs/luck/{results.json, pairs.npz} (written by analysis/luck_pilot.py); reads saved results only
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'runs', 'luck')
RAW, ADJ, INK, INK2, GRID, BG = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
Z = 2.80
plt.rcParams.update({'font.size': 11, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2, 'ytick.color': INK2,
                     'axes.spines.top': False, 'axes.spines.right': False, 'figure.facecolor': BG, 'axes.facecolor': BG,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.axisbelow': True})

def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=150, bbox_inches='tight')
    plt.close(fig)

def main():
    res, d = json.load(open(os.path.join(OUT, 'results.json'))), np.load(os.path.join(OUT, 'pairs.npz'))
    m, b = res['main'], res['main']['boot']
    y, L = d['main_result'], d['main_luck']
    raw, adj = y.mean(1), (y - L).mean(1)

    # (a) distributions: two panels on a shared x (raw is discrete, adjusted continuous)
    fig, axs = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True)
    lo, hi = min(adj.min(), 0) - 0.05, max(adj.max(), 1) + 0.05
    for ax, x, c, lab, sd in ((axs[0], raw, RAW, 'raw', m['sd_raw']), (axs[1], adj, ADJ, 'luck-adjusted', m['sd_adj'])):
        bins = np.arange(np.floor(lo / 0.02) * 0.02 - 0.01, hi + 0.02, 0.02)  # 0, 0.5, 1 at bin centers
        w, e = np.histogram(x, bins)
        ax.bar(e[:-1], w / len(x), width=(e[1] - e[0]) * 0.85, align='edge', color=c)
        ax.set_ylabel('share of pairs')
        ax.text(0.01, 0.92, f'{lab}  (SD {sd:.3f})', transform=ax.transAxes, va='top', color=INK)
    axs[1].set_xlabel('pair score of S vs V_h (mean over the two seats)')
    axs[0].set_title(f"How much does luck adjustment shrink the pair-score spread? ({m['n']} pairs)", loc='left', color=INK)
    save(fig, 'a_distributions.png')

    # (b) pairs needed vs minimum detectable edge
    mde = np.linspace(0.25, 2, 200)
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for k, c, lab in (('raw', RAW, 'raw'), ('adj', ADJ, 'luck-adjusted')):
        n = (Z * m['sd_' + k] / (mde / 100)) ** 2
        lo, hi = [(Z * s / (mde / 100)) ** 2 for s in b['sd_' + k]]
        ax.fill_between(mde, lo, hi, color=c, alpha=0.18, lw=0)
        ax.plot(mde, n, color=c, lw=2)
        n5 = res['n_needed']['0.5'][k]['n']
        ax.plot([0.5], [n5], 'o', color=c, ms=8, mec=BG, mew=2)
        ax.annotate(f'{lab}: {n5:,.0f} pairs at 0.5 pt', (0.5, n5), xytext=(10, 4), textcoords='offset points', color=INK)
    ax.axvline(0.5, color=INK2, lw=1, ls='--')
    ax.set_yscale('log')
    ax.set(xlabel='minimum detectable edge (win-rate points)', ylabel='pairs needed (80% power, α = 0.05)', xlim=(0.25, 2))
    ax.set_title('How many pairs does a gate need to detect a given edge?', loc='left', color=INK)
    ax.text(0.99, 0.97, 'bands: 95% bootstrap CI of the pair SD', transform=ax.transAxes, ha='right', va='top', color=INK2, fontsize=9)
    save(fig, 'b_pairs_needed.png')

    # (c) raw result vs sum of luck, per game
    fig, ax = plt.subplots(figsize=(8, 4.6))
    jit = np.random.default_rng(0).uniform(-0.06, 0.06, y.size)
    ax.scatter(L.ravel(), y.ravel() + jit, s=9, color=RAW, alpha=0.35, lw=0)
    ax.set(xlabel='Σ luck of S over the game (reference-solver win-prob units)', ylabel='S won (jittered)', yticks=[0, 1])
    ax.set_title(f"Does the dice luck explain who wins? (per game, r = {m['corr_game']:.2f})", loc='left', color=INK)
    save(fig, 'c_result_vs_luck.png')

if __name__ == '__main__':
    main()
