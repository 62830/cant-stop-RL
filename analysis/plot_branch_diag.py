import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# runs/diag/results.json -> runs/diag/residual_by_set.png: mean |V - y| per held-out set x network, 95% bootstrap CIs
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, 'runs', 'diag')
SERIES = ['#2a78d6', '#eb6834', '#1baf7a']  # reference categorical slots 1-3
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'

def main(path=os.path.join(D, 'results.json')):
    res = json.load(open(path))
    nets, sets = list(res['nets']), ['visited', 'end', 'close']
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=150)
    w = 0.8 / len(nets)
    for k, (name, col) in enumerate(zip(nets, SERIES)):
        r = res['nets'][name]
        x = np.arange(len(sets)) + (k - (len(nets) - 1) / 2) * w
        m = np.array([r[s]['mean_abs'] for s in sets])
        lo, hi = (np.array([r[s]['ci'][j] for s in sets]) for j in (0, 1))
        ax.bar(x, m, w - 0.02, color=col, label=name, zorder=2)
        ax.errorbar(x, m, [m - lo, hi - m], fmt='none', ecolor=INK, elinewidth=1, capsize=2, zorder=3)
        for xi, s, h in zip(x, sets, hi):
            if s != 'visited':
                ax.text(xi, h, f"{r[s]['ratio']:.1f}×", ha='center', va='bottom', fontsize=7, color=MUTED)
    ax.set_xticks(range(len(sets)), [f"{s}\n(n={res['sizes'][s]})" for s in sets], color=INK)
    ax.set_ylabel('residual  mean |V(b) − best(∅|b)|', color=MUTED)
    ax.set_title('Value-net residual by held-out set (labels: ratio to visited)', color=INK, fontsize=10, loc='left')
    ax.grid(axis='y', color=GRID, zorder=0)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.spines['left'].set_color(GRID)
    ax.spines['bottom'].set_color(GRID)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(path), 'residual_by_set.png'))

if __name__ == '__main__':
    main(*sys.argv[1:])
