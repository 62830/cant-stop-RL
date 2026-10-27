import sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
S = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']
OPP = {'prev': ('vs previous snapshot', S[0]), 'vh': ('vs best V_h solver', S[1]),
       'v2tuned': ('vs V2Tuned (report)', S[2]), 'v2': ('vs v2 (held out)', S[3])}

def style(ax, title, xlabel):
    ax.set_facecolor(SURF)
    ax.set_title(title, loc='left', color=INK, fontsize=11)
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.grid(color=GRID, lw=0.8)
    ax.tick_params(colors=INK2, labelsize=8, length=0)
    for s in ax.spines.values():
        s.set_visible(False)

def plain(ax):
    for f in (ax.yaxis.set_major_formatter, ax.yaxis.set_minor_formatter):
        f(FormatStrFormatter('%.3g'))

def smooth(y, k):
    return np.convolve(y, np.ones(k) / k, 'valid') if len(y) >= k else y

def main(run='phase4', out=None):
    path = f'runs/{run}/metrics.jsonl'
    R = [json.loads(l) for l in open(path)]
    by = lambda k: [r for r in R if r['kind'] == k]
    it, up, cal, lad = by('iter'), by('update'), by('calib'), by('ladder')
    fig, axs = plt.subplots(2, 3, figsize=(16, 9), facecolor=SURF)
    fig.suptitle(f'TD training dashboard: runs/{run}', x=0.01, ha='left', color=INK, fontsize=14)

    ax = axs[0, 0]
    style(ax, 'Training loss (BCE vs soft solver targets)', 'update step')
    if up:
        loss = np.array([r['loss'] for r in up])
        ax.plot(loss, color=GRID, lw=1)
        k = 50
        ax.plot(np.arange(len(smooth(loss, k))) + k - 1, smooth(loss, k), color=S[0], lw=2, label=f'{k}-step mean')
        ax.set_yscale('log')
        plain(ax)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2)

    ax = axs[0, 1]
    style(ax, 'Bellman residual mean |V(s) - y(s)| (held-out boards)', 'iteration')
    rs = [(r['iter'], r['residual']) for r in it if r.get('residual') is not None]
    if rs:
        x, y = zip(*rs)
        ax.plot(x, y, color=S[0], lw=2, marker='o', ms=4)
        ax.set_yscale('log')
        plain(ax)
        ax.annotate(f'{y[-1]:.3f}', (x[-1], y[-1]), xytext=(4, 4), textcoords='offset points', color=INK, fontsize=8)

    ax = axs[0, 2]
    style(ax, 'Ladder win rate (95% Wilson CI)', 'snapshot iteration')
    ax.axhline(0.5, color=INK2, lw=1, ls='--')
    for o, (name, c) in OPP.items():
        rr = sorted((r for r in lad if r['opp'] == o), key=lambda r: r['iter'])
        if rr:
            x = [r['iter'] for r in rr]
            ax.fill_between(x, [r['lo'] for r in rr], [r['hi'] for r in rr], color=c, alpha=0.15, lw=0)
            ax.plot(x, [r['p'] for r in rr], color=c, lw=2, marker='o', ms=5, label=name)
    gate = [r for r in lad if r['opp'].startswith('gate_')]
    for r in gate:
        name, c = OPP[r['opp'][5:]]
        ax.errorbar(r['iter'] + 1.5, r['p'], [[r['p'] - r['lo']], [r['hi'] - r['p']]], color=c, marker='D', ms=7, capsize=3,
                    mec=SURF, mew=1.5, ls='none')
    ax.set_ylim(0.2, 1.0)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc='lower left', title='diamonds: final gate, n=1000' if gate else None,
              title_fontsize=8)

    ax = axs[1, 0]
    style(ax, 'Reliability at turn starts vs self-play outcomes', 'predicted P(to-move wins)')
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls='--')
    if cal:
        first, last = cal[0], cal[-1]
        for b, label, c in ((first['warm'], f"warm start (ECE {first['ece_warm']:.3f})", S[1]),
                            (last['net'], f"iter {last['iter']} snapshots (ECE {last['ece_net']:.3f})", S[0])):
            b = np.array(b)
            ax.plot(b[:, 1], b[:, 2], color=c, lw=2, marker='o', ms=5, label=label)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc='upper left')
    ax.set_ylabel('observed win rate', color=INK2, fontsize=9)

    ax = axs[1, 1]
    style(ax, 'Self-play games/s (8 workers)', 'iteration')
    if it:
        ax.plot([r['iter'] for r in it], [r['games_per_s'] for r in it], color=S[0], lw=2)
        ax.set_ylim(bottom=0)

    ax = axs[1, 2]
    style(ax, 'Mean solve time per turn start (s)', 'iteration')
    if it:
        ax.plot([r['iter'] for r in it], [r['solve_s'] for r in it], color=S[0], lw=2)
        ax.set_ylim(bottom=0)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out or f'runs/{run}/dashboard.png', dpi=110, facecolor=SURF)

if __name__ == '__main__':
    main(*sys.argv[1:])
