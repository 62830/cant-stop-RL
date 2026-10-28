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
    # trailing k-mean, with x aligned to each window's last step
    k = min(k, len(y))
    return np.arange(k - 1, len(y)), np.convolve(y, np.ones(k) / k, 'valid')

def mark_resumes(ax, xs):
    for x in xs:
        ax.axvline(x, color=INK2, lw=1, ls=':')

def main(run='phase4', out=None):
    path = f'runs/{run}/metrics.jsonl'
    R = [json.loads(l) for l in open(path)]
    by = lambda k: [r for r in R if r['kind'] == k]
    it, up, cal, lad = by('iter'), by('update'), by('calib'), by('ladder')
    resumes = [r['iter'] for r in by('resume')]
    fig, axs = plt.subplots(3, 3, figsize=(16, 13.5), facecolor=SURF)
    fig.suptitle(f'TD training dashboard: runs/{run}' + (f' (dotted: resumed at iter {resumes})' if resumes else ''), x=0.01, ha='left', color=INK, fontsize=14)

    ax = axs[0, 0]
    style(ax, 'Training loss (BCE vs soft solver targets)', 'update step')
    if up:
        # update records: one per step (loss float) or one per iteration (loss list)
        per = [r['loss'] if isinstance(r['loss'], list) else [r['loss']] for r in up]
        loss = np.concatenate(per)
        start = np.cumsum([0] + [len(l) for l in per])
        ax.plot(loss, color=GRID, lw=1)
        k = 50
        ax.plot(*smooth(loss, k), color=S[0], lw=2, label=f'{k}-step mean')
        first = {}
        for r, s0 in zip(up, start):
            first.setdefault(r['iter'], s0)
        mark_resumes(ax, [first[i] for i in resumes if i in first])
        ax.set_yscale('log')
        plain(ax)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2)

    ax = axs[0, 1]
    style(ax, 'Bellman residual mean |V(s) - y(s)| (held-out boards)', 'iteration')
    rs = [(r['iter'], r['residual']) for r in it if r.get('residual') is not None]
    if rs:
        x, y = zip(*rs)
        ax.plot(x, y, color=S[0], lw=2, marker='o', ms=4)
        mark_resumes(ax, resumes)
        ax.set_yscale('log')
        plain(ax)
        ax.annotate(f'{y[-1]:.3f}', (x[-1], y[-1]), xytext=(4, 4), textcoords='offset points', color=INK, fontsize=8)

    ax = axs[0, 2]
    style(ax, 'Ladder win rate (95% Wilson CI)', 'snapshot iteration')
    mark_resumes(ax, resumes)
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
    gate_n = '/'.join(f"{n}{' pairs' if pr else ''}" for n, pr in sorted({(r['n'], r.get('paired', False)) for r in gate}, reverse=True))
    ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc='lower left', title=f'diamonds: final gate, n={gate_n}' if gate else None,
              title_fontsize=8)

    ax = axs[1, 0]
    style(ax, 'Reliability at turn starts vs self-play outcomes', 'predicted P(to-move wins)')
    ax.plot([0, 1], [0, 1], color=INK2, lw=1, ls='--')
    if cal:
        first, last = cal[0], cal[-1]
        ref = 'initial random net' if any(r.get('no_warm') for r in by('config')) else 'warm start'
        for b, label, c in ((first['warm'], f"{ref} (ECE {first['ece_warm']:.3f})", S[1]),
                            (last['net'], f"iter {last['iter']} snapshots (ECE {last['ece_net']:.3f})", S[0])):
            b = np.array(b)
            ax.plot(b[:, 1], b[:, 2], color=c, lw=2, marker='o', ms=5, label=label)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2, loc='upper left')
    ax.set_ylabel('observed win rate', color=INK2, fontsize=9)

    ax = axs[1, 1]
    style(ax, 'Self-play games/s (8 workers)', 'iteration')
    mark_resumes(ax, resumes)
    if it:
        ax.plot([r['iter'] for r in it], [r['games_per_s'] for r in it], color=S[0], lw=2)
        ax.set_ylim(bottom=0)

    ax = axs[1, 2]
    style(ax, 'Mean solve time per turn start (s)', 'iteration')
    mark_resumes(ax, resumes)
    if it:
        ax.plot([r['iter'] for r in it], [r['solve_s'] for r in it], color=S[0], lw=2)
        ax.set_ylim(bottom=0)

    sets = {'visited': S[0], 'end': S[1], 'close': S[2]}
    ax = axs[2, 0]
    style(ax, 'Residual mean |V - y| on the R7 sets (line: subset, dots: all rows)', 'iteration')
    mark_resumes(ax, resumes)
    rr = [r for r in it if r.get('resid_sets')]
    if rr:
        for k, c in sets.items():
            label = k
            for full in (False, True):
                pts = [(r['iter'], r['resid_sets'][k]) for r in rr if k in r['resid_sets'] and bool(r.get('resid_full')) == full]
                if pts:
                    ax.plot(*zip(*pts), color=c, lw=0 if full else 2, marker='o', ms=6 if full else 0, label=label)
                    label = None
        ax.set_yscale('log')
        plain(ax)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2)

    ax = axs[2, 1]
    style(ax, 'New training pairs per iteration by source', 'iteration')
    mark_resumes(ax, resumes)
    rr = [r for r in it if r.get('pairs_src')]
    if rr:
        for k, c in sets.items():
            ax.plot([r['iter'] for r in rr], [r['pairs_src'][k] for r in rr], color=c, lw=2, label=k)
        ax.set_ylim(bottom=0)
        ax.legend(frameon=False, fontsize=8, labelcolor=INK2)

    ax = axs[2, 2]
    style(ax, 'Close calls per turn start (line) and mean gap (dashed, right)', 'iteration')
    mark_resumes(ax, resumes)
    rr = [r for r in it if r.get('close_per_turn') is not None]
    if rr:
        x = [r['iter'] for r in rr]
        ax.plot(x, [r['close_per_turn'] for r in rr], color=S[0], lw=2)
        ax.set_ylim(bottom=0)
        ax2 = ax.twinx()
        ax2.plot(x, [r['close_gap'] for r in rr], color=S[3], lw=2, ls='--')
        ax2.tick_params(colors=INK2, labelsize=8, length=0)
        for sp in ax2.spines.values():
            sp.set_visible(False)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out or f'runs/{run}/dashboard.png', dpi=110, facecolor=SURF)

if __name__ == '__main__':
    main(*sys.argv[1:])
