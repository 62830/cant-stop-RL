import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# runs/warm_bias/{results.json, data.npz} -> four PNGs in the same dir (plotting only; no computation beyond the beta line)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, 'runs', 'warm_bias')
SERIES = ['#2a78d6', '#eb6834', '#1baf7a']  # reference categorical slots 1-3
INK, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
XS = ['H', 'T', 'S74', 'S208', 'S280']
SETS = ['P_H', 'P_T', 'P_S']
LABEL = dict(P_H='P_H (V_h games)', P_T='P_T (td30 games)', P_S='P_S (scratch30 games)')

def style(ax):
    ax.set_facecolor(SURF)
    ax.grid(color=GRID, lw=0.8, zorder=0)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)

def fig(*a, **k):
    f, ax = plt.subplots(*a, dpi=150, facecolor=SURF, **k)
    for x in np.ravel(ax):
        style(x)
    return f, ax

def title(f, ax, t, sub):
    f.suptitle(t, x=0.01, ha='left', color=INK, fontsize=11)
    f.text(0.01, 0.905, sub, ha='left', color=MUTED, fontsize=8)

def save(f, name):
    f.tight_layout(rect=(0, 0, 1, 0.88))
    f.savefig(os.path.join(D, name), facecolor=SURF)
    plt.close(f)

def scatter(res, z):
    names = list(z['names'])
    s, h, t = names.index('S'), names.index('H'), names.index('T')
    f, axs = fig(1, 2, figsize=(10, 4.8))
    for ax, kind, col in zip(axs, ('raw', 'solved'), ('raw', 'sol')):
        lim = 0
        for k, c in zip(SETS, SERIES):
            v = z[f'{k}__{col}']
            v = v[z[f'{k}__sel']] if k == 'P_H' else v
            x, y = v[:, h] - v[:, s], v[:, t] - v[:, s]
            lim = max(lim, np.abs(x).max(), np.abs(y).max())
            ax.scatter(x, y, s=9, color=c, alpha=0.45, lw=0.4, edgecolor=SURF, label=f'{LABEL[k]}, n={len(x)}', zorder=2)
        b = res['positions'][kind]['pooled']['beta']
        g = np.array([-lim, lim])
        ax.plot(g, g, color=MUTED, lw=1, ls=':', zorder=1)
        ax.axhline(0, color=MUTED, lw=1, ls='--', zorder=1)
        ax.plot(g, b['est'] * g, color=INK, lw=2, zorder=3)
        ax.text(0.97 * lim, 0.97 * lim, 'y = x (full imprint)', color=MUTED, fontsize=7, ha='right', va='top', rotation=0)
        ax.text(0.97 * lim, 0.03 * lim, 'y = 0 (no imprint)', color=MUTED, fontsize=7, ha='right', va='bottom')
        ax.text(-0.97 * lim, 0.9 * lim, f"β = {b['est']:.2f} [{b['ci'][0]:.2f}, {b['ci'][1]:.2f}]", color=INK, fontsize=9,
                ha='left', va='top', bbox=dict(facecolor=SURF, edgecolor=GRID))
        ax.set_xlim(-lim, lim), ax.set_ylim(-lim, lim)
        ax.set_xlabel(f'V_h − S ({kind})', color=MUTED, fontsize=9)
        ax.set_ylabel(f'td30 − S ({kind})', color=MUTED, fontsize=9)
        ax.set_title('raw V at turn start' if kind == 'raw' else 'one-turn solved value', color=INK, fontsize=9, loc='left')
    axs[0].legend(frameon=False, fontsize=7, loc='lower right')
    title(f, axs, 'Does td30 inherit V_h’s deviations from scratch30?',
          'per turn start, mover POV; black: through-origin OLS fit (β, 95% game-clustered bootstrap CI), pooled over sets')
    save(f, 'imprint_scatter.png')

def dots(res):
    f, axs = fig(1, 2, figsize=(10, 3.8), sharey=True)
    y = np.arange(len(XS))[::-1]
    for ax, kind in zip(axs, ('raw', 'solved')):
        P = res['positions'][kind]
        for j, (k, c) in enumerate(zip(SETS + ['pooled'], SERIES + [INK])):
            m = np.array([P[k][x]['mean_abs']['est'] for x in XS])
            lo, hi = (np.array([P[k][x]['mean_abs']['ci'][i] for x in XS]) for i in (0, 1))
            yy = y + (1.5 - j) * 0.17
            ax.errorbar(m, yy, xerr=[m - lo, hi - m], fmt='o', ms=4 if k != 'pooled' else 5.5, color=c, ecolor=c,
                        elinewidth=1.5, capsize=0, label=LABEL.get(k, 'pooled'), zorder=3)
        ax.set_yticks(y, ['V_h', 'td30 it82 (T)', 'scratch it74', 'scratch it208', 'scratch it280'], color=INK, fontsize=8)
        ax.set_xlim(left=0)
        ax.set_xlabel(f'mean |V_X − V_S|  ({kind}, win prob)', color=MUTED, fontsize=9)
        ax.set_title('raw V at turn start' if kind == 'raw' else 'one-turn solved value', color=INK, fontsize=9, loc='left')
    axs[1].legend(frameon=False, fontsize=7, loc='lower right')
    title(f, axs, 'Is td30 further from scratch30 than scratch30’s own compute-matched snapshot?',
          'deviation from S = scratch30 it297 at turn starts; dots: mean, bars: 95% game-clustered bootstrap CI')
    save(f, 'deviation_dots.png')

def decisions(res):
    Dp = res['decisions']['pooled']
    f, axs = fig(1, 2, figsize=(10, 3.8))
    x, w = np.arange(len(XS)), 0.38
    for ax, met, lab, sc in ((axs[0], 'disagree', 'disagreement with S’s choice (%)', 100),
                             (axs[1], 'regret_per_turn', 'regret per turn under S (win-prob points)', 100)):
        for j, (kind, c) in enumerate(zip(('gostop', 'pick'), SERIES)):
            m = np.array([Dp[X][kind][met]['est'] for X in XS]) * sc
            lo, hi = (np.array([Dp[X][kind][met]['ci'][i] for X in XS]) * sc for i in (0, 1))
            xx = x + (j - 0.5) * w
            ax.bar(xx, m, w - 0.04, color=c, label='go/stop' if kind == 'gostop' else 'dice pick', zorder=2)
            ax.errorbar(xx, m, [m - lo, hi - m], fmt='none', ecolor=INK, elinewidth=1, capsize=2, zorder=3)
        ax.set_xticks(x, ['V_h', 'td30 (T)', 'S it74', 'S it208', 'S it280'], color=INK, fontsize=8)
        ax.set_ylabel(lab, color=MUTED, fontsize=9)
        ax.grid(axis='x', visible=False)
    axs[0].legend(frameon=False, fontsize=8)
    title(f, axs, 'How often, and how costly, are decisions that differ from scratch30’s?',
          'all real decisions in the P_H, P_T and P_S games, pooled; bars: 95% game-clustered bootstrap CI')
    save(f, 'decisions.png')

def trend(res):
    f, axs = fig(1, 2, figsize=(10, 3.8))
    for ax, j, ref in ((axs[0], 1, 'scratch30 final (S)'), (axs[1], 2, 'V_h')):
        for run, c in zip(('scratch30', 'td30'), SERIES):
            r = np.array(res['trend'][run])
            ax.plot(r[:, 0], r[:, j], color=c, lw=2, marker='o', ms=4, label=run, zorder=3)
            ax.annotate(run, (r[-1, 0], r[-1, j]), xytext=(5, 0), textcoords='offset points', color=INK, fontsize=8, va='center')
        ax.set_xlabel('snapshot (iteration)', color=MUTED, fontsize=9)
        ax.set_title(f'mean |V − {ref}|', color=INK, fontsize=9, loc='left')
        ax.set_ylim(0, max(max(r[j] for r in res['trend'][q]) for q in res['trend']) * 1.05)
    axs[0].set_ylabel('mean |ΔV| on P_H (win prob)', color=MUTED, fontsize=9)
    axs[0].legend(frameon=False, fontsize=8)
    title(f, axs, 'Does td30 move away from V_h and toward scratch30 as it trains?',
          'raw value at the 900 P_H turn starts, every 10th snapshot (td30 it0 = warm-start fit, scratch30 it0 = random init)')
    save(f, 'trend.png')

def main(d=D):
    global D
    D = d
    res, z = json.load(open(os.path.join(d, 'results.json'))), np.load(os.path.join(d, 'data.npz'))
    scatter(res, z), dots(res), decisions(res), trend(res)

if __name__ == '__main__':
    main(*sys.argv[1:])
