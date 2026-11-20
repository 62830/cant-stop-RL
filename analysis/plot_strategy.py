import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# runs/strategy/{opening, games}.json -> PNGs in the same dir (plotting only)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, 'runs', 'strategy')
SERIES = dict(ens='#2a78d6', S='#eb6834', vh='#1baf7a')  # reference categorical slots 1-3 (validated all-pairs, light)
LABEL = dict(ens='ensemble', S='S (single net)', vh='V_h (heuristic)')
DIV = matplotlib.colors.LinearSegmentedColormap.from_list('div', ['#e34948', '#f0efec', '#2a78d6'])  # red <- gray -> blue
INK, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
COLS = list(range(2, 13))

def fig(*a, **k):
    f, ax = plt.subplots(*a, dpi=150, facecolor=SURF, **k)
    for x in np.ravel(ax):
        x.set_facecolor(SURF)
        x.grid(color=GRID, lw=0.8, zorder=0)
        x.set_axisbelow(True)
        for s in ('top', 'right'):
            x.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            x.spines[s].set_color(GRID)
        x.tick_params(colors=MUTED, labelsize=8)
    return f, ax

def title(f, t, sub):
    f.suptitle(t, x=0.01, ha='left', color=INK, fontsize=11)
    f.text(0.01, 0.905, sub, ha='left', color=MUTED, fontsize=8)

def head(ax, t):
    ax.set_title(t, loc='left', color=INK, fontsize=9)

def lab(ax, x, y, s, color=INK, **k):
    ax.annotate(s, (x, y), xytext=(4, 0), textcoords='offset points', color=color, fontsize=7.5, va='center', **k)

def claims(G):
    f, (a, b) = fig(1, 2, figsize=(11, 4.2))
    ms = [m for m in SERIES if m in G]
    e, x, w = G[ms[0]], np.arange(11), 0.38
    any_, win = [e['claim_any'][str(c)] for c in COLS], [e['claim_winner'][str(c)] for c in COLS]
    a.bar(x - w / 2 - 0.01, any_, w, color=SERIES['ens'], label='claimed by anyone', zorder=3)
    a.bar(x + w / 2 + 0.01, win, w, color=SERIES['S'], label='claimed by the winner', zorder=3)
    for i, v in enumerate(any_):
        a.text(i - w / 2, v + 0.01, f'{v:.0%}', ha='center', color=INK, fontsize=6.5)
    hp = e['hit_prob']
    a.set_xticks(x, [f'{c}\n{hp[str(c)]:.0%}' for c in COLS])
    a.set_xlabel('column / P(makeable on a roll)', color=MUTED, fontsize=8)
    a.set_ylabel('share of games', color=MUTED, fontsize=8)
    a.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    a.legend(frameon=False, fontsize=7.5, labelcolor=INK, loc='upper left')
    head(a, f'Raw claim rate per column ({LABEL[ms[0]]} self-play)')
    for m in ms:
        y = [G[m]['claim_norm'][str(c)] for c in COLS]
        b.plot(x, y, '-o', color=SERIES[m], lw=2, ms=5, zorder=3, label=LABEL[m])
    b.axhline(1, color=MUTED, lw=1, ls='--', zorder=2)
    b.set_xticks(x, COLS)
    b.set_xlabel('column', color=MUTED, fontsize=8)
    b.set_ylabel('claim rate x turns-to-top alone (mean = 1)', color=MUTED, fontsize=8)
    b.legend(frameon=False, fontsize=7.5, labelcolor=INK, loc='upper left')
    head(b, 'Adjusted for difficulty: exact turns to top the column alone')
    title(f, 'Which columns get topped', 'self-play: ' + ', '.join(f"{LABEL[m]} {G[m]['games']}" for m in ms) + ' games; right: >1 = claimed more often than its difficulty (length and roll odds) predicts')
    f.subplots_adjust(top=0.8, bottom=0.17, wspace=0.25, left=0.06, right=0.97)
    f.savefig(os.path.join(D, 'claims.png'))

def opening(O):
    o = O['ens']['first_mover']
    f, (a, b) = fig(1, 2, figsize=(11, 4.6), gridspec_kw=dict(width_ratios=[1, 1.15]))
    # value of every first-roll result: upper triangle (a <= b), a + b = two markers (or a double), diagonal = +2 on one column
    vals = o['one_roll_values']
    M = np.full((11, 11), np.nan)
    for k, v in vals.items():
        cs = [int(c) for c in k.split('+')]
        if len(cs) == 2:
            M[cs[0] - 2, cs[1] - 2] = v
    base = O['ens']['first_mover']['v']
    lim = np.nanmax(np.abs(100 * (M - base)))
    im = a.imshow(100 * (M - base), cmap=DIV, vmin=-lim, vmax=lim, origin='upper')
    for i in range(11):
        for j in range(11):
            if not np.isnan(M[i, j]):
                d = 100 * (M[i, j] - base)
                a.text(j, i, f'{d:+.1f}', ha='center', va='center', fontsize=5.5, color='white' if abs(d) > 0.6 * lim else INK)
    a.set_xticks(range(11), COLS)
    a.set_yticks(range(11), COLS)
    a.grid(False)
    a.set_xlabel('second marker', color=MUTED, fontsize=8)
    a.set_ylabel('first marker (diagonal: one column +2)', color=MUTED, fontsize=8)
    cb = f.colorbar(im, ax=a, fraction=0.04, pad=0.02)
    cb.ax.tick_params(labelsize=7, colors=MUTED)
    cb.outline.set_visible(False)
    head(a, 'First roll: win-chance change from each two-step result (pts)')
    # P(bank) by progress, split by whether a marker sits on a column top
    t = o['bank_by_rel']
    for k, c, nm in (('topped', SERIES['S'], 'a marker at a column top'), ('not_topped', SERIES['ens'], 'no marker at a top')):
        xs = [r['x'] for r in t[k] if r['reach'] > 1e-3]
        ys = [r['p_bank'] for r in t[k] if r['reach'] > 1e-3]
        b.plot(xs, ys, '-o', color=c, lw=2, ms=4, zorder=3)
        lab(b, xs[-1], ys[-1], nm, INK)
    b.set_xlabel('progress this turn: sum over markers of steps / column length', color=MUTED, fontsize=8)
    b.set_ylabel('P(bank) at the go/stop decision', color=MUTED, fontsize=8)
    b.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
    b.set_xlim(0, max(r['x'] for r in t['topped'] + t['not_topped'] if r['reach'] > 1e-3) + 0.8)
    head(b, 'Turn 1 stop rule: bank when you top a column')
    title(f, 'The opening (first mover, exact turn-1 policy)',
          f"V(empty board) = {base:.3f}; reach-weighted from the exact solve; turn 1 busts {o['p_bust']:.0%}, banks {o['p_bank']:.0%}")
    f.subplots_adjust(top=0.8, bottom=0.13, wspace=0.25, left=0.06, right=0.97)
    f.savefig(os.path.join(D, 'opening.png'))

def bailout(G):
    f, axs = fig(1, 3, figsize=(12, 4.2))
    panels = ((lambda g: g['bank_vs_threat']['top'], 'threat', 'P(bank) right after a pick that topped a column', 100),
              (lambda g: g['turn_vs_threat']['bank_no_top'], 'threat', 'share of turns ending in a plain bank (no column topped)', 100),
              (lambda g: g['steps_at_bank_vs_v'], 'v', 'steps banked in a plain bank (mean, turn >= 3)', 30))
    for ax, (get, xk, yl, nmin) in zip(axs, panels):
        for m in ('ens', 'S', 'vh'):
            if m not in G:
                continue
            rows = [r for r in get(G[m]) if r['n'] >= nmin and r['rate'] is not None]
            xs, ys = [(r['lo'] + min(r['hi'], 1)) / 2 for r in rows], [r['rate'] for r in rows]
            ax.plot(xs, ys, '-o', color=SERIES[m], lw=2, ms=4, zorder=3, label=LABEL[m])
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
        ax.set_xlabel("opponent's chance to win on their next turn" if xk == 'threat' else 'own win probability at turn start',
                      color=MUTED, fontsize=8)
        ax.set_ylabel(yl, color=MUTED, fontsize=8)
    for ax in axs[:2]:
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
        ax.set_ylim(bottom=0)
    axs[0].legend(frameon=False, fontsize=7.5, labelcolor=INK)
    head(axs[0], 'Claim-banks fade as the threat grows')
    head(axs[1], 'Plain banks stop by ~30-50% threat')
    head(axs[2], 'Banks smaller gains when ahead')
    title(f, 'When the agent stops banking', 'self-play go/stop decisions; winning banks excluded; bins with >= 100 decisions or turns (>= 30 banks, right)')
    f.subplots_adjust(top=0.8, bottom=0.13, wspace=0.28, left=0.06, right=0.98)
    f.savefig(os.path.join(D, 'bailout.png'))

if __name__ == '__main__':
    G = json.load(open(os.path.join(D, 'games.json')))
    O = json.load(open(os.path.join(D, 'opening.json')))
    claims(G)
    opening(O)
    bailout(G)
