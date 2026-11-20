import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_strategy import plt, matplotlib, fig, title, head, lab, D, SERIES, INK, MUTED, SURF

# runs/strategy/followup.json -> declined_tops.png, bailout_lanes.png, race_count.png (plotting only)
SEQ = matplotlib.colors.LinearSegmentedColormap.from_list('seq', ['#eaf2fb', '#2a78d6', '#0d3a73'])
pct = matplotlib.ticker.PercentFormatter(1)

def declined_tops(F):
    T = F['tops']
    f, (a, b, c) = fig(1, 3, figsize=(13, 4.3), gridspec_kw=dict(width_ratios=[1, 1, 1.2]))
    L = T['live_vs_topped']
    keys = [('p_bust_after_top', 'after the top\n(column dead)'), ('p_bust_same_but_c_live', 'same markers,\ncolumn 1 short'),
            ('p_bust_after_other', 'after the\nother pick')]
    x = np.arange(3)
    for off, nm, col in ((-0.2, 'declined, 3 markers', SERIES['ens']), (0.2, 'taken, 3 markers', SERIES['S'])):
        ys = [L[nm][k] for k, _ in keys]
        a.bar(x + off, ys, 0.38, color=col, zorder=3, label=nm.replace(', 3 markers', ''))
        for xi, y in zip(x + off, ys):
            a.text(xi, y + 0.005, f'{y:.0%}', ha='center', color=INK, fontsize=7)
    a.set_xticks(x, [s for _, s in keys])
    a.yaxis.set_major_formatter(pct)
    a.set_ylabel('P(next roll busts)', color=MUTED, fontsize=8)
    a.legend(frameon=False, fontsize=7.5, labelcolor=INK, title='topping pick', title_fontsize=7.5)
    head(a, 'Topping kills a live column (3 markers out)')
    g = T['decline_rate_by']['opp_count']
    ks = list(g)
    b.bar(range(len(ks)), [g[k]['declined'] for k in ks], 0.6, color=SERIES['ens'], zorder=3)
    for i, k in enumerate(ks):
        b.text(i, g[k]['declined'] + 0.005, f"{g[k]['declined']:.0%}", ha='center', color=INK, fontsize=7)
    b.set_xticks(range(len(ks)), [k.replace('-999', '+') for k in ks])
    b.yaxis.set_major_formatter(pct)
    b.set_xlabel("opponent's race count at turn start (lower = closer to winning)", color=MUTED, fontsize=8)
    b.set_ylabel('share of topping offers declined', color=MUTED, fontsize=8)
    head(b, 'Declines grow as the opponent closes in')
    ung = [r['exact_cost_pts_per_game'] for r in T['rules'] if r['gate'] == 'always' and r['rule'] != 'always top']
    R = [r for r in T['rules'] if r['gate'] != 'always' or r['rule'] == 'always top']
    best = min(R, key=lambda r: r['exact_cost_pts_per_game'])
    for r in R:
        col = SERIES['S'] if r['rule'] == 'always top' else SERIES['vh'] if '6/7/8' in r['rule'] else SERIES['ens']
        c.plot(r['exact_cost_pts_per_game'], r['agreement'], 'o', color=col, ms=6, zorder=3)
    lab(c, R[0]['exact_cost_pts_per_game'], R[0]['agreement'], 'always top')
    xs, ys = [r['exact_cost_pts_per_game'] for r in R], [r['agreement'] for r in R]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    short = best['rule'].split(' and ', 1)[1].replace('the other pick leaves P(bust)', 'other pick bust').replace(', unless', ',\nunless')
    c.annotate(f"best: {best['gate']},\n{short}", (best['exact_cost_pts_per_game'], best['agreement']),
               xytext=(best['exact_cost_pts_per_game'] + 0.3 * (x1 - x0), best['agreement'] - 0.05 * (y1 - y0)),
               color=INK, fontsize=7.5, va='center', arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.8))
    c.plot([], [], 'o', color=SERIES['ens'], label='other pick bust <= x (exact)')
    c.plot([], [], 'o', color=SERIES['vh'], label='other pick keeps 2 of 6/7/8')
    c.legend(frameon=False, fontsize=7, labelcolor=INK, loc='lower left')
    c.set_xlim(x0 - 0.1 * (x1 - x0), x1 + 0.15 * (x1 - x0))
    c.set_ylim(y0 - 0.35 * (y1 - y0), y1 + 0.1 * (y1 - y0))
    c.yaxis.set_major_formatter(pct)
    c.set_xlabel('exact cost vs the agent (win-probability pts per game)', color=MUTED, fontsize=8)
    c.set_ylabel('agreement with the agent on topping offers', color=MUTED, fontsize=8)
    head(c, f'Rules gated on the opponent being close (ungated: {min(ung):.1f}-{max(ung):.1f} pts)')
    title(f, 'Declined topping picks', f"ensemble, {T['turns']} re-solved self-play turns; a topping pick is offered {T['offers_per_turn']:.2f}x per turn "
          f"and declined {T['declined_share']:.0%} of the time ({T['declined_share_real_choice']:.0%} when the top doesn't win)")
    f.subplots_adjust(top=0.8, bottom=0.17, wspace=0.3, left=0.05, right=0.97)
    f.savefig(os.path.join(D, 'declined_tops.png'))

def bailout_lanes(F):
    G = F['bailout']['grid']
    rows = [(0, 'all', 'opp 0'), (1, 'all', 'opp 1')] + [(2, s, f'opp 2, {s}') for s in ('3-3 steps to 3rd', '2-2 steps to 3rd', '1-1 steps to 3rd')]
    f, axs = fig(1, 3, figsize=(15, 4.3), gridspec_kw=dict(width_ratios=[1, 1, 1.1]))
    for ax, k, nm in ((axs[0], 'p_bank_top', 'P(bank | just topped a column)'), (axs[1], 'p_bank_no_top', 'P(bank | nothing topped)')):
        M = np.full((len(rows), 3), np.nan)
        N = np.zeros_like(M)
        for i, (o, s, _) in enumerate(rows):
            for m in range(3):
                g = G.get(f'opp {o}, me {m}, {s}')
                if g and g[k] is not None:
                    M[i, m], N[i, m] = g[k], g['n_top'] if k == 'p_bank_top' else g['n_no_top']
        vmax = 1 if k == 'p_bank_top' else np.nanmax(M)
        ax.imshow(M, cmap=SEQ, vmin=0, vmax=vmax, aspect='auto')
        for i in range(len(rows)):
            for m in range(3):
                if not np.isnan(M[i, m]):
                    ax.text(m, i, f'{M[i, m]:.0%}' if k == 'p_bank_top' else f'{M[i, m]:.2%}', ha='center', va='center', fontsize=8,
                            color='white' if M[i, m] > 0.6 * vmax else INK)
                    ax.text(m, i + 0.3, f'n={int(N[i, m])}', ha='center', va='center', fontsize=6, color='white' if M[i, m] > 0.6 * vmax else MUTED)
                else:
                    ax.text(m, i, 'topping wins' if k == 'p_bank_top' and m == 2 else '', ha='center', va='center', fontsize=7, color=MUTED)
        ax.grid(False)
        ax.set_xticks(range(3), ['0', '1', '2'])
        ax.set_yticks(range(len(rows)), [r[2] for r in rows])
        ax.set_xlabel('own lanes topped', color=MUTED, fontsize=8)
        head(ax, nm)
    B = [r for r in F['bailout']['by_opp_count'] if r['p_bank_top'] is not None]
    x = np.arange(len(B))
    c = axs[2]
    c.plot(x, [r['p_bank_top'] for r in B], '-o', color=SERIES['ens'], lw=2, ms=5, zorder=3)
    lab(c, x[-1], B[-1]['p_bank_top'], 'just topped')
    c.plot(x, [20 * r['p_bank_no_top'] for r in B], '-o', color=SERIES['S'], lw=2, ms=5, zorder=3)
    lab(c, x[-1], 20 * B[-1]['p_bank_no_top'], 'nothing topped (x20)')
    c.set_xticks(x, [f"{r['lo']}-{r['hi']}" if r['hi'] < 999 else f"{r['lo']}+" for r in B], fontsize=7)
    c.set_xlim(-0.4, len(B) + 1.6)
    c.yaxis.set_major_formatter(pct)
    c.set_ylim(0, 1)
    c.set_xlabel("opponent's race count (lower = closer to winning)", color=MUTED, fontsize=8)
    c.set_ylabel('P(bank) at the go/stop decision', color=MUTED, fontsize=8)
    head(c, 'Same, by the opponent\'s race count')
    title(f, 'Bail-out by lanes topped', "ensemble self-play go/stop decisions (winning banks excluded); opponent on 2 lanes split by steps left on their closest open column")
    f.subplots_adjust(top=0.8, bottom=0.13, wspace=0.4, left=0.1, right=0.98)
    f.savefig(os.path.join(D, 'bailout_lanes.png'))

def race_count(F):
    P = F['proxy']
    f, (a, b) = fig(1, 2, figsize=(11, 4.3), gridspec_kw=dict(width_ratios=[1.3, 1]))
    rows = P['ratio table']['rows']
    mid = [r['mid'] for r in rows]
    a.vlines(mid, [r['q10'] for r in rows], [r['q90'] for r in rows], color=SERIES['ens'], lw=2, alpha=0.35, zorder=2)
    a.plot(mid, [r['v'] for r in rows], 'o', color=SERIES['ens'], ms=6, zorder=3, label="agent's V (mean, 10-90% band)")
    q = P['half-turns/step (2,3,4,5,8,14), log ratio']
    xs = np.exp(np.linspace(np.log(0.12), np.log(7), 100))
    a.plot(xs, 1 / (1 + np.exp(-(q['a'] + q['b'] * np.log(xs)))), color=SERIES['S'], lw=2, zorder=3, label='fit')
    a.set_xscale('log')
    a.set_xticks([0.25, 0.5, 1, 2, 4], ['1/4', '1/2', '1', '2', '4'])
    a.yaxis.set_major_formatter(pct)
    a.set_xlabel("opponent's count / your count (turn start, you to move)", color=MUTED, fontsize=8)
    a.set_ylabel('win probability', color=MUTED, fontsize=8)
    a.legend(frameon=False, fontsize=7.5, labelcolor=INK, loc='upper left')
    head(a, 'Race-count ratio tracks the agent\'s V')
    ks = [('lanes topped (3x3 table of mean V)', 'lanes topped only'), ('unweighted steps, log ratio', 'remaining steps, unweighted'),
          ('half-turns/step (2,3,4,5,8,14), difference', 'race count, difference'), ('half-turns/step (2,3,4,5,8,14), log ratio', 'race count, ratio'),
          ('exact turns/step, log ratio', 'exact turns/step, ratio')]
    ys = [P[k]['mae_pts_turn2plus'] for k, _ in ks]
    b.barh(range(len(ks)), ys, 0.6, color=[SERIES['ens'] if 'race' in n else MUTED for _, n in ks], zorder=3)
    for i, (y, (k, _)) in enumerate(zip(ys, ks)):
        b.text(y + 0.1, i, f"{y:.1f} pts  (ρ {P[k]['spearman']:.3f})", va='center', color=INK, fontsize=7.5)
    b.set_yticks(range(len(ks)), [n for _, n in ks])
    b.set_xlim(0, max(ys) * 1.45)
    b.set_xlabel('mean absolute error vs V, turns >= 2 (win-probability pts)', color=MUTED, fontsize=8)
    head(b, 'Accuracy of countable proxies')
    title(f, 'A mental win-chance proxy: the race count', 'count = sum over your best 3 open columns of remaining steps x weight '
          '(7: 2, 6/8: 3, 5/9: 4, 4/10: 5, 3/11: 8, 2/12: 14; topped = 0); ensemble self-play turn starts')
    f.subplots_adjust(top=0.8, bottom=0.13, wspace=0.45, left=0.06, right=0.97)
    f.savefig(os.path.join(D, 'race_count.png'))

if __name__ == '__main__':
    F = json.load(open(os.path.join(D, 'followup.json')))
    if 'tops' in F:
        declined_tops(F)
    bailout_lanes(F)
    race_count(F)
