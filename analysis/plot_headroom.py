import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# runs/headroom/{results.json, pilot.json} -> four PNGs in the same dir (plotting only)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, 'runs', 'headroom')
SERIES = dict(H='#2a78d6', T='#eb6834', S='#1baf7a')  # reference categorical slots 1-3
INK, MUTED, GRID, SURF, BAND = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb', '#f0efec'
LABEL = dict(H='H (V_h)', T='T (td30 it82)', S='S (scratch30 it297)')
BASES = list(SERIES)

def fig(*a, **k):
    f, ax = plt.subplots(*a, dpi=150, facecolor=SURF, **k)
    for x in np.ravel(ax):
        x.set_facecolor(SURF)
        x.grid(color=GRID, lw=0.8, zorder=0)
        for s in ('top', 'right'):
            x.spines[s].set_visible(False)
        for s in ('left', 'bottom'):
            x.spines[s].set_color(GRID)
        x.tick_params(colors=MUTED, labelsize=8)
    return f, ax

def title(f, t, sub):
    f.suptitle(t, x=0.01, ha='left', color=INK, fontsize=11)
    f.text(0.01, 0.905, sub, ha='left', color=MUTED, fontsize=8)

def save(f, name):
    f.tight_layout(rect=(0, 0, 1, 0.88))
    f.savefig(os.path.join(D, name), facecolor=SURF)
    plt.close(f)

def pts(s):
    return 100 * s['est'], 100 * np.array(s['ci'])

def errbar(ax, x, s, c, **k):
    e, ci = pts(s)
    ax.errorbar(x, e, yerr=[[e - ci[0]], [ci[1] - e]], fmt='o', color=c, ms=7, lw=2, capsize=0, zorder=3, **k)
    return e, ci

def headroom(R):
    f, ax = fig(figsize=(8, 4.6))
    hs = [100 * np.array(R['bases'][X]['all'][e]['headroom']['ci']) for X in BASES for e in ('split', 'sig', 'naive')]
    lo, top = min(h[0] for h in hs) - 0.4, max(4, max(h[1] for h in hs) + 0.4)
    for y0, y1, c, t in ((1, 3, BAND, '1-3 pts: real but small'), (3, top, '#e4e3df', '> 3 pts: values matter')):
        ax.axhspan(y0, y1, color=c, zorder=0)
        ax.text(2.55, y1 - 0.08, t, ha='right', va='top', color=MUTED, fontsize=7)
    ax.text(2.55, 0.92, '< 1 pt: exhausted', ha='right', va='top', color=MUTED, fontsize=7)
    ax.axhline(0, color=INK, lw=0.8, zorder=1)
    marks = (('split', 'v', 'lower: split-sample (biased low)'), ('sig', 'o', 'middle: switch only if half-sample margin > 2 SE'),
             ('naive', '^', 'upper: same-sample (biased high)'))
    for i, X in enumerate(BASES):
        for j, (est, m, lab) in enumerate(marks):
            e, ci = pts(R['bases'][X]['all'][est]['headroom'])
            ax.errorbar(i + (j - 1) * 0.18, e, yerr=[[e - ci[0]], [ci[1] - e]], fmt=m, color=SERIES[X], ms=7, lw=2,
                        mfc=SERIES[X] if est == 'sig' else SURF, zorder=3, label=lab if i == 0 else None)
            if est == 'sig':
                ax.text(i + 0.05, ci[1] + 0.12, f'{e:.2f}', color=INK, fontsize=8, ha='center')
    ax.set_xticks(range(3), [LABEL[X] for X in BASES], color=INK, fontsize=9)
    ax.set_xlim(-0.5, 2.6)
    ax.set_ylim(lo, top)
    ax.set_ylabel('headroom (win-prob points / game)', color=MUTED, fontsize=8)
    leg = ax.legend(frameon=False, fontsize=7, loc='upper left')
    for h in leg.legend_handles:
        h.set_color(MUTED)
    title(f, 'How much win rate does one-turn play leave on the table, per value function?',
          f"c x mean regret, {R['bases']['H']['n_contested_sampled']} contested decisions/base vs H, K={R['K']}, R={R['R']}, "
          'control-variate Q; true headroom (>= 0) lies between the brackets; 95% CI')
    save(f, 'headroom.png')

def breakdown(R):
    f, axs = fig(1, 2, figsize=(10, 4.2), sharey=True, gridspec_kw=dict(width_ratios=(2, 3)))
    for ax, groups in zip(axs, (('gostop', 'pick'), ('early', 'mid', 'late'))):
        for i, X in enumerate(BASES):
            for j, g in enumerate(groups):
                errbar(ax, j + (i - 1) * 0.22, R['bases'][X][g]['sig']['headroom'], SERIES[X],
                       label=LABEL[X] if j == 0 and ax is axs[0] else None)
        ax.axhline(0, color=INK, lw=0.8)
        names = dict(gostop='go / stop', pick='pick', early='early', mid='mid', late='late')
        ax.set_xticks(range(len(groups)), [names[g] for g in groups], color=INK, fontsize=9)
    c = R['phase_cuts']
    axs[1].set_xlabel(f'game phase: total banked fraction (sum over columns, both players) tertiles at {c[0]:.2f}, {c[1]:.2f}',
                      color=MUTED, fontsize=8)
    axs[0].set_ylabel('headroom contribution (pts / game)', color=MUTED, fontsize=8)
    axs[0].legend(frameon=False, fontsize=8, loc='upper left')
    title(f, 'Where is the headroom: go/stop or picks, early or late?',
          'per-game contribution (subsets sum to the total), middle estimator (switch only if half-sample margin > 2 SE); '
          '95% game-clustered CI')
    save(f, 'breakdown.png')

def truncation(P):
    f, axs = fig(1, 2, figsize=(10, 4.6))
    for X in BASES:
        b = P['bases'][X]['contested']
        q = np.array(b['qmean_pairs'])
        axs[0].scatter(q[:, 2], q[:, 0], s=12, color=SERIES[X], alpha=0.6, lw=0, label=LABEL[X])
    axs[0].plot([0, 1], [0, 1], color=MUTED, lw=1, ls='--')
    axs[0].set_xlabel('Q, full-length rollouts', color=MUTED, fontsize=8)
    axs[0].set_ylabel('Q, K=2 truncated (scored by mean of H, T, S)', color=MUTED, fontsize=8)
    axs[0].legend(frameon=False, fontsize=8)
    for i, X in enumerate(BASES):
        b = P['bases'][X]['contested']
        for j, k in enumerate(('2', '3')):
            r = b['bias_delta'][k]
            e, h = 100 * r['mean'], 196 * r['se']
            axs[1].errorbar(j + (i - 1) * 0.2, e, yerr=h, fmt='o', color=SERIES[X], ms=6, lw=2, zorder=3)
    axs[1].axhline(0, color=INK, lw=0.8)
    axs[1].set_xticks(range(2), ['K=2 - full', 'K=3 - full'], color=INK, fontsize=9)
    axs[1].set_xlim(-0.6, 1.6)
    axs[1].set_ylabel('bias of Q(a) - Q(X choice), win-prob pts per decision', color=MUTED, fontsize=8)
    title(f, 'Does truncating rollouts at K turns bias Q or regret? (pilot)',
          f"same decisions and dice (K=2/3 are prefixes of the full rollouts), {P['bases']['H']['contested']['n']} contested "
          f"decisions per base, R={P['R']}, control-variate Q; right: mean +-1.96 SE over (decision, action) pairs")
    save(f, 'truncation.png')

def h_vs_s(R):
    h = R['h_vs_s']
    f, axs = fig(1, 2, figsize=(9, 4))
    for ax, keys, lab in ((axs[0], ('H_is_best', 'S_is_best'), 'fraction where the choice is rollout-best'),
                          (axs[1], ('regret_H', 'regret_S'), 'mean regret of the choice (pts)')):
        for i, (k, X) in enumerate(zip(keys, 'HS')):
            s = h[k]
            sc = 1 if ax is axs[0] else 100
            e, ci = sc * s['est'], sc * np.array(s['ci'])
            ax.errorbar(i, e, yerr=[[e - ci[0]], [ci[1] - e]], fmt='o', color=SERIES[X], ms=8, lw=2, zorder=3)
            ax.text(i + 0.06, e, f'{e:.2f}' if sc == 1 else f'{e:.2f}', fontsize=8, color=INK, va='center')
        ax.set_xticks(range(2), ["H's choice", "S's choice"], color=INK, fontsize=9)
        ax.set_xlim(-0.5, 1.6)
        ax.set_ylabel(lab, color=MUTED, fontsize=8)
    d = h['regret_H_minus_S']
    title(f, 'When H and S disagree, which one picks the rollout-best action?',
          f"{h['n']} sampled decisions where H's and S's solvers differ (all bases); regret H - S = "
          f"{100 * d['est']:.2f} pts [{100 * d['ci'][0]:.2f}, {100 * d['ci'][1]:.2f}]; 95% game-clustered CI")
    save(f, 'h_vs_s.png')

def main():
    P = json.load(open(os.path.join(D, 'pilot.json')))
    truncation(P)
    if os.path.exists(os.path.join(D, 'results.json')):
        R = json.load(open(os.path.join(D, 'results.json')))
        headroom(R)
        breakdown(R)
        h_vs_s(R)

if __name__ == '__main__':
    main()
