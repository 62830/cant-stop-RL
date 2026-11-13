import os, sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# runs/ens_s*/metrics.jsonl, scratch30's archived metrics.jsonl, runs/ensemble/results.json -> three PNGs in runs/ensemble
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.environ.get('ENS_DIR', os.path.join(ROOT, 'runs', 'ensemble'))
S30 = os.path.join(ROOT, '..', 'cant_stop_archive', 'cant_stop_scratch', 'runs', 'scratch30', 'metrics.jsonl')
SERIES = ['#2a78d6', '#eb6834', '#1baf7a']  # reference categorical slots 1-3
INK, MUTED, GRID, SURF, REF = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb', '#8a8984'
RUNS = ['ens_s2', 'ens_s4', 'ens_s5']

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

def title(f, t, sub):
    f.suptitle(t, x=0.01, ha='left', color=INK, fontsize=11)
    f.text(0.01, 0.905, sub, ha='left', color=MUTED, fontsize=8)

def save(f, name):
    f.tight_layout(rect=(0, 0, 1, 0.88))
    f.savefig(os.path.join(D, name), facecolor=SURF)
    plt.close(f)

def metrics(path):
    recs = [json.loads(l) for l in open(path)]
    games = next(r for r in recs if r['kind'] == 'config')['games']
    it = [(r['iter'] * games, r['residual']) for r in recs if r['kind'] == 'iter' and r['residual'] is not None]
    ece = [(r['iter'] * games, r['ece_net']) for r in recs if r['kind'] == 'calib']
    drops = [r['iter'] * games for r in recs if r['kind'] == 'lr_drop']
    drops += [r['iter'] * games for r in recs if r['kind'] == 'resume'][1:]  # scratch30: second resume set lr 1e-4
    return np.array(it), np.array(ece), drops

def smooth(y, k=9):
    return np.convolve(np.pad(y, (k // 2, k // 2), mode='edge'), np.ones(k) / k, mode='valid')

def curves():
    f, axs = fig(1, 2, figsize=(10, 4))
    lab = dict(ens_s2='ens_s2 (M1)', ens_s4='ens_s4 (M2, stop warmup)', ens_s5='ens_s5 (M3, stop warmup)')
    ser = [(lab[r], os.path.join(ROOT, 'runs', r, 'metrics.jsonl'), c, '-') for r, c in zip(RUNS, SERIES)]
    ser.append(('scratch30 (S)', S30, REF, '--'))
    for name, path, c, ls in ser:
        rs, ece, drops = metrics(path)
        for ax, (x, y), raw in ((axs[0], rs.T, True), (axs[1], ece.T, False)):
            if raw:
                ax.plot(x, y, color=c, lw=0.6, alpha=0.25)
            ax.plot(x, smooth(y, 9 if raw else 3), color=c, lw=2, ls=ls, label=name)
            for d in drops:
                ax.axvline(d, color=c, lw=1, ls=':', alpha=0.9)
    axs[0].set(ylabel='held-out one-turn residual |V − solve(V)|', xlabel='self-play games')
    axs[1].set(ylabel='ECE of V at turn starts (10 bins)', xlabel='self-play games')
    axs[0].set_ylim(0, 0.03)
    for ax in axs:
        ax.yaxis.label.set_color(INK)
        ax.xaxis.label.set_color(INK)
        ax.yaxis.label.set_fontsize(8)
        ax.xaxis.label.set_fontsize(8)
    axs[1].legend(fontsize=8, frameon=False, labelcolor=INK)
    title(f, 'Do the three from-scratch runs converge like scratch30, and where does the LR drop act?',
          'thick: moving average (thin: raw residual); dotted vertical: lr 1e-3 → 1e-4 (ens: in-run at 0.67 of time, Adam kept; '
          'scratch30: on resume, Adam and buffer reset)')
    save(f, 'a_training_curves.png')

def dot(ax, y, est, ci, c, label):
    ax.plot(ci, [y, y], color=c, lw=2, solid_capstyle='round', zorder=3)
    ax.plot(est, y, 'o', ms=8, color=c, mec=SURF, mew=2, zorder=4)
    ax.text(ci[1], y + 0.22, label, color=MUTED, fontsize=7, ha='right', va='bottom', zorder=5,
            bbox=dict(fc=SURF, ec='none', pad=1))

def forest(res):
    G, dec = res.get('gates', {}), res.get('decision', {})
    rows = [(k, G[f'{k}_vs_H']) for k in ('M1', 'M2', 'M3', 'E3', 'E4') if f'{k}_vs_H' in G]
    f, (a1, a2) = fig(1, 2, figsize=(10, 4), gridspec_kw=dict(width_ratios=[2.2, 1]))
    ys = list(range(len(rows) + 1))[::-1]
    for y, (k, g) in zip(ys, rows):
        c = SERIES[0] if k.startswith('E') else INK
        dot(a1, y, 100 * g['adj'][0], [100 * x for x in g['adj'][1]], c,
            f"n={g['n']}  SD adj {g['sd_adj']:.3f} (raw {g['sd_raw']:.3f})")
    s = res['S_vs_H_ref']
    dot(a1, ys[-1], 100 * s['adj'], [100 * (s['adj'] - s['pm']), 100 * (s['adj'] + s['pm'])], REF, 'PR #15, n=1000')
    a1.set_yticks(ys, [k for k, _ in rows] + ['S (ref)'], color=INK, fontsize=9)
    a1.axvline(50, color=MUTED, lw=1)
    a1.set_xlabel('adjusted pair score vs V_h, % (95% CI)', fontsize=8, color=INK)
    a1.set_ylim(-0.6, len(rows) + 0.8)
    if 'E3_minus_meanM' in dec:
        e = dec['E3_minus_meanM']
        dot(a2, 0, 100 * e['est'], [100 * x for x in e['ci']], SERIES[0], f"{100 * e['est']:+.2f} pts")
        a2.set_yticks([0], ['E3 − mean(M)'], color=INK, fontsize=9)
        a2.set_ylim(-1, 1)
    a2.axvline(0, color=MUTED, lw=1)
    a2.set_xlabel('difference, pts (95% CI)', fontsize=8, color=INK)
    title(f, 'Does the 3-net ensemble beat its members (and S) against V_h?',
          'luck-adjusted paired eval (ref = V_h); 50 = parity; M1–M3 = final ens_s2, s4, s5 nets, E3 = mean logit, E4 = E3 + S')
    save(f, 'b_gates_vs_vh.png')

def vs_s(res):
    G, dec, dist = res.get('gates', {}), res.get('decision', {}), res.get('dist', {})
    f, (a1, a2) = fig(1, 2, figsize=(10, 4))
    rows = [(k, G[k]['adj'], f"n={G[k]['n']}  SD adj {G[k]['sd_adj']:.3f}") for k in ('E3_vs_S_pilot', 'E3_vs_S') if k in G]
    if 'E3_vs_S' in dec:
        p = dec['E3_vs_S']
        rows.append(('pooled', p['adj'], f"n={p['n']}"))
    for y, (k, adj, lab) in zip(range(len(rows))[::-1], rows):
        dot(a1, y, 100 * adj[0], [100 * x for x in adj[1]], SERIES[0] if k == 'pooled' else INK, lab)
    a1.set_yticks(range(len(rows))[::-1], [r[0].replace('E3_vs_S', 'main').replace('main_pilot', 'pilot') for r in rows], color=INK, fontsize=9)
    a1.axvline(50, color=MUTED, lw=1)
    a1.set_ylim(-0.6, len(rows) - 0.2)
    a1.set_xlabel('E3 adjusted pair score vs S, % (95% CI)', fontsize=8, color=INK)
    if dist:
        items = [(k, v) for k, v in dist['pairwise'].items()] + [(f'{k}–S', dist['to_S'][k]) for k in ('M1', 'M2', 'M3', 'E3', 'H')]
        ys = np.arange(len(items))[::-1]
        for y, (k, v) in zip(ys, items):
            c = SERIES[1] if '–S' not in k else (REF if k.startswith('H') else INK)
            a2.barh(y, 100 * v['mean'], height=0.55, color=c, zorder=3)
            a2.text(100 * v['mean'] + 0.05, y, f"{100 * v['mean']:.2f}", va='center', fontsize=7, color=MUTED)
        a2.set_yticks(ys, [k.replace('-', '–') for k, _ in items], color=INK, fontsize=9)
        a2.set_xlabel(f"mean |ΔV| on P_H ({dist['n']} V_h turn starts), win-prob pts", fontsize=8, color=INK)
    title(f, 'Does E3 beat S head-to-head, and are the members diverse enough for averaging to matter?',
          'left: luck-adjusted paired eval (ref = V_h), 50 = parity; right: orange = member pairs, black = distance to S, gray = V_h to S')
    save(f, 'c_e3_vs_s_diversity.png')

if __name__ == '__main__':
    os.makedirs(D, exist_ok=True)
    curves()
    res = json.load(open(os.path.join(D, 'results.json'))) if os.path.exists(os.path.join(D, 'results.json')) else {}
    if res:
        forest(res)
        vs_s(res)
