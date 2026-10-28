# Bellman residual (mean |V - y| on the 200 fixed held-out boards) across TD runs, vs iteration and vs wall-clock.
# usage: python analysis/residual_compare.py OUT.png metrics.jsonl [metrics.jsonl ...]
# Each file is split into lr segments (a resume with a new --lr starts one); prints each segment's floor statistics.
import sys, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
S = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']  # fixed categorical order, never cycled

def segments(path):
    segs, cfg = [], None
    for r in map(json.loads, open(path)):
        if r['kind'] == 'config':
            cfg = r
        elif r['kind'] == 'iter' and r['residual'] is not None:
            label = f"{cfg['run']} {cfg['hidden']} lr {cfg['lr']:g}"
            if not segs or segs[-1]['label'] != label:
                segs.append(dict(label=label, it=[], t=[], r=[]))
            for k, v in (('it', r['iter']), ('t', r['t'] / 60), ('r', r['residual'])):
                segs[-1][k].append(v)
    return [{k: np.array(v) if k != 'label' else v for k, v in s.items()} for s in segs]

def smooth(y, k=5):
    return np.array([y[max(0, i - k + 1):i + 1].mean() for i in range(len(y))])

def floor(s):
    third = s['it'] >= s['it'][0] + 2 * (s['it'][-1] - s['it'][0]) / 3
    slope = 100 * np.polyfit(s['it'][third], s['r'][third], 1)[0] if third.sum() >= 3 else None
    return s['r'][-10:].mean(), s['r'][-10:].std(), slope

def main(out, *paths):
    segs = [s for p in paths for s in segments(p)]
    if len(segs) > len(S):
        raise SystemExit(f'{len(segs)} segments > {len(S)} colours; compare fewer runs')
    fig, axs = plt.subplots(1, 2, figsize=(14, 5.5), facecolor=SURF)
    for ax, x, xl in ((axs[0], 'it', 'iteration'), (axs[1], 't', 'wall-clock (min, cumulative over resumes)')):
        ax.set_facecolor(SURF)
        ax.set_xlabel(xl, color=INK2, fontsize=9)
        ax.grid(color=GRID, lw=0.8, which='both')
        ax.tick_params(colors=INK2, labelsize=8, length=0, which='both')
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_yscale('log')
        for f in (ax.yaxis.set_major_formatter, ax.yaxis.set_minor_formatter):
            f(FormatStrFormatter('%.3f'))
        for c, s in zip(S, segs):
            ax.plot(s[x], s['r'], color=c, lw=0.8, alpha=0.35)
            ax.plot(s[x], smooth(s['r']), color=c, lw=2, label=s['label'])
            ax.annotate(s['label'].split(' ')[0] + ' ' + s['label'].split(' ')[-1], (s[x][-1], smooth(s['r'])[-1]), xytext=(4, 0),
                        textcoords='offset points', va='center', fontsize=8, color=INK2)
    axs[0].set_title('Held-out Bellman residual mean |V(s) - y(s)| (log; bold = trailing 5-measurement mean)', loc='left', color=INK, fontsize=11)
    axs[0].set_ylabel('residual', color=INK2, fontsize=9)
    axs[1].legend(frameon=False, fontsize=8, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=110, facecolor=SURF)
    print('segment | iters | minutes | floor (mean last 10 ± sd) | slope over last third (per 100 iters)')
    for s in segs:
        m, sd, sl = floor(s)
        print(f"{s['label']} | {s['it'][0]}-{s['it'][-1]} | {s['t'][0]:.1f}-{s['t'][-1]:.1f} | {m:.5f} ± {sd:.5f} | {'n/a' if sl is None else f'{sl:+.5f}'}")

if __name__ == '__main__':
    main(*sys.argv[1:])
