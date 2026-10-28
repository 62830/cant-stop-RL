# before/after chart of runs/phase45/bench.json -> runs/phase45/speed.png
import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'runs', 'phase45')
OLD, NEW, INK, MUTED, GRID = '#a3a29c', '#2a78d6', '#0b0b0b', '#52514e', '#e4e3df'

def bars(ax, labels, old, new, title, unit):
    x, w = np.arange(len(labels)), 0.38
    for dx, vals, col, name in ((-w / 2, old, OLD, 'before'), (w / 2, new, NEW, 'after')):
        bs = ax.bar(x + dx, vals, w - 0.03, color=col, label=name)
        ax.bar_label(bs, fmt='%.0f' if max(vals) >= 20 else '%.1f', padding=2, fontsize=8, color=MUTED)
    ax.set_xticks(x, labels)
    ax.set_title(title, loc='left', fontsize=11, color=INK)
    ax.set_ylabel(unit, color=MUTED)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.spines['left'].set_color(GRID)
    ax.spines['bottom'].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_axisbelow(True)

def main():
    r = json.load(open(os.path.join(OUT, 'bench.json')))
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2), facecolor='#fcfcfb')
    st = r['stages']['empty']
    old = [st['old']['enumerate'], st['old']['stop_boards'] + st['old']['value'], st['old']['fill']]
    bars(axs[0], ['enumerate', 'stop values (net)', 'fill'], old, [st['new'][k] for k in ('enumerate', 'value', 'fill')],
         f"Solve stages, empty board ({st['situations']:,} situations)", 'ms')
    labs, o, n = [], [], []
    for b in ('mid', 'empty'):
        for v in ('vh', 'net'):
            labs.append(f"{b}\n{'V_h' if v == 'vh' else 'net'}")
            o.append(r['solve'][b][v]['old'])
            n.append(r['solve'][b][v]['new'])
    bars(axs[1], labs, o, n, 'Single-process solve', 'ms per solve')
    sp = r['selfplay']
    old8 = next(s['games_per_s'] for s in sp if s['impl'] == 'old')
    new = [s for s in sp if s['impl'] == 'new']
    bars(axs[2], [f"{s['workers']} workers" for s in new], [old8 if s['workers'] == 8 else 0 for s in new], [s['games_per_s'] for s in new],
         'Self-play with the net (before: 8 workers only)', 'games / s')
    for t in axs[2].texts:
        if t.get_text() in ('0', '0.0'):
            t.set_visible(False)
    axs[1].legend(frameon=False, loc='upper left')
    for ax in axs:
        ax.set_facecolor('#fcfcfb')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'speed.png'), dpi=130)

if __name__ == '__main__':
    main()
