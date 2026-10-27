import os, sys, json, time, random
from functools import partial
from multiprocessing import Pool
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import fastrules as fr
from game import State
from baselines import V2, V2Tuned
from eval import evaluate, wilson
from heuristics import VH, KINDS, features, fit, log_loss
from solver_agent import SolverAgent

OUT = os.path.join(ROOT, 'runs', 'phase3')
CACHE = os.path.join(OUT, 'cache')
RES = os.path.join(OUT, 'results.json')
GUESS = VH('top3', (4.0,))
N_SELF, N_H2H, N_GATE, N_ABL = 1500, 500, 2000, 1000

def load():
    return json.load(open(RES)) if os.path.exists(RES) else {}

def save(res):
    json.dump(res, open(RES, 'w'), indent=1)

def stage(res, name, f):
    if name not in res:
        t = time.time()
        res[name] = f()
        res[name + '_seconds'] = round(time.time() - t, 1)
        save(res)
        print(name, json.dumps(res[name])[:400], flush=True)
    return res[name]

def vh(d):
    return VH(d['kind'], d['theta'], d['bias'])

def _selfplay(args):
    make, n, seed, g0 = args
    random.seed(seed)
    a, b = make(), make()
    progs, ms, ys, gs = [], [], [], []
    for g in range(n):
        a.reset(), b.reset()
        s, agents, starts = State(), [a, b], []
        while not s.is_terminal():
            if not s.tmp_track and not s.wait_for_go and not s.end_turn:
                prog, p, _ = fr.from_state(s)
                starts.append((prog, p))
            s, _ = s.update(agents[s.cur_agent_idx].act(s))
        w = s.get_winner()
        for prog, p in starts:
            progs.append(prog), ms.append(p), ys.append(p == w), gs.append(g0 + g)
    return np.array(progs), np.array(ms), np.array(ys, float), np.array(gs)

def selfplay(v, n, seed, name):
    path = os.path.join(CACHE, name + '.npz')
    if not os.path.exists(path):
        chunks = 16
        jobs = [(partial(SolverAgent, v), n // chunks + (i < n % chunks), seed * 1000 + i, i * 100000) for i in range(chunks)]
        with Pool(8) as pool:
            parts = pool.map(_selfplay, jobs)
        np.savez(path, **{k: np.concatenate([q[i] for q in parts]) for i, k in enumerate(('progs', 'm', 'y', 'g'))})
    d = np.load(path)
    return d['progs'], d['m'], d['y'], d['g']

def feats(progs, m, kind):
    X = np.empty((len(m), 2 if kind == 'claim_top3' else 1))
    for q in (0, 1):
        X[m == q] = features(progs[m == q], q, kind)
    return X

def predict(v, progs, m):
    return 1 / (1 + np.exp(-(feats(progs, m, v.kind) @ v.theta + v.bias)))

def ece(p, y, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))

def fit_round(data, rng_seed=0):
    progs, m, y, g = data
    games = np.unique(g)
    test_games = np.random.default_rng(rng_seed).choice(games, len(games) // 5, replace=False)
    te = np.isin(g, test_games)
    out = {'n_games': int(len(games)), 'n_train': int((~te).sum()), 'n_test': int(te.sum()),
           'base_rate_to_move': float(y.mean()), 'prior_logloss': log_loss(np.full(te.sum(), y[~te].mean()), y[te]), 'fits': {}}
    preds = {}
    for kind in KINDS:
        theta, b = fit(feats(progs[~te], m[~te], kind), y[~te])
        v = VH(kind, theta, b)
        p = predict(v, progs[te], m[te])
        preds[kind] = p
        out['fits'][kind] = {'kind': kind, 'theta': theta.tolist(), 'bias': float(b), 'test_logloss': log_loss(p, y[te]),
                             'test_ece': ece(p, y[te]), 'train_logloss': log_loss(predict(v, progs[~te], m[~te]), y[~te])}
    # game-clustered bootstrap CI of each candidate's log-loss minus the best one's
    best = min(KINDS, key=lambda k: out['fits'][k]['test_logloss'])
    tg, inv = np.unique(g[te], return_inverse=True)
    rng = np.random.default_rng(1)
    ll = {k: -(y[te] * np.log(preds[k]) + (1 - y[te]) * np.log(1 - preds[k])) for k in KINDS}
    sums = {k: np.bincount(inv, ll[k]) for k in KINDS}
    cnt = np.bincount(inv)
    boots = {k: [] for k in KINDS}
    for _ in range(1000):
        s = rng.integers(0, len(tg), len(tg))
        for k in KINDS:
            boots[k].append(sums[k][s].sum() / cnt[s].sum())
    for k in KINDS:
        b = np.array(boots[k])
        out['fits'][k]['test_logloss_ci'] = np.percentile(b, [2.5, 97.5]).tolist()
        out['fits'][k]['diff_vs_best_ci'] = np.percentile(b - np.array(boots[best]), [2.5, 97.5]).tolist()
    out['best'] = best
    out['ranking'] = sorted(KINDS, key=lambda k: out['fits'][k]['test_logloss'])
    return out, te

def reliability(p, y, bins=10):
    idx = np.minimum((p * bins).astype(int), bins - 1)
    rows = []
    for b in range(bins):
        s = idx == b
        if s.sum() >= 20:
            lo, hi = wilson(y[s].sum(), s.sum())
            rows.append({'pred': float(p[s].mean()), 'obs': float(y[s].mean()), 'lo': lo, 'hi': hi, 'n': int(s.sum())})
    return rows

def ev(a, b, n, seed):
    t = time.time()
    p, (lo, hi) = evaluate(a, b, n, seed=seed)
    return {'p': p, 'lo': lo, 'hi': hi, 'n': n, 'wall_s': round(time.time() - t, 1)}

def timing(make_a, make_b, n=10, seed=0):
    from game import Game
    random.seed(seed)
    a, b, solves, t = make_a(), make_b(), 0, time.time()
    for i in range(n):
        a.reset(), b.reset()
        Game([a, b] if i % 2 == 0 else [b, a]).play()
        solves += a.n_solves + getattr(b, 'n_solves', 0)
    dt = time.time() - t
    return {'s_per_game': dt / n, 'solves_per_game': solves / n, 's_per_solve': dt / max(solves, 1)}

def main():
    os.makedirs(CACHE, exist_ok=True)
    res = load()
    res['guess'] = {'kind': GUESS.kind, 'theta': GUESS.theta.tolist(), 'bias': GUESS.bias}
    r1 = stage(res, 'round1', lambda: fit_round(selfplay(GUESS, N_SELF, 1, 'round1'))[0])
    v1 = vh(r1['fits'][r1['best']])
    res['round2_policy'] = r1['fits'][r1['best']]
    d2 = selfplay(v1, N_SELF, 2, 'round2')
    r2 = stage(res, 'round2', lambda: fit_round(d2)[0])
    top = r2['ranking'][:2]
    va, vb = vh(r2['fits'][top[0]]), vh(r2['fits'][top[1]])
    h2h = stage(res, 'h2h', lambda: {'a': top[0], 'b': top[1], **ev(partial(SolverAgent, va), partial(SolverAgent, vb), N_H2H, 3)})
    best_kind = top[1] if h2h['hi'] < 0.5 else top[0]
    best = vh(r2['fits'][best_kind])
    res['best'] = r2['fits'][best_kind]
    save(res)
    S = partial(SolverAgent, best)
    stage(res, 'gate_vs_v2', lambda: ev(S, V2, N_GATE, 4))
    stage(res, 'gate_vs_v2tuned', lambda: ev(S, V2Tuned, N_GATE, 5))
    stage(res, 'ref_v2tuned_vs_v2', lambda: ev(V2Tuned, V2, N_GATE, 4))
    stage(res, 'ref_guess_vs_v2', lambda: ev(partial(SolverAgent, GUESS), V2, N_ABL, 6))
    for name, kw in (('abl_v2_picks', dict(v2_picks=True)), ('abl_v2_gostop', dict(v2_gostop=True)),
                     ('abl_depth2', dict(depth=2)), ('abl_depth4', dict(depth=4))):
        stage(res, name, lambda: ev(partial(SolverAgent, best, **kw), V2, N_ABL, 6))
    stage(res, 'timing', lambda: {
        'exact_vs_v2': timing(S, V2), 'exact_selfplay': timing(S, S),
        'depth2_vs_v2': timing(partial(SolverAgent, best, depth=2), V2),
        'depth4_vs_v2': timing(partial(SolverAgent, best, depth=4), V2)})
    # self-play strength check of calibration-MLE vs a sharper scale (selection by self-play only)
    sharp = VH(best.kind, best.theta * 4 / best.theta[0], 0.0)
    res['sharp'] = {'kind': sharp.kind, 'theta': sharp.theta.tolist(), 'bias': 0.0}
    stage(res, 'h2h_best_vs_guess', lambda: ev(S, partial(SolverAgent, GUESS), N_H2H, 7))
    stage(res, 'h2h_sharp_vs_guess', lambda: ev(partial(SolverAgent, sharp), partial(SolverAgent, GUESS), N_H2H, 7))
    # reliability on held-out round-2 games: fitted best vs the round-0 guess
    _, te = fit_round(d2)
    progs, m, y, _ = d2
    res['reliability'] = {'best': reliability(predict(best, progs[te], m[te]), y[te]),
                          'guess': reliability(predict(GUESS, progs[te], m[te]), y[te]),
                          'best_ece': ece(predict(best, progs[te], m[te]), y[te]),
                          'guess_ece': ece(predict(GUESS, progs[te], m[te]), y[te]),
                          'guess_logloss': log_loss(predict(GUESS, progs[te], m[te]), y[te]),
                          'hist': np.histogram(predict(best, progs[te], m[te]), 20, (0, 1))[0].tolist()}
    save(res)
    plots(res)

SURF, INK, INK2, GRID = '#fcfcfb', '#0b0b0b', '#52514e', '#e4e3df'
S1, S2, NEUTRAL = '#2a78d6', '#eb6834', '#8a8984'

def _ax(ax):
    ax.set_facecolor(SURF)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)

def plots(res):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'text.color': INK, 'axes.labelcolor': INK2, 'figure.facecolor': SURF})
    b = res['best']
    name = {'top3': 'top-3 progress diff', 'sum': 'sum-of-progress diff', 'claim_top3': 'top-3 diff + w·claimed diff'}

    # reliability
    rel = res['reliability']
    fig, (ax, axh) = plt.subplots(2, 1, figsize=(6, 6.6), height_ratios=[4, 1], sharex=True)
    _ax(ax), _ax(axh)
    ax.plot([0, 1], [0, 1], color=NEUTRAL, lw=1, ls='--', zorder=1)
    for key, col, lab in (('guess', S2, f"initial guess c={res['guess']['theta'][0]:g} (ECE {rel['guess_ece']:.3f})"),
                          ('best', S1, f"fitted {name[b['kind']]} (ECE {rel['best_ece']:.3f})")):
        r = rel[key]
        x, o = [q['pred'] for q in r], [q['obs'] for q in r]
        ax.errorbar(x, o, yerr=[[q['obs'] - q['lo'] for q in r], [q['hi'] - q['obs'] for q in r]], color=col, lw=2,
                    marker='o', ms=6, mec=SURF, mew=1.5, capsize=0, elinewidth=1.2, label=lab, zorder=3)
    ax.set_ylabel('observed win rate of player to move')
    ax.set_title('V_h reliability on held-out self-play turn starts', loc='left', fontsize=11, color=INK)
    ax.legend(frameon=False, fontsize=9, loc='upper left', labelcolor=INK2)
    ax.set_xlim(0, 1), ax.set_ylim(0, 1)
    h = np.array(rel['hist'])
    axh.bar(np.linspace(0.025, 0.975, 20), h / h.sum(), width=0.046, color=S1)
    axh.set_xlabel('predicted P(to move wins)')
    axh.set_ylabel('share', fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'reliability.png'), dpi=150)
    plt.close(fig)

    # log-loss by candidate (round 2, held out)
    f = res['round2']['fits']
    ks = sorted(f, key=lambda k: -f[k]['test_logloss'])
    fig, ax = plt.subplots(figsize=(6.4, 2.6))
    _ax(ax)
    ax.grid(axis='y', visible=False)
    for i, k in enumerate(ks):
        lo, hi = f[k]['test_logloss_ci']
        ax.plot([lo, hi], [i, i], color=S1, lw=2, solid_capstyle='round')
        ax.plot(f[k]['test_logloss'], i, 'o', color=S1, ms=8, mec=SURF, mew=1.5)
        ax.annotate(f"{f[k]['test_logloss']:.4f}", (hi, i), xytext=(6, 0), textcoords='offset points', va='center', fontsize=9, color=INK2)
    ax.axvline(res['round2']['prior_logloss'], color=NEUTRAL, ls='--', lw=1)
    ax.set_ylim(-0.7, len(ks) - 0.5)
    ax.annotate('constant prior', (res['round2']['prior_logloss'], -0.5), xytext=(-4, 0), textcoords='offset points',
                ha='right', fontsize=8, color=INK2)
    ax.set_yticks(range(len(ks)), [name[k] for k in ks])
    ax.set_xlabel('held-out log-loss (lower is better), 95% game-bootstrap CI')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'logloss.png'), dpi=150)
    plt.close(fig)

    # ablations vs v2
    rows = [('V2Tuned (reference)', res['ref_v2tuned_vs_v2'], NEUTRAL),
            ('solver, guess V_h', res['ref_guess_vs_v2'], S1),
            ('solver, best V_h (exact)', res['gate_vs_v2'], S1),
            ('solver go/stop + v2 picks', res['abl_v2_picks'], S1),
            ('v2 go/stop + solver picks', res['abl_v2_gostop'], S1),
            ('solver depth k=2', res['abl_depth2'], S1),
            ('solver depth k=4', res['abl_depth4'], S1)][::-1]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    _ax(ax)
    ax.grid(axis='y', visible=False)
    for i, (lab, r, col) in enumerate(rows):
        ax.barh(i, r['p'], height=0.6, color=col)
        ax.plot([r['lo'], r['hi']], [i, i], color=INK, lw=1.2)
        ax.annotate(f"{100 * r['p']:.1f}%  (n={r['n']})", (r['hi'], i), xytext=(5, 0), textcoords='offset points',
                    va='center', fontsize=9, color=INK2)
    ax.axvline(0.5, color=NEUTRAL, ls='--', lw=1)
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlim(0, 1)
    ax.set_xlabel('win rate vs v2, 95% Wilson CI')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'ablations.png'), dpi=150)
    plt.close(fig)

if __name__ == '__main__':
    if sys.argv[1:] == ['plots']:
        plots(load())
    else:
        main()
