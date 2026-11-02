import os, sys, json, glob, time, random, argparse
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from multiprocessing import Pool
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import fastrules as fr
from game import State
from solver_agent import SolverAgent, tmp_of
from td_train import BEST_VH, ZERO
from branching import load_sets

# How far do V_h (H) and the warm-started td30 (T) deviate from scratch30's final net (S, the V_h-unbiased reference)?
# Positions: turn starts of V_h games (P_H, the branch_diag visited set) and of fresh T / S self-play games; decision points
# are every go/stop and pick faced in those games. All CIs: game-clustered bootstrap, stratified by set
ARCH = os.path.join(os.path.dirname(ROOT), 'cant_stop_archive')
NAMES = ('S', 'H', 'T', 'S74', 'S208', 'S280')
XS = NAMES[1:]
SEEDS = dict(P_H=10000, P_T=30000, P_S=30100)
GO, STOP = -1, -2

def game_job(args):
    # one game played by srcs[player]; every value fn in srcs solves each turn start and acts on each real decision.
    # Returns turn starts (prog, p, raw [k], solved [k]) and decisions (turn, is_pick, choice keys [k], regret under S [k])
    srcs, player, seed = args
    names = list(srcs)
    vs = [_vfn_multi(srcs[n]) for n in names]
    agents = [SolverAgent(v) for v in vs]
    random.seed(seed)
    s, starts, decs = State(), [], []
    while not s.is_terminal():
        if not s.tmp_track and not s.wait_for_go and not s.end_turn:
            prog, p, _ = fr.from_state(s)
            starts.append((prog, p, [float(v(prog[None], p)[0]) for v in vs], [a.solution(prog, p, ZERO).value(ZERO) for a in agents]))
        legal = s.get_legal_actions()
        if len(legal) > 1:
            go = s.is_wait_for_go()
            kf = (lambda a: GO if a.is_go() else STOP) if go else (lambda a: fr.key(tmp_of(*s.unsafe_pick(a))))
            ch = [kf(a.act(s)) for a in agents]
            sol, (_, _, tmp) = agents[0].sol, fr.from_state(s)  # agents[0] is S
            i = sol.row(tmp)
            if not go and np.isnan(sol.go[i]):  # going on from a winning stop: every pick wins
                reg = [0.0] * len(ch)
            else:
                val = {GO: sol.go[i], STOP: sol.stop[i]} if go else {kf(a): sol.value(tmp_of(*s.unsafe_pick(a))) for a in legal}
                reg = [0.0 if np.isnan(val[c]) else float(val[ch[0]] - val[c]) for c in ch]
            decs.append((len(starts) - 1, not go, ch, reg))
        act = agents[names.index(player)].act(s) if len(legal) > 1 else legal[0]
        s, _ = s.update(act)
    return starts, decs

_vc = {}

def _vfn_multi(src):
    # td_train._vfn keeps one net per worker; here several are live at once
    if isinstance(src, str) and src not in _vc:
        from value_net import NetV
        _vc[src] = NetV.load(src).fused()
    return _vc[src] if isinstance(src, str) else src

def collect(pool, srcs, games):
    # -> per set: turn-start arrays and decision arrays, with game ids
    out = {}
    for name, player in (('P_H', 'H'), ('P_T', 'T'), ('P_S', 'S')):
        seeds = range(SEEDS[name], SEEDS[name] + games)
        res = pool.map(game_job, [(srcs, player, g) for g in seeds], chunksize=1)
        st = [(g,) + r for g, (s, _) in enumerate(res) for r in s]
        dc = [(g,) + r for g, (_, d) in enumerate(res) for r in d]
        out[name] = dict(game=np.array([r[0] for r in st]), prog=np.array([r[1] for r in st], np.int8),
                         p=np.array([r[2] for r in st]), raw=np.array([r[3] for r in st]), sol=np.array([r[4] for r in st]),
                         d_game=np.array([r[0] for r in dc]), d_pick=np.array([r[2] for r in dc]), d_choice=np.array([r[3] for r in dc]),
                         d_regret=np.array([r[4] for r in dc]))
    return out

def select_ph(d, n=900, seed0=SEEDS['P_H']):
    # branch_diag / branching.build_sets' visited sample of the V_h games' turn starts
    return np.random.default_rng(seed0).choice(len(d['p']), min(n, len(d['p'])), replace=False)

def boot_weights(groups, B, seed=0):
    # groups: per set, number of games. Stratified cluster bootstrap -> W [B, total games] resample counts
    rng, W = np.random.default_rng(seed), []
    for n in groups:
        W.append(np.stack([np.bincount(rng.integers(n, size=n), minlength=n) for _ in range(B)]))
    return np.concatenate(W, 1).astype(np.float64)

def cluster_sums(cols, cl, n):
    return np.stack([np.bincount(cl, c, n) for c in cols], 1)

def stat(A, W, f):
    # f(column sums) -> point estimate, 95% CI over bootstrap resamples
    pt, bs = f(A.sum(0)), f((W @ A).T)
    return dict(est=float(pt), ci=[float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))])

def position_metrics(V, cl, n, W):
    # V: {name: values [m]} at positions with game cluster ids cl
    S, H, T, S74 = V['S'], V['H'], V['T'], V['S74']
    one = np.ones_like(S)
    r = {}
    x = H - S
    for k in NAMES:
        d = V[k] - S
        A = cluster_sums([np.abs(d), d * d, d, one, d * x, x * x], cl, n)
        r[k] = dict(mean_abs=stat(A, W, lambda s: s[0] / s[3]), rms=stat(A, W, lambda s: np.sqrt(s[1] / s[3])),
                    signed=stat(A, W, lambda s: s[2] / s[3]), beta=stat(A, W, lambda s: s[4] / s[5]))
    y = T - S
    A = cluster_sums([x * y, x * x, np.abs(y), np.abs(x), np.abs(S74 - S), one], cl, n)
    r['beta'] = stat(A, W, lambda s: s[0] / s[1])
    r['ratio_T_H'] = stat(A, W, lambda s: s[2] / s[3])
    r['diff_T_S74'] = stat(A, W, lambda s: (s[2] - s[4]) / s[5])
    A = cluster_sums([x * x, x * x], cl, n)
    r['beta_HH'] = float(A[:, 0].sum() / A[:, 1].sum())  # sanity: T := H gives 1
    return r

def decision_metrics(D, n_turns_per_game, n, W):
    cl, pick = D['d_game'], D['d_pick']
    tg = cluster_sums([n_turns_per_game.astype(float)], np.arange(n), n)
    r, ch = {}, D['d_choice']
    for j, x in enumerate(NAMES):
        dis, reg = (ch[:, j] != ch[:, 0]).astype(float), D['d_regret'][:, j]
        e = {}
        for kind, m in (('gostop', ~pick), ('pick', pick), ('all', np.ones_like(pick))):
            A = cluster_sums([dis * m, reg * m, m.astype(float)], cl, n)
            e[kind] = dict(n=int(m.sum()), disagree=stat(A, W, lambda s: s[0] / s[2]), regret=stat(A, W, lambda s: s[1] / s[2]))
            A = np.c_[A, tg]
            e[kind]['regret_per_turn'] = stat(A, W, lambda s: s[1] / s[3])
        r[x] = e
    reg = D['d_regret']
    A = np.c_[cluster_sums([reg[:, NAMES.index('T')] - reg[:, NAMES.index('S74')]], cl, n), tg]
    r['diff_T_S74_per_turn'] = stat(A, W, lambda s: s[0] / s[1])
    return r

def analyze(data, B):
    # per set and pooled (concatenated, games re-indexed)
    sets = list(data)
    out = dict(positions=dict(raw={}, solved={}), decisions={}, n={})
    parts = {}
    for k in sets:
        d = data[k]
        ng = int(d['game'].max()) + 1
        idx = d['sel'] if 'sel' in d else np.arange(len(d['p']))
        parts[k] = (d, ng, idx)
        out['n'][k] = dict(games=ng, positions=len(idx), decisions=len(d['d_game']), turns=len(d['p']))
    for k in sets + ['pooled']:
        ks = sets if k == 'pooled' else [k]
        off = np.cumsum([0] + [parts[q][1] for q in ks])
        n, W = int(off[-1]), boot_weights([parts[q][1] for q in ks], B)
        cl = np.concatenate([parts[q][0]['game'][parts[q][2]] + o for q, o in zip(ks, off)])
        for kind, col in (('raw', 'raw'), ('solved', 'sol')):
            V = {x: np.concatenate([parts[q][0][col][parts[q][2], j] for q in ks]) for j, x in enumerate(NAMES)}
            out['positions'][kind][k] = position_metrics(V, cl, n, W)
        D = {f: np.concatenate([parts[q][0][f] for q in ks]) for f in ('d_pick', 'd_choice', 'd_regret')}
        D['d_game'] = np.concatenate([parts[q][0]['d_game'] + o for q, o in zip(ks, off)])
        tpg = np.concatenate([np.bincount(parts[q][0]['game'], minlength=parts[q][1]) for q in ks])
        out['decisions'][k] = decision_metrics(D, tpg, n, W)
    return out

def verdict(res):
    P, Dp = res['positions'], res['decisions']['pooled']
    c1 = all(P[k]['pooled']['beta']['ci'][1] < 0.2 for k in ('raw', 'solved'))
    c2 = all(P[k]['pooled']['diff_T_S74']['ci'][0] <= 0 for k in ('raw', 'solved'))
    c3 = Dp['diff_T_S74_per_turn']['ci'][0] <= 0
    v = 'negligible' if c1 and c2 and c3 else 'imprint but decision-irrelevant' if c2 and c3 else 'not negligible'
    return dict(c1_beta_upper_lt_0_2=c1, c2_dT_le_dS74=c2, c3_regret_T_le_S74=c3, verdict=v)

RULE = ('Pre-registered (pooled positions; (1),(2) must hold for both raw V and one-turn solved value): '
        '(1) beta 95% CI upper < 0.2; (2) d(T) <= d(S74), CI-aware: holds unless d(T)-d(S74) is significantly > 0 '
        '(paired bootstrap CI lower > 0); (3) T regret/turn under S <= S74, CI-aware the same way. '
        'negligible iff (1)(2)(3); "imprint but decision-irrelevant" iff (2)(3) but not (1); else "not negligible".')

def trend(ph, run_dirs, every):
    # forward passes on P_H: mean |V - S| and |V - H| per snapshot
    from value_net import NetV
    prog, p = ph['prog'][ph['sel']], ph['p'][ph['sel']]
    S, H = ph['raw'][ph['sel'], 0], ph['raw'][ph['sel'], 1]
    out = {}
    for run, d in run_dirs.items():
        cs = sorted(glob.glob(os.path.join(d, 'it*.pt')))
        cs = cs[::every] + ([cs[-1]] if (len(cs) - 1) % every else [])
        rows = []
        for c in cs:
            v = NetV.load(c)
            x = np.empty(len(p))
            for q in (0, 1):
                x[p == q] = v(prog[p == q], q)
            rows.append([int(os.path.basename(c)[2:6]), float(np.abs(x - S).mean()), float(np.abs(x - H).mean())])
        out[run] = rows
    return out

def main():
    ap = argparse.ArgumentParser()
    rel = lambda *a: os.path.join(ARCH, *a)
    ap.add_argument('--S', default=rel('cant_stop_scratch/runs/scratch30/ckpt/it0297.pt'))
    ap.add_argument('--T', default=rel('cant_stop_td30/runs/td30/ckpt/it0082.pt'))
    ap.add_argument('--S74', default=rel('cant_stop_scratch/runs/scratch30/ckpt/it0074.pt'))
    ap.add_argument('--S208', default=rel('cant_stop_scratch/runs/scratch30/ckpt/it0208.pt'))
    ap.add_argument('--S280', default=rel('cant_stop_scratch/runs/scratch30/ckpt/it0280.pt'))
    ap.add_argument('--scratch_ckpt', default=rel('cant_stop_scratch/runs/scratch30/ckpt'))
    ap.add_argument('--td_ckpt', default=rel('cant_stop_td30/runs/td30/ckpt'))
    ap.add_argument('--r7', default=os.path.join(ROOT, 'runs/branch60/r7_sets.npz'), help='P_H check: branch_diag visited set')
    ap.add_argument('--games', type=int, default=100)
    ap.add_argument('--n_ph', type=int, default=900)
    ap.add_argument('--every', type=int, default=10)
    ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--out', default=os.path.join(ROOT, 'runs', 'warm_bias'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    srcs = dict(S=a.S, H=BEST_VH, T=a.T, S74=a.S74, S208=a.S208, S280=a.S280)
    t0 = time.time()
    with Pool(a.workers) as pool:
        data = collect(pool, srcs, a.games)
    print(f'games done {time.time() - t0:.0f}s', flush=True)
    ph = data['P_H']
    ph['sel'] = select_ph(ph, a.n_ph)
    if a.games == 100 and a.n_ph == 900 and os.path.exists(a.r7):
        vb, vm = load_sets(a.r7)['visited'][:2]
        assert np.array_equal(ph['prog'][ph['sel']], vb) and np.array_equal(ph['p'][ph['sel']], vm), 'P_H != branch_diag set'
    np.savez_compressed(os.path.join(a.out, 'data.npz'), names=np.array(NAMES),
                        **{f'{k}__{f}': v for k, d in data.items() for f, v in d.items()})
    res = analyze(data, a.boot)
    res.update(rule=RULE, **verdict(res), names=list(NAMES), seeds=SEEDS, games=a.games,
               sources={k: os.path.relpath(v, ROOT) if isinstance(v, str) else repr(v)
                        for k, v in srcs.items()})
    res['trend'] = trend(ph, dict(scratch30=a.scratch_ckpt, td30=a.td_ckpt), a.every)
    res['seconds'] = round(time.time() - t0, 1)
    json.dump(res, open(os.path.join(a.out, 'results.json'), 'w'), indent=1)
    P, Dp = res['positions'], res['decisions']['pooled']
    for k in ('raw', 'solved'):
        print(k, {x: round(P[k]['pooled'][x]['mean_abs']['est'], 4) for x in NAMES}, 'beta', P[k]['pooled']['beta'])
    print({x: (round(Dp[x]['all']['disagree']['est'], 3), round(Dp[x]['all']['regret_per_turn']['est'], 5)) for x in NAMES})
    print(res['verdict'], f"{res['seconds']}s")

if __name__ == '__main__':
    main()
