import os, sys, json, time, random, argparse
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from collections import OrderedDict
from multiprocessing import Pool
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'analysis')]
import fastrules as fr
from game import State
from solver_agent import SolverAgent, tmp_of
from turn_solver import solve_turn, best_move
from td_train import BEST_VH, _vfn
from warm_bias import ARCH, GO, STOP, boot_weights, cluster_sums, stat

# Rollout policy-improvement headroom of one-turn play. Games: SolverAgent(X) vs SolverAgent(H). At X's contested decisions
# Q(s, a) = P(X wins) after a, X then playing for itself and H for the opponent (CRN: per-(rollout, turn) dice streams),
# truncated K full turns per player after the current turn and scored by the mean of H, T, S (terminal: 0 / 1)
BASES = ('H', 'T', 'S')
SEED0, DELTA = 40000, 0.03
OUT = os.path.join(ROOT, 'runs', 'headroom')
MAXROLLS = 100  # > any turn's rolls (each advances >= 1 of 86 cells)

V = {}

def init(srcs):
    V.update({k: _vfn(s, keep=True) for k, s in srcs.items()})
    V['E'] = lambda progs, m: sum(V[k](progs, m) for k in BASES) / 3

def phase_x(prog):
    return float((prog[:, 2:] / fr.LEN[2:]).sum())

# ---- decision collection
def game_job(args):
    # one game, X at seat g % 2; at X's decisions (>1 distinct action) records candidates, X's solved values and the
    # H / T / S solvers' choices (pick = resulting in-turn progress key, as in warm_bias)
    X, g, seed = args
    agents, opp, px = {k: SolverAgent(V[k]) for k in BASES}, SolverAgent(V['H']), g % 2
    random.seed(seed)
    s, decs = State(), []
    while not s.is_terminal():
        legal = s.get_legal_actions()
        if s.cur_agent_idx != px or len(legal) == 1:
            s, _ = s.update(opp.act(s) if s.cur_agent_idx != px else legal[0])
            continue
        go = s.is_wait_for_go()
        kf = (lambda a: GO if a.is_go() else STOP) if go else (lambda a: int(fr.key(tmp_of(*s.unsafe_pick(a)))))
        acts = {k: a.act(s) for k, a in agents.items()}
        prog, p, tmp = fr.from_state(s)
        sol = agents[X].sol
        i = sol.row(tmp)
        cand = [GO, STOP] if go else sorted({kf(a) for a in legal})
        if len(cand) > 1 and not np.isnan(sol.go[i]):
            val = [sol.go[i], sol.stop[i]] if go else [sol.value(fr.decode(k)) for k in cand]
            ch = {k: kf(a) for k, a in acts.items()}
            top = np.sort(val)[::-1]
            decs.append(dict(game=g, p=p, prog=prog.tolist(), tmp=tmp.tolist(), pick=not go, cand=cand,
                             val=[float(v) for v in val], ch=ch, close=bool(top[0] - top[1] < DELTA),
                             disagree=len(set(ch.values())) > 1, phase=phase_x(prog)))
        s, _ = s.update(acts[X])
    return decs

# ---- rollouts
class Roller:
    # Q rollouts for one decision; memo: (board, rollout, turn) -> values per K, so continuations shared across actions
    # (CRN) are computed once; LRU of turn solutions
    def __init__(self, prog, p, Ks, seed, vx, vo, ev):
        self.prog, self.p, self.Ks, self.seed, self.vf, self.ev = np.array(prog, np.int8), p, Ks, seed, (vx, vo), ev
        self.tk = np.array([np.inf if k is None else 2 * k + 1 for k in Ks])
        self.memo, self.sols, self.n_turns = {}, OrderedDict(), 0

    def sol(self, board, mover):
        key = (board.tobytes(), mover)
        if key not in self.sols:
            self.sols[key] = solve_turn(board, mover, self.vf[mover != self.p])
            if len(self.sols) > 16:
                self.sols.popitem(last=False)
        self.sols.move_to_end(key)
        return self.sols[key]

    def rolls(self, j, t):
        return np.random.default_rng([self.seed, j, t]).choice(fr.N_ROLLS, MAXROLLS, p=fr.ROLL_PROB)

    def turn(self, sol, board, mover, tmp, act, rolls):
        # finishes mover's turn after act (GO: roll next, STOP: bank, else decide go/stop at tmp) -> (banked board, eps):
        # eps = realized - expected mover's stop value under its own solve, exactly mean 0 (the control variate)
        self.n_turns += 1
        if act == STOP:
            return fr.apply_stop(board, mover, tmp), 0.0
        it = iter(rolls)
        if act != GO:
            tmp = fr.decode(act)
        e = sol.go[sol.row(tmp)] if act == GO else sol.best[sol.row(tmp)]
        while True:
            if act != GO:
                if fr.stop_wins(board, mover, tmp) or not sol.should_go(tmp):
                    return fr.apply_stop(board, mover, tmp), sol.stop[sol.row(tmp)] - e
            act = None
            mv = best_move(sol, tmp, next(it))
            if mv is None:
                return board, sol.bust_val - e
            tmp = mv[1]

    def cont(self, board, j, t):
        # (values, CV sums) per K, X's view, from the start of turn t (odd: opponent to move)
        key = (board.tobytes(), j, t)
        if key in self.memo:
            return self.memo[key]
        mover = self.p if t % 2 == 0 else 1 - self.p
        w = [q for q in (0, 1) if fr.score(board, q) >= 3]
        v, c = np.full(len(self.Ks), np.nan), np.zeros(len(self.Ks))
        if w:
            v[:] = float(w[0] == self.p)
        else:
            at, on = self.tk == t, self.tk > t
            if at.any():
                e = float(self.ev(board[None], mover)[0])
                v[at] = e if mover == self.p else 1 - e
            if on.any():
                nxt, eps = self.turn(self.sol(board, mover), board, mover, np.zeros(13, np.int8), GO, self.rolls(j, t))
                rv, rc = self.cont(nxt, j, t + 1)
                v[on], c[on] = rv[on], (eps if mover == self.p else -eps) + rc[on]
        self.memo[key] = v, c
        return v, c

    def q(self, tmp, act, j):
        # -> (raw, control-variate corrected) values per K
        b, eps = self.turn(self.sol(self.prog, self.p), self.prog, self.p, np.array(tmp, np.int8), act, self.rolls(j, 0))
        v, c = self.cont(b, j, 1)
        return v, v - c - eps

def decision_job(args):
    # -> Q (raw), Qcv (control variate) [K, actions, R]
    d, X, Ks, R, seed = args
    t0 = time.time()
    r = Roller(d['prog'], d['p'], Ks, seed, V[X], V['H'], V['E'])
    Q = np.array([[r.q(d['tmp'], a, j) for j in range(R)] for a in d['cand']]).transpose(2, 3, 0, 1)
    return dict(id=d['id'], Q=np.round(Q[0], 6).tolist(), Qcv=np.round(Q[1], 6).tolist(), sec=time.time() - t0, turns=r.n_turns)

def regrets(Q, ci):
    # Q [actions, R], ci = X's choice -> naive and split-sample (cross-fitted halves) regret, MC SE, a* (all rollouts)
    m = Q.mean(1)
    a = int(np.argmax(m))
    A, B = Q[:, 0::2], Q[:, 1::2]
    split = 0.5 * ((B[A.mean(1).argmax()] - B[ci]).mean() + (A[B.mean(1).argmax()] - A[ci]).mean())
    def sig(F, G):
        # switch away from X's choice only if the selection half's margin is > 2 SE
        b, d = F.mean(1).argmax(), F[F.mean(1).argmax()] - F[ci]
        return (G[b] - G[ci]).mean() if d.mean() > 2 * d.std(ddof=1) / np.sqrt(len(d)) else 0.0
    return dict(naive=float(m[a] - m[ci]), split=float(split), sig=float(0.5 * (sig(A, B) + sig(B, A))), se=float((Q[a] - Q[ci]).std(ddof=1) / np.sqrt(Q.shape[1])),
                best=a, x_best=bool(m[ci] >= m[a] - 1e-12))

# ---- stages
def collect(pool, n_games):
    path = os.path.join(OUT, 'decisions.json')
    if os.path.exists(path):
        D = json.load(open(path))
        if D['n_games'] == n_games:
            return D
    D = dict(n_games=n_games, seeds={X: SEED0 + b * n_games for b, X in enumerate(BASES)}, dec={})
    for X in BASES:
        res = pool.map(game_job, [(X, g, D['seeds'][X] + g) for g in range(n_games)], chunksize=1)
        D['dec'][X] = [dict(d, id=f'{X}{i}') for i, d in enumerate(x for r in res for x in r)]
    json.dump(D, open(path, 'w'))
    return D

def sample(D, X, n, contested, seed):
    pool = [d for d in D['dec'][X] if (d['close'] or d['disagree']) == contested]
    idx = np.random.default_rng(seed).choice(len(pool), min(n, len(pool)), replace=False)
    return [pool[i] for i in sorted(idx)]

def run_rollouts(pool, jobs, path):
    # appends one line per decision; resumes by skipping ids already in path
    done = {json.loads(l)['id'] for l in open(path)} if os.path.exists(path) else set()
    todo = [j for j in jobs if j[0]['id'] not in done]
    t0 = time.time()
    with open(path, 'a') as f:
        for i, r in enumerate(pool.imap_unordered(decision_job, todo, chunksize=1)):
            f.write(json.dumps(r) + '\n')
            f.flush()
            if i % 25 == 0:
                print(f'{path}: {i + 1}/{len(todo)} {time.time() - t0:.0f}s', flush=True)
    return {r['id']: r for r in map(json.loads, open(path))}

def jobs_for(decs, X, Ks, R):
    return [(d, X, Ks, R, SEED0 * 1000 + int(d['id'][1:]) * 3 + BASES.index(X)) for d in decs]

def pilot(pool, D, a):
    Ks, names = [2, 3, None], ['2', '3', 'full']
    sel = {X: sample(D, X, a.pilot_n, True, 1) + sample(D, X, a.pilot_unc, False, 2) for X in BASES}
    res = run_rollouts(pool, [j for X in BASES for j in jobs_for(sel[X], X, Ks, a.pilot_R)], os.path.join(OUT, 'pilot.jsonl'))
    out, sc = dict(Ks=names, R=a.pilot_R, bases={}), []
    for X in BASES:
        rows = dict(contested=[], uncontested=[])
        for d in sel[X]:
            r = res[d['id']]
            ci = d['cand'].index(d['ch'][X])
            Q = {src: np.array(r[f]) for src, f in (('cv', 'Qcv'), ('raw', 'Q'))}
            rows['contested' if d['close'] or d['disagree'] else 'uncontested'].append(dict(
                reg={src: {k: regrets(q[i], ci) for i, k in enumerate(names)} for src, q in Q.items()},
                delta=np.delete(Q['cv'] - Q['cv'][:, ci:ci + 1], ci, 1).mean(2).T, qmean=Q['cv'].mean(2).T, q_sd={src: q.std(2).mean() for src, q in Q.items()}))
            sc.append(r['sec'] / (Q['cv'].shape[1] * Q['cv'].shape[2]))
        e = {}
        for kind, rs in rows.items():
            qm = np.concatenate([x['qmean'] for x in rs])  # [actions, K]
            dl = np.concatenate([x['delta'] for x in rs])  # Q(a) - Q(X's choice), a != choice [pairs, K]
            m = lambda f: float(np.mean([f(x) for x in rs]))
            e[kind] = dict(n=len(rs), rollout_sd={src: m(lambda x: x['q_sd'][src]) for src in ('cv', 'raw')}, **{
                f'regret_{src}_{k}': dict(split=m(lambda x: x['reg'][src][k]['split']), naive=m(lambda x: x['reg'][src][k]['naive']),
                                          mc_se_mean=m(lambda x: x['reg'][src][k]['se']),
                                          se_of_mean=float(np.std([x['reg'][src][k]['split'] for x in rs], ddof=1) / np.sqrt(len(rs))))
                for src in ('cv', 'raw') for k in names},
                bias_q={k: dict(mean=float((qm[:, i] - qm[:, 2]).mean()), mean_abs=float(np.abs(qm[:, i] - qm[:, 2]).mean()),
                                se=float((qm[:, i] - qm[:, 2]).std(ddof=1) / np.sqrt(len(qm)))) for i, k in enumerate(names[:2])},
                bias_delta={k: dict(mean=float((dl[:, i] - dl[:, 2]).mean()), se=float((dl[:, i] - dl[:, 2]).std(ddof=1) / np.sqrt(len(dl))),
                                    n=len(dl)) for i, k in enumerate(names[:2])},
                bias_regret={k: {est: m(lambda x: x['reg']['cv'][k][est] - x['reg']['cv']['full'][est]) for est in ('split', 'naive')}
                             for k in names[:2]},
                qmean_pairs=qm.round(5).tolist())
        e['c'] = sum(d['close'] or d['disagree'] for d in D['dec'][X]) / D['n_games']
        out['bases'][X] = e
    out['worker_sec_per_rollout_full'] = float(np.mean(sc))
    json.dump(out, open(os.path.join(OUT, 'pilot.json'), 'w'), indent=1)
    for X in BASES:
        c = out['bases'][X]['contested']
        print(X, 'c', round(out['bases'][X]['c'], 2), 'delta bias', c['bias_delta'], 'sd', c['rollout_sd'],
              {k: (round(c[f'regret_cv_{k}']['split'], 4), round(c[f'regret_cv_{k}']['naive'], 4), round(c[f'regret_cv_{k}']['mc_se_mean'], 4))
               for k in names}, 'biasQ', {k: round(v['mean'], 4) for k, v in c['bias_q'].items()})

def summarize(D, sel, res, K, ki, B):
    # per base: headroom/game = c x mean regret, game-clustered bootstrap over the n_games games
    allx = [d for X in BASES for d in D['dec'][X] if d['close'] or d['disagree']]
    cuts = np.quantile([d['phase'] for d in allx], [1 / 3, 2 / 3]).tolist()
    n = D['n_games']
    W = boot_weights([n], B)
    out = dict(K=K, n_games=n, delta=DELTA, phase_cuts=cuts, evaluator='mean(H, T, S)', control_variate='per turn: realized - solver-expected stop value of the mover under its own V (exactly mean 0), coefficient 1', seeds=D['seeds'], bases={})
    hs = []
    for X in BASES:
        dd, rr = sel[X], []
        for d in dd:
            Q = np.array(res[d['id']]['Qcv'])[ki]
            rr.append(regrets(Q, d['cand'].index(d['ch'][X])))
            g = regrets(np.array(res[d['id']]['Q'])[ki], d['cand'].index(d['ch'][X]))
            rr[-1].update(split_raw=g['split'], naive_raw=g['naive'])
            ok = Q.mean(1)
            for k in ('H', 'S'):
                rr[-1][f'q_{k}'] = float(ok[d['cand'].index(d['ch'][k])])
            rr[-1]['q_best'] = float(ok.max())
            if d['ch']['H'] != d['ch']['S']:
                hs.append((X, d, rr[-1]))
        cl = np.array([d['game'] for d in dd])
        ncont = np.bincount([d['game'] for d in D['dec'][X] if d['close'] or d['disagree']], minlength=n).astype(float)
        one = np.ones(len(dd))
        ph = np.digitize([d['phase'] for d in dd], cuts)
        subsets = dict(all=one > 0, gostop=np.array([not d['pick'] for d in dd]), pick=np.array([d['pick'] for d in dd]),
                       close=np.array([d['close'] for d in dd]), disagree=np.array([d['disagree'] for d in dd]),
                       early=ph == 0, mid=ph == 1, late=ph == 2)
        e = dict(n_contested_sampled=len(dd), n_contested_total=int(ncont.sum()),
                 n_decisions_total=len(D['dec'][X]), c=stat(ncont[:, None], W, lambda s: s[0] / n))
        for est in ('split', 'sig', 'naive', 'split_raw', 'naive_raw'):
            r = np.array([x[est] for x in rr])
            for name, m in (subsets if est in ('split', 'sig', 'naive') else dict(all=subsets['all'])).items():
                A = np.c_[cluster_sums([r * m, m.astype(float), one], cl, n), ncont]
                e.setdefault(name, dict(n=int(m.sum())))[est] = dict(
                    regret=stat(A, W, lambda s: s[0] / s[1]), headroom=stat(A, W, lambda s: s[0] / s[2] * s[3] / n))
        A = cluster_sums([np.array([x['x_best'] for x in rr], float), one], cl, n)
        e['frac_x_best'] = stat(A, W, lambda s: s[0] / s[1])
        e['mc_se_mean'] = float(np.mean([x['se'] for x in rr]))
        out['bases'][X] = e
        lo, mid, hi = (band(max(0.0, e['all'][k]['headroom']['est'])) for k in ('split', 'sig', 'naive'))
        e['verdict'] = dict(lower_split=lo, middle_sig=mid, upper_naive=hi)
    # H vs S on their disagreements: a* agreement and regret (clustered by base x game)
    gid = np.array([BASES.index(X) * n + d['game'] for X, d, _ in hs])
    W2 = boot_weights([3 * n], B)
    one = np.ones(len(hs))
    aH, aS = (np.array([r[f'q_{k}'] >= r['q_best'] - 1e-12 for *_, r in hs], float) for k in 'HS')
    rH = np.array([r['q_best'] - r['q_H'] for *_, r in hs])
    rS = np.array([r['q_best'] - r['q_S'] for *_, r in hs])
    A = cluster_sums([aH, aS, rH, rS, rH - rS, one], gid, 3 * n)
    out['h_vs_s'] = dict(n=len(hs), n_by_base={X: sum(x[0] == X for x in hs) for X in BASES},
                         H_is_best=stat(A, W2, lambda s: s[0] / s[5]), S_is_best=stat(A, W2, lambda s: s[1] / s[5]),
                         regret_H=stat(A, W2, lambda s: s[2] / s[5]), regret_S=stat(A, W2, lambda s: s[3] / s[5]),
                         regret_H_minus_S=stat(A, W2, lambda s: s[4] / s[5]),
                         note='control-variate Q; regret vs a* from all rollouts (naive); H - S difference needs no a*')
    return out

def band(h):
    return ('<1 pt: value-level gains essentially exhausted' if h < 0.01 else
            '1-3 pts: real but small' if h <= 0.03 else '>3 pts: better values clearly matter')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['pilot', 'full'])
    ap.add_argument('--T', default=os.path.join(ARCH, 'cant_stop_td30/runs/td30/ckpt/it0082.pt'))
    ap.add_argument('--S', default=os.path.join(ARCH, 'cant_stop_scratch/runs/scratch30/ckpt/it0297.pt'))
    ap.add_argument('--games', type=int, default=100, help='games per base (seeds 40000 + base * games + g)')
    ap.add_argument('--pilot_n', type=int, default=50)
    ap.add_argument('--pilot_unc', type=int, default=17, help='uncontested decisions per base in the pilot')
    ap.add_argument('--pilot_R', type=int, default=64)
    ap.add_argument('--n', type=int, default=400, help='contested decisions per base (full run)')
    ap.add_argument('--R', type=int, default=64)
    ap.add_argument('--K', type=int, default=2, help='primary truncation; full-length values come from the same rollouts')
    ap.add_argument('--boot', type=int, default=2000)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    srcs = dict(H=BEST_VH, T=a.T, S=a.S)
    with Pool(a.workers, initializer=init, initargs=(srcs,)) as pool:
        t0 = time.time()
        D = collect(pool, a.games)
        print({X: (len(D['dec'][X]), sum(d['close'] or d['disagree'] for d in D['dec'][X])) for X in BASES},
              f'collect {time.time() - t0:.0f}s', flush=True)
        if a.stage == 'pilot':
            pilot(pool, D, a)
        elif a.stage == 'full':
            sel = {X: sample(D, X, a.n, True, 3) for X in BASES}
            path = os.path.join(OUT, f'full_K{a.K}_R{a.R}.jsonl')
            res = run_rollouts(pool, [j for X in BASES for j in jobs_for(sel[X], X, [a.K, None], a.R)], path)
            out = summarize(D, sel, res, a.K, 0, a.boot)
            out['full_length'] = summarize(D, sel, res, 'full', 1, a.boot)
            out.update(R=a.R, sources={k: os.path.relpath(v, ROOT) if isinstance(v, str) else repr(v) for k, v in srcs.items()},
                       summary_seconds=round(time.time() - t0), uncontested='regret 0 by assumption (checked in pilot.json)',
                       estimators='split: cross-fitted split-sample (biased low: a noisy a* is often worse than X); naive: same-sample '
                                  '(biased high, winner\'s curse); sig: cross-fitted, switch from X only when the selection half margin '
                                  '> 2 SE (less biased middle); true headroom >= 0')
            json.dump(out, open(os.path.join(OUT, 'results.json'), 'w'), indent=1)
            for X in BASES:
                e = out['bases'][X]
                print(X, 'c', e['c'], 'headroom', e['all']['split']['headroom'], 'naive', e['all']['naive']['headroom']['est'])
            print('H vs S', out['h_vs_s'])

if __name__ == '__main__':
    main()
