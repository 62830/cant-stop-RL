# Phase 4.5 speed benchmark: legacy (bench/legacy.py) vs current solver; writes runs/phase45/bench.json
# usage: python bench/speed.py [--parts stages,solve,parallel,selfplay] [--games 32]
import os, sys, json, time, argparse
import multiprocessing as mp
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'bench')]
import fastrules as fr
import turn_solver as ts
import legacy
from heuristics import VH

NET = os.path.join(ROOT, 'runs', 'phase4', 'it0041.pt')
VHB = VH('claim_top3', (0.6431, 0.6591), 0.2690)
EMPTY = (np.zeros((2, 13), np.int8), 0)

def mid_board():
    sys.path.insert(0, os.path.join(ROOT, 'tests'))
    from test_solver import _mid
    return _mid()

def net(old):
    from value_net import NetV
    v = NetV.load(NET)
    return v if old else v.fused()

def best_of(f, reps):
    ts_ = []
    for _ in range(reps):
        t = time.perf_counter()
        f()
        ts_.append(time.perf_counter() - t)
    return 1000 * min(ts_)

def stages(prog, p, reps=5):
    # per-stage ms (min over reps) on one board, both implementations, net value
    lo, nw = net(True), net(False)
    keys, dep, idx = legacy.enumerate_turn(prog, p)
    boards, win = legacy.stop_boards(prog, p, keys)
    ask = np.concatenate([boards[~win & (keys != 0)], prog[None]])
    stop, bv = legacy.stop_values(prog, p, lo, keys)
    old = dict(enumerate=best_of(lambda: legacy.enumerate_turn(prog, p), reps), stop_boards=best_of(lambda: legacy.stop_boards(prog, p, keys), reps),
               value=best_of(lambda: lo(ask, 1 - p), reps), fill=best_of(lambda: legacy.fill(prog, p, keys, dep, idx, stop, bv, -1), reps))
    keys, dep, idx, ch = fr.enumerate_turn(prog, p, children=True)
    stop, bv = ts.stop_values(prog, p, nw, keys)
    new = dict(enumerate=best_of(lambda: fr.enumerate_turn(prog, p), reps), value=best_of(lambda: ts.stop_values(prog, p, nw, keys), reps),
               fill=best_of(lambda: ts._fill(prog, p, keys, dep, ch, stop, bv, -1), reps))
    return dict(situations=len(keys), old=old, new=new)

def cold(*args):
    # a solve without the turn-graph cache (it would hit on repeats of one board)
    ts._graphs.clear()
    return ts.solve_turn(*args)

def solves(prog, p, reps=3):
    out = {}
    for name, vo, vn in (('vh', VHB, VHB), ('net', net(True), net(False))):
        out[name] = dict(old=best_of(lambda: legacy.solve_turn(prog, p, vo), reps), new=best_of(lambda: cold(prog, p, vn), reps))
    return out

# ---- workers (spawned)
_w = {}

def _init(old):
    import torch
    torch.set_num_threads(1)
    _w['old'], _w['v'] = old, net(old)
    if old:
        import solver_agent
        solver_agent.solve_turn = legacy.solve_turn
    (legacy.solve_turn if old else cold)(*EMPTY, _w['v'])

def _solve(_):
    t = time.perf_counter()
    (legacy.solve_turn if _w['old'] else cold)(*EMPTY, _w['v'])
    return time.perf_counter() - t

def _ready(_):
    time.sleep(0.2)
    return os.getpid()

def _play(seed):
    import td_train
    td_train._vfn = lambda src: _w['v']
    return td_train.play((None, seed))

def pooled(n, old, fn, jobs):
    with mp.get_context('spawn').Pool(n, _init, (old,)) as pool:
        pool.map(_ready, range(n), chunksize=1)
        t = time.perf_counter()
        res = pool.map(fn, jobs, chunksize=1)
        return res, time.perf_counter() - t

def parallel(procs, per=4):
    # old/new interleaved per process count (the laptop throttles under sustained load)
    rows = []
    for n in procs:
        for old in (True, False):
            res, wall = pooled(n, old, _solve, range(n * per))
            rows.append(dict(impl='old' if old else 'new', procs=n, ms_per_solve=1000 * float(np.mean(res)), solves_per_s=n * per / wall))
            print(rows[-1], flush=True)
    return rows

def selfplay(workers, games, old_workers=8):
    rows = []
    for old, n in [(o, w) for w in workers for o in ((True, False) if w == old_workers else (False,))]:
        res, wall = pooled(n, old, _play, range(10 ** 6, 10 ** 6 + games))
        rows.append(dict(impl='old' if old else 'new', workers=n, games=games, games_per_s=games / wall,
                         solve_ms=1000 * sum(r[1] for r in res) / sum(r[2] for r in res), turns=sum(len(r[0]) for r in res)))
        print(rows[-1], flush=True)
    return rows

def table(rows, cols):
    print(' | '.join(cols))
    for r in rows:
        print(' | '.join(f'{r[c]:.1f}' if isinstance(r[c], float) else str(r[c]) for c in cols))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--parts', default='stages,solve,parallel,selfplay')
    ap.add_argument('--procs', default='1,4,6,8,12')
    ap.add_argument('--workers', default='4,6,8,12')
    ap.add_argument('--games', type=int, default=48)
    ap.add_argument('--out', default=os.path.join(ROOT, 'runs', 'phase45', 'bench.json'))
    a = ap.parse_args()
    parts = a.parts.split(',')
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    boards = dict(mid=mid_board(), empty=EMPTY)
    if 'stages' in parts:
        res['stages'] = {k: stages(*b) for k, b in boards.items()}
        for k, s in res['stages'].items():
            print(f"stages {k} ({s['situations']} situations), ms:", {i: {c: round(v, 1) for c, v in s[i].items()} for i in ('old', 'new')})
    if 'solve' in parts:
        res['solve'] = {k: solves(*b) for k, b in boards.items()}
        for k, s in res['solve'].items():
            print(f'solve {k}, ms:', {v: {i: round(t, 1) for i, t in d.items()} for v, d in s.items()})
    if 'parallel' in parts:
        res['parallel'] = parallel([int(x) for x in a.procs.split(',')])
        table(res['parallel'], ['impl', 'procs', 'ms_per_solve', 'solves_per_s'])
    if 'selfplay' in parts:
        res['selfplay'] = selfplay([int(x) for x in a.workers.split(',')], a.games)
        table(res['selfplay'], ['impl', 'workers', 'games_per_s', 'solve_ms'])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, 'w'), indent=1)

if __name__ == '__main__':
    main()
