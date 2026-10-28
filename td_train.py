import os, copy, json, time, random, argparse
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')  # before numpy: workers inherit it (numba's GEMM, any net width)
from collections import deque
from multiprocessing import Pool
import numpy as np
import fastrules as fr
from game import State
from heuristics import VH
from solver_agent import SolverAgent
from turn_solver import solve_turn, _stop_boards
from branching import SETS, Turn, branch, build_sets

BEST_VH = VH('claim_top3', (0.6431, 0.6591), 0.2690)
ZERO = np.zeros(13, np.int8)

def pov(progs, ms):
    # boards flipped so the player to move is row 0
    progs, ms = np.asarray(progs), np.asarray(ms)
    return np.where((ms == 0)[:, None, None], progs, progs[:, ::-1])

def log(path, **rec):
    line = json.dumps(rec)
    with open(path, 'a') as f:
        f.write(line + '\n')
    return line

def bins(p, y, k=10):
    idx = np.minimum((p * k).astype(int), k - 1)
    rows = [[b, float(p[idx == b].mean()), float(y[idx == b].mean()), int((idx == b).sum())] for b in range(k) if (idx == b).any()]
    return rows, float(sum(abs(r[1] - r[2]) * r[3] for r in rows) / len(p))

# ---- workers: the value_fn source is a VH or a checkpoint path (loaded once per worker per snapshot, as a FusedNet)
_cache = {}

def _vfn(src):
    if not isinstance(src, str):
        return src
    if src not in _cache:
        from value_net import NetV
        _cache.clear()
        _cache[src] = NetV.load(src).fused()
    return _cache[src]

def play(args):
    # one greedy self-play game; at each turn start records (board, to_move, y = best(empty | s), V(s), won).
    # With br = (k_end, k_close, delta, rho_min), after the game each turn start adds extra boards (branching R1/R2, own
    # rng; the game is unchanged), solved for the opponent with the same net: extra (board, to_move, y, source), and
    # per turn start the gaps of its close calls
    src, seed = args[:2]
    br = args[2] if len(args) > 2 else None
    v = _vfn(src)
    random.seed(seed)
    agent, s, rec, t, solves, sols = SolverAgent(v), State(), [], 0.0, 0, []
    while not s.is_terminal():
        if not s.tmp_track and not s.wait_for_go and not s.end_turn:
            prog, p, _ = fr.from_state(s)
            t0 = time.time()
            sol = agent.solution(prog, p, ZERO)
            t += time.time() - t0
            solves += 1
            rec.append((prog, p, sol.value(ZERO), float(v(prog[None], p)[0])))
            if br and (br[0] or br[1]):
                sols.append(sol)
        s, _ = s.update(agent.act(s))
    w = s.get_winner()
    extra, gaps = [], []
    if sols:
        k_end, k_close, delta, rho_min = br
        rng, exclude = np.random.default_rng(seed), {(r[0].tobytes(), r[1]) for r in rec}
        for sol in sols:
            tn = Turn(sol, delta, rho_min)
            gaps.append([c[4] for c in tn.close])
            for b, tag in branch(tn, k_end, k_close, rng, exclude):
                t0 = time.time()
                extra.append((b, 1 - tn.p, solve_turn(b, 1 - tn.p, v).value(ZERO), tag))
                t += time.time() - t0
                solves += 1
    return [r + (float(r[1] == w),) for r in rec], t, solves, extra, gaps

def residual_job(args):
    src, progs, ps = args
    v = _vfn(src)
    return [(solve_turn(b, int(p), v).value(ZERO), float(v(b[None], int(p))[0])) for b, p in zip(progs, ps)]

def heldout(pool, a, n_games):
    # held-out residual boards: turn starts of the first third of the V_h warm-up games (deterministic in seed);
    # the remaining games' turn starts are returned for the warm-start fit
    games = pool.map(play, [(BEST_VH, a.seed * 10 ** 6 + i) for i in range(n_games)], chunksize=1)
    held = [r for g in games[:a.warm_games // 3] for r in g[0]]
    held = [held[i] for i in np.random.default_rng(a.seed).choice(len(held), min(a.n_resid, len(held)), replace=False)]
    return np.array([r[0] for r in held]), np.array([r[1] for r in held]), [r for g in games[a.warm_games // 3:] for r in g[0]]

def stop_sample(args):
    # stop boards a turn from (prog, p) can bank, subsampled; queried for the opponent (1 - p)
    prog, p, k, seed = args
    keys, _, _ = fr.enumerate_turn(prog, p)
    boards, win = _stop_boards(prog, p, keys)
    boards = boards[~win]
    sel = np.random.default_rng(seed).choice(len(boards), min(k, len(boards)), replace=False)
    return np.concatenate([boards[sel], prog[None]]), np.r_[np.full(len(sel), 1 - p), p]

# ---- learner
def fit(net, X, y, steps, batch, opt):
    # returns the per-step losses
    import torch
    X, y = torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)
    bce = torch.nn.BCEWithLogitsLoss()
    net.train()
    losses = []
    for _ in range(steps):
        i = torch.randint(len(X), (batch,))
        loss = bce(net(X[i])[:, 0], y[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(round(loss.item(), 5))
    net.eval()
    return losses

def ckpt(ckdir, it):
    return os.path.join(ckdir, f'it{it:04d}.pt')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='phase4')
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--minutes', type=float, default=90)
    ap.add_argument('--iters', type=int, default=10 ** 9)
    ap.add_argument('--workers', type=int, default=8, help='self-play processes, ~420 MB each; 12 is ~7%% faster if RAM allows')
    ap.add_argument('--games', type=int, default=96)
    ap.add_argument('--hidden', default='128,64')
    ap.add_argument('--buffer', type=int, default=15000)
    ap.add_argument('--replay', type=float, default=8)
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--warm_games', type=int, default=96)
    ap.add_argument('--n_resid', type=int, default=200)
    ap.add_argument('--resid_every', type=int, default=2)
    ap.add_argument('--calib_every', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--no_warm', action='store_true', help='skip the V_h fit: snapshot 0 is the randomly initialized (seeded) net')
    ap.add_argument('--branch_end', type=int, default=1, help='extra end boards per turn start (branching R1)')
    ap.add_argument('--branch_close', type=int, default=2, help='extra close-call boards per turn start (branching R1)')
    ap.add_argument('--close_delta', type=float, default=0.02)
    ap.add_argument('--rho_min', type=float, default=0.01)
    ap.add_argument('--n_set', type=int, default=900, help='turn starts in the visited / end / close residual sets (R7); 0: none')
    ap.add_argument('--set_games', type=int, default=100)
    ap.add_argument('--set_seed', type=int, default=10000, help='first V_h game seed of the R7 sets (disjoint from training)')
    ap.add_argument('--resid_sub', type=int, default=300, help='R7 rows per measurement (fixed subset; 0: all)')
    ap.add_argument('--resid_full_every', type=int, default=20, help='R7 on all rows every this many iterations')
    a = ap.parse_args()
    out = os.path.join('runs', a.run)
    ckdir = os.path.join(out, 'ckpt')
    mf = os.path.join(out, 'metrics.jsonl')
    prior = sorted(f for f in os.listdir(ckdir) if f.startswith('it') and f.endswith('.pt')) if os.path.isdir(ckdir) else []
    if os.path.exists(mf) and not a.resume:
        raise SystemExit(f'{out} already has a run; pass --resume to continue it or choose another --run')
    if a.resume and not prior:
        raise SystemExit(f'--resume: no checkpoints in {ckdir}')
    os.makedirs(ckdir, exist_ok=True)
    if os.path.exists(os.path.join(out, 'DONE')):
        os.remove(os.path.join(out, 'DONE'))
    pidf = os.path.join(out, 'train.pid')
    open(pidf, 'w').write(str(os.getpid()))
    pool = Pool(a.workers)  # before any torch work in this process (fork safety)
    import torch
    from value_net import NetV, make_net, features
    torch.manual_seed(a.seed)
    torch.set_num_threads(4)
    X = lambda progs, ms: features(pov(progs, ms), 0)
    log(mf, kind='config', **vars(a), t=time.time())
    t0 = time.time()
    H_progs, H_p, starts = heldout(pool, a, a.warm_games // 3 if a.resume or a.no_warm else a.warm_games)
    t1 = time.time()
    rows = build_sets(pool, a.set_seed, a.set_games, a.n_set, a.close_delta, a.rho_min)[1] if a.n_set else []
    sub = np.random.default_rng(a.seed).choice(len(rows), min(a.resid_sub or len(rows), len(rows)), replace=False)
    print(log(mf, kind='sets', **{k: sum(k in r for r in rows) for k in SETS}, seconds=round(time.time() - t1, 1)), flush=True)

    if a.resume:
        it = int(prior[-1][2:6])
        net = NetV.load(ckpt(ckdir, it), threads=4).net
        warm = NetV.load(ckpt(ckdir, 0), threads=4)
        last_t = max((json.loads(l).get('t', 0) for l in open(mf) if '"kind": "iter"' in l), default=0)
        log(mf, kind='resume', iter=it, t=last_t, note='fresh buffer and optimizer')
    else:
        net = make_net(tuple(int(h) for h in a.hidden.split(',')))
        info = dict(init='random')
        if not a.no_warm:
            # warm start: fit the net to V_h on stop boards (queried for the opponent) + turn starts of the V_h games
            parts = pool.map(stop_sample, [(r[0], r[1], 150, i) for i, r in enumerate(starts)])
            wb, wm = np.concatenate([q[0] for q in parts]), np.concatenate([q[1] for q in parts])
            wy = np.concatenate([BEST_VH(wb[wm == q], q) for q in (0, 1)])
            wb, wm = np.concatenate([wb[wm == 0], wb[wm == 1]]), np.concatenate([wm[wm == 0], wm[wm == 1]])
            opt = torch.optim.Adam(net.parameters())
            WX = X(wb, wm)
            for lr, steps in ((3e-3, 3000), (1e-3, 2000), (3e-4, 2000)):
                for g in opt.param_groups:
                    g['lr'] = lr
                fit(net, WX, wy, steps, 1024, opt)
            info = dict(init='vh_fit', n_boards=len(wb), n_starts=len(starts), mae_stop=float(np.abs(NetV(net)(pov(wb, wm), 0) - wy).mean()))
        warm = NetV(copy.deepcopy(net))
        hp = np.concatenate([warm(H_progs[H_p == q], q) for q in (0, 1)])
        hv = np.concatenate([BEST_VH(H_progs[H_p == q], q) for q in (0, 1)])
        print(log(mf, kind='warm', **info, mae_held=float(np.abs(hp - hv).mean()), seconds=round(time.time() - t0, 1)), flush=True)
        it, last_t = 0, 0
        warm.save(ckpt(ckdir, 0))

    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    buf = deque(maxlen=a.buffer)
    calib = []
    t_start = time.time() - last_t
    t_end = time.time() + a.minutes * 60
    chunks = np.array_split(np.arange(len(H_p)), 2 * a.workers)
    for _ in range(a.iters):
        if time.time() >= t_end:
            break
        t0 = time.time()
        src = ckpt(ckdir, it)  # the snapshot (target net) workers play and bootstrap with
        do_resid = it % a.resid_every == 0
        rjobs = pool.map_async(residual_job, [(src, H_progs[c], H_p[c]) for c in chunks] if do_resid else [])
        full = bool(rows) and it % a.resid_full_every == 0
        sel = [(k, rows[i][k]) for k in SETS for i in (range(len(rows)) if full else sub) if k in rows[i]] if do_resid or full else []
        sjobs = pool.map_async(residual_job, [(src, np.array([sel[i][1][0] for i in c]), np.array([sel[i][1][1] for i in c]))
                                              for c in np.array_split(np.arange(len(sel)), 2 * a.workers) if len(c)])
        br = (a.branch_end, a.branch_close, a.close_delta, a.rho_min)
        res = list(pool.imap_unordered(play, [(src, (a.seed + 1) * 10 ** 7 + it * 1000 + i, br) for i in range(a.games)]))
        resid = np.array([r for part in rjobs.get() for r in part]).reshape(-1, 2)
        sres = np.array([r for part in sjobs.get() for r in part]).reshape(-1, 2)
        wall = time.time() - t0
        recs = [r for g in res for r in g[0]]
        extra = [e for g in res for e in g[3]]
        gaps = [g for r in res for g in r[4]]
        buf.extend((r[0], r[1], r[2], 'visited') for r in recs)
        buf.extend(extra)
        calib += recs
        bp, bm, by = (np.array([r[i] for r in buf]) for i in range(3))
        steps = max(1, int(a.replay * (len(recs) + len(extra)) / a.batch))
        losses = fit(net, X(bp, bm), by, steps, a.batch, opt)
        it += 1
        NetV(net).save(ckpt(ckdir, it))
        vp = np.array([r[3] for r in recs])
        d = resid[:, 1] - resid[:, 0]
        tags = np.array([k for k, _ in sel])
        rsets = {k: float(np.abs(sres[tags == k, 1] - sres[tags == k, 0]).mean()) for k in SETS if (tags == k).any()}
        npairs = dict(visited=len(recs), **{k: sum(e[3] == k for e in extra) for k in ('end', 'close')})
        lines = [json.dumps(dict(kind='update', iter=it - 1, loss=losses)),
                 json.dumps(dict(kind='iter', iter=it, t=round(time.time() - t_start, 1),
                                 residual=float(np.abs(d).mean()) if do_resid else None, resid_signed=float(d.mean()) if do_resid else None,
                                 v_mean=float(vp.mean()), v_std=float(vp.std()), y_mean=float(np.mean([r[2] for r in recs])),
                                 resid_sets=rsets or None, resid_n=(len(rows) if full else len(sub)) if sel else None,
                                 games_per_s=a.games / wall, targets_per_s=sum(npairs.values()) / wall,
                                 solve_s=sum(g[1] for g in res) / sum(g[2] for g in res), solves_per_turn=sum(g[2] for g in res) / len(recs),
                                 close_per_turn=float(np.mean([len(g) for g in gaps])) if gaps else None,
                                 close_gap=float(np.mean([x for g in gaps for x in g])) if any(gaps) else None,
                                 buffer=len(buf), pairs=len(recs), pairs_src=npairs, steps=steps, iter_s=round(time.time() - t0, 1)))]
        if it % a.calib_every == 0:
            cp, cm, cv, co = (np.array([r[i] for r in calib]) for i in (0, 1, 3, 4))
            (nb, ne), (wbins, we) = bins(cv, co), bins(warm(pov(cp, cm), 0), co)
            lines.append(json.dumps(dict(kind='calib', iter=it, n=len(co), net=nb, ece_net=ne, warm=wbins, ece_warm=we)))
            calib = []
        with open(mf, 'a') as f:
            f.write('\n'.join(lines) + '\n')
        print(lines[1][:400], flush=True)
    pool.close()
    open(os.path.join(out, 'DONE'), 'w').close()
    os.remove(pidf)

if __name__ == '__main__':
    main()
