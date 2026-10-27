import os, json, time, random, argparse
from collections import deque
from multiprocessing import Pool
import numpy as np
import fastrules as fr
from game import State
from heuristics import VH
from solver_agent import SolverAgent
from turn_solver import solve_turn, _stop_boards

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

# ---- workers (value_fn is a NetV or VH; rebuilt from pickles per job, cached by id)
_cache = {}

def _vfn(key, v):
    if key not in _cache:
        _cache.clear()
        _cache[key] = v
    return _cache[key]

def play(args):
    # one greedy self-play game; at each turn start records (board, to_move, y = best(empty | s), V(s))
    key, v, seed = args
    v = _vfn(key, v)
    random.seed(seed)
    agent, s, rec, t, solves = SolverAgent(v), State(), [], 0.0, 0
    while not s.is_terminal():
        if not s.tmp_track and not s.wait_for_go and not s.end_turn:
            prog, p, _ = fr.from_state(s)
            t0 = time.time()
            sol = agent.solution(prog, p, ZERO)
            t += time.time() - t0
            solves += 1
            rec.append((prog, p, sol.value(ZERO), float(v(prog[None], p)[0])))
        s, _ = s.update(agent.act(s))
    w = s.get_winner()
    return [r + (float(r[1] == w),) for r in rec], t, solves

def residual_job(args):
    key, v, progs, ps = args
    v = _vfn(key, v)
    return [(solve_turn(b, int(p), v).value(ZERO), float(v(b[None], int(p))[0])) for b, p in zip(progs, ps)]

def stop_sample(args):
    # stop boards a turn from (prog, p) can bank, subsampled; queried for the opponent (1 - p)
    prog, p, k, seed = args
    keys, _, _ = fr.enumerate_turn(prog, p)
    boards, win = _stop_boards(prog, p, keys)
    boards = boards[~win]
    sel = np.random.default_rng(seed).choice(len(boards), min(k, len(boards)), replace=False)
    return np.concatenate([boards[sel], prog[None]]), np.r_[np.full(len(sel), 1 - p), p]

# ---- learner
def fit(net, X, y, steps, batch, opt, logf=None, it=None):
    import torch
    X, y = torch.as_tensor(X, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)
    bce = torch.nn.BCEWithLogitsLoss()
    net.train()
    for k in range(steps):
        i = torch.randint(len(X), (batch,))
        loss = bce(net(X[i])[:, 0], y[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if logf:
            log(logf, kind='update', iter=it, step=k, loss=round(loss.item(), 5))
    net.eval()
    return loss.item()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='phase4')
    ap.add_argument('--minutes', type=float, default=90)
    ap.add_argument('--iters', type=int, default=10 ** 9)
    ap.add_argument('--workers', type=int, default=8)
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
    a = ap.parse_args()
    out = os.path.join('runs', a.run)
    ckdir = os.path.join(out, 'ckpt')
    os.makedirs(ckdir, exist_ok=True)
    mf = os.path.join(out, 'metrics.jsonl')
    pool = Pool(a.workers)  # before any torch work in this process (fork safety)
    import torch
    from value_net import NetV, make_net, features
    torch.manual_seed(a.seed)
    torch.set_num_threads(4)
    X = lambda progs, ms: features(pov(progs, ms), 0)
    log(mf, kind='config', **vars(a), t=time.time())

    # warm start: self-play with best V_h; first third of games -> held-out residual boards, rest -> stop boards fit to V_h
    t0 = time.time()
    games = pool.map(play, [('vh', BEST_VH, a.seed * 10 ** 6 + i) for i in range(a.warm_games)], chunksize=1)
    held = [r for g in games[:a.warm_games // 3] for r in g[0]]
    held = [held[i] for i in np.random.default_rng(a.seed).choice(len(held), min(a.n_resid, len(held)), replace=False)]
    H_progs, H_p = np.array([r[0] for r in held]), np.array([r[1] for r in held])
    starts = [r for g in games[a.warm_games // 3:] for r in g[0]]
    parts = pool.map(stop_sample, [(r[0], r[1], 150, i) for i, r in enumerate(starts)])
    wb, wm = np.concatenate([q[0] for q in parts]), np.concatenate([q[1] for q in parts])
    wy = np.concatenate([BEST_VH(wb[wm == q], q) for q in (0, 1)])
    wb, wm = np.concatenate([wb[wm == 0], wb[wm == 1]]), np.concatenate([wm[wm == 0], wm[wm == 1]])
    net = make_net(tuple(int(h) for h in a.hidden.split(',')))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    WX = X(wb, wm)
    for lr, steps in ((3e-3, 3000), (1e-3, 2000), (3e-4, 2000)):
        for g in opt.param_groups:
            g['lr'] = lr
        fit(net, WX, wy, steps, 1024, opt)
    v = NetV(net, threads=4)
    hp = np.concatenate([v(H_progs[H_p == q], q) for q in (0, 1)])
    hv = np.concatenate([BEST_VH(H_progs[H_p == q], q) for q in (0, 1)])
    wp = v(pov(wb, wm), 0)
    line = log(mf, kind='warm', n_boards=len(wb), n_starts=len(starts), mae_stop=float(np.abs(wp - wy).mean()),
        mae_held=float(np.abs(hp - hv).mean()), seconds=round(time.time() - t0, 1))
    warm = make_net(net_hidden := tuple(int(h) for h in a.hidden.split(',')))
    warm.load_state_dict(net.state_dict())
    warm_v = NetV(warm, threads=4)
    NetV(net).save(os.path.join(ckdir, 'it0000.pt'))
    print(line, flush=True)

    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    buf = deque(maxlen=a.buffer)
    calib = []
    t_start = time.time()
    it = 0
    while it < a.iters and time.time() - t_start < a.minutes * 60:
        t0 = time.time()
        snap = NetV(make_net(net_hidden), threads=1)
        snap.net.load_state_dict(net.state_dict())
        key = f'it{it}'
        chunks = np.array_split(np.arange(len(H_p)), 2 * a.workers)
        do_resid = it % a.resid_every == 0
        rjobs = pool.map_async(residual_job, [(key, snap, H_progs[c], H_p[c]) for c in chunks] if do_resid else [])
        res = list(pool.imap_unordered(play, [(key, snap, (a.seed + 1) * 10 ** 7 + it * 1000 + i) for i in range(a.games)]))
        resid = np.array([r for part in rjobs.get() for r in part]).reshape(-1, 2)
        wall = time.time() - t0
        recs = [r for g in res for r in g[0]]
        buf.extend((r[0], r[1], r[2]) for r in recs)
        calib += recs
        bp = np.array([r[0] for r in buf])
        bm = np.array([r[1] for r in buf])
        by = np.array([r[2] for r in buf])
        steps = max(1, int(a.replay * len(recs) / a.batch))
        fit(net, X(bp, bm), by, steps, a.batch, opt, mf, it)
        it += 1
        NetV(net).save(os.path.join(ckdir, f'it{it:04d}.pt'))
        vp = np.array([r[3] for r in recs])
        line = log(mf, kind='iter', iter=it, t=round(time.time() - t_start, 1), residual=float(np.abs(resid[:, 1] - resid[:, 0]).mean()) if do_resid else None,
            resid_signed=float((resid[:, 1] - resid[:, 0]).mean()) if do_resid else None,
            v_mean=float(vp.mean()), v_std=float(vp.std()), y_mean=float(np.mean([r[2] for r in recs])),
            games_per_s=a.games / wall, solve_s=sum(g[1] for g in res) / sum(g[2] for g in res),
            buffer=len(buf), pairs=len(recs), steps=steps, iter_s=round(time.time() - t0, 1))
        if it % a.calib_every == 0:
            cp, cm, cv, co = (np.array([r[i] for r in calib]) for i in (0, 1, 3, 4))
            cw = warm_v(pov(cp, cm), 0)
            nb, ne = bins(cv, co)
            wbins, we = bins(cw, co)
            line += '\n' + log(mf, kind='calib', iter=it, n=len(co), net=nb, ece_net=ne, warm=wbins, ece_warm=we)
            calib = []
        print(line[:400], flush=True)
    pool.close()
    open(os.path.join(out, 'DONE'), 'w').close()

if __name__ == '__main__':
    main()
