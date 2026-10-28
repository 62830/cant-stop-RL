import os, sys, json, time, argparse
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from multiprocessing import Pool
import numpy as np
from numba import njit
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import fastrules as fr
from turn_solver import solve_turn, turn_graph
from td_train import BEST_VH, play, residual_job

# Step 0 of the branching spec: residual of existing nets on visited / end / close held-out sets.
# The forward pass, close-call detection and board extraction are the pieces Step 1 moves into the library.
DELTA, RHO_MIN = 0.02, 0.01
NETS = {'td30': os.path.join(ROOT, 'runs/td30/it0082.pt'), 'scratch30': os.path.join(ROOT, 'runs/scratch30/it0297.pt')}
WIDE60 = os.path.join(os.path.dirname(ROOT), 'cant_stop_wide/runs/wide60/it0100.pt')  # PR #10, not merged

@njit(cache=True)
def _order(prog, p, keys):
    # rows by increasing total advance: every child advances >= 1, so parents come first
    adv, t = np.zeros(len(keys), np.int64), np.zeros(13, np.int8)
    for i in range(len(keys)):
        fr.decode_into(keys[i], t)
        for c in range(2, 13):
            if t[c] > 0:
                adv[i] += t[c] - prog[p, c]
    return np.argsort(adv)

@njit(cache=True)
def _rec(recs, nr, kind, i, chosen, rej, gap, rho):
    recs[nr, 0], recs[nr, 1], recs[nr, 2], recs[nr, 3], recs[nr, 4], recs[nr, 5] = kind, i, chosen, rej, gap, rho
    return nr + 1

@njit(cache=True)
def _forward(order, ptr, cmv, crow, best, stop, go, start, force_go, delta, rho_min):
    # reach probabilities under pi (go iff go > stop; pick the child with max best) from row start (rho = 1).
    # returns rho, end (stop mass per row), bust mass, close-call records (kind 0 go/stop, 1 pick; row, chosen, rejected,
    # gap, rho) at rows with rho >= rho_min (start's forced go is not a decision)
    n = len(best)
    rho, end = np.zeros(n), np.zeros(n)
    rho[start] = 1.0
    bust = 0.0
    recs, nr = np.empty((256, 6)), 0
    sv, pu = np.empty(13, np.int64), np.empty(len(fr.UPAIRS), np.int64)
    cand = np.empty(6, np.int64)
    for i in order:
        if rho[i] == 0.0:
            continue
        forced = force_go and i == start
        goes = forced or go[i] > stop[i]  # NaN go (winning stop, unexpanded) -> stop
        if nr + 260 > len(recs):
            recs = np.concatenate((recs, np.empty_like(recs)))
        if not forced and rho[i] >= rho_min and not np.isnan(go[i]) and stop[i] > -np.inf and abs(go[i] - stop[i]) <= delta:
            nr = _rec(recs, nr, 0, i, goes, -1, abs(go[i] - stop[i]), rho[i])
        if not goes:
            end[i] += rho[i]
            continue
        sv[:], pu[:] = -1, -1
        for e in range(ptr[i], ptr[i + 1]):
            if cmv[e] < 11:
                sv[cmv[e] + 2] = crow[e]
            else:
                pu[cmv[e] - 11] = crow[e]
        for r in range(fr.N_ROLLS):
            k = 0
            for j in range(fr.ROLL_NPAIRS[r]):
                u = fr.ROLL_UPAIR[r, j]
                if pu[u] >= 0:
                    cand[k] = pu[u]
                    k += 1
                else:
                    for c in (fr.UPAIRS[u, 0], fr.UPAIRS[u, 1]):
                        if sv[c] >= 0:
                            cand[k] = sv[c]
                            k += 1
            if k == 0:
                bust += rho[i] * fr.ROLL_PROB[r]
                continue
            b1, b2 = -1, -1  # best and second-best distinct child rows
            for q in range(k):
                c = cand[q]
                if b1 < 0 or best[c] > best[b1]:
                    if c != b1:
                        b2 = b1
                    b1 = c
                elif c != b1 and (b2 < 0 or best[c] > best[b2]):
                    b2 = c
            rho[b1] += rho[i] * fr.ROLL_PROB[r]
            if b2 >= 0 and rho[i] >= rho_min and best[b1] - best[b2] <= delta:
                nr = _rec(recs, nr, 1, i, b1, b2, best[b1] - best[b2], rho[i] * fr.ROLL_PROB[r])
    return rho, end, bust, recs[:nr]

class Turn:
    # a solved turn start (prog, p) with the solver policy's forward pass
    def __init__(self, prog, p, value_fn):
        self.prog, self.p = prog, int(p)
        self.sol = solve_turn(prog, p, value_fn)
        self.child = turn_graph(prog, p, -1, 0)[3]
        self.order = _order(prog, self.p, self.sol.keys)
        self.win = self.sol.stop >= 1.0
        self.rho, self.end, self.bust, recs = self.forward()
        self.close = merge_close(recs)

    def forward(self, start=0, force_go=False, light=False):
        # light: no close-call records and no mass check (sampling from a counterfactual branch)
        s = self.sol
        rho, end, bust, recs = _forward(self.order, *self.child, s.best, s.stop, s.go, start, force_go, DELTA,
                                        np.inf if light else RHO_MIN)
        if not light:
            assert abs(end.sum() + bust - 1) < 1e-9, end.sum() + bust
        return rho, end, bust, recs

    def board(self, row):
        # board after banking row's tmp (row -1: the bust board, unchanged); the opponent moves next
        return self.prog.copy() if row < 0 else fr.apply_stop(self.prog, self.p, fr.decode(self.sol.keys[row]))

    def sample_end(self, rng, exclude, start=0, force_go=False):
        # one non-terminal end board of mu (from start, optionally forcing go there), skipping excluded (board bytes, to move)
        end, bust = (self.end, self.bust) if start == 0 and not force_go else self.forward(start, force_go, True)[1:3]
        rows = np.r_[np.flatnonzero((end > 0) & ~self.win), -1]
        mass = np.r_[end[rows[:-1]], bust]
        while mass.sum() > 0:
            j = rng.choice(len(rows), p=mass / mass.sum())
            b = self.board(rows[j])
            if (b.tobytes(), 1 - self.p) not in exclude:
                return b
            mass[j] = 0
        return None

    def rejected_board(self, rng, rec, exclude):
        kind, i, chosen, rej = rec[:4]
        if kind == 0 and chosen:  # pi goes, stop rejected
            b = self.board(i)
            return None if self.win[i] or (b.tobytes(), 1 - self.p) in exclude else b
        if kind == 0:  # pi stops, go rejected
            return self.sample_end(rng, exclude, i, True)
        return self.sample_end(rng, exclude, rej)

    def sample_close(self, rng, exclude):
        # one close call drawn with probability ~ rho (resampling if its board is excluded) -> its rejected-option board
        recs = list(self.close)
        while recs:
            w = np.array([r[5] for r in recs])
            rec = recs.pop(rng.choice(len(recs), p=w / w.sum()))
            b = self.rejected_board(rng, rec, exclude)
            if b is not None:
                return b
        return None

def merge_close(recs):
    # one decision per (kind, row, chosen, rejected): rolls offering the same top two options merge, summing rho;
    # kept if rho >= RHO_MIN. Returns tuples (kind, row, chosen, rejected, gap, rho)
    out = {}
    for k, i, c, r, gap, rho in recs:
        key = (int(k), int(i), int(c), int(r))
        out[key] = (gap, out.get(key, (0, 0.0))[1] + rho)
    return [k + v for k, v in out.items() if v[1] >= RHO_MIN]

def legal(b):
    L = fr.LEN[2:]
    h = b[:, 2:]
    return bool((h >= 0).all() and (h <= L).all() and not ((h == L) & (h[::-1] > 0)).any() and (h == L).sum(1).max() < 3)

def candidates(sample, rng, exclude, to_move, k):
    # up to k distinct boards from sample(rng, exclude) (to_move: who moves on them)
    out, ex = [], set(exclude)
    for _ in range(k):
        b = sample(rng, ex)
        if b is None:
            break
        out.append(b)
        ex.add((b.tobytes(), to_move))
    return out

def game_job(g, k=4):
    # one V_h game with seed g (td_train.heldout: seed 0 -> games 0..95): per turn start, close-call stats and up to k
    # end / close candidate boards excluding this game's turn starts (incl. each turn's actual next start); main() takes
    # the first candidate not excluded globally (R2)
    recs = play((BEST_VH, g))[0]
    rng = np.random.default_rng(g)
    starts = {(r[0].tobytes(), r[1]) for r in recs}
    out = []
    for prog, p, *_ in recs:
        t = Turn(prog, p, BEST_VH)
        out.append(dict(prog=prog, p=int(p), end=candidates(t.sample_end, rng, starts, 1 - p, k),
                        close=candidates(t.sample_close, rng, starts, 1 - p, k), n_close=len(t.close), gaps=[r[4] for r in t.close],
                        exp_close=float(sum(r[5] for r in t.close)), win_mass=float(t.end[t.win].sum()), bust=t.bust))
    return out

def ci(x):
    return [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]

def build_sets(pool, seed0, n_games, n):
    # n turn starts of V_h games seed0.. (visited) and, paired with each, its end and close board (None if none survives
    # R2: excluded are all turn starts of these games and every board already taken, globally)
    flat = [r for g in pool.map(game_job, range(seed0, seed0 + n_games), chunksize=1) for r in g]
    sel = np.random.default_rng(seed0).choice(len(flat), min(n, len(flat)), replace=False)
    seen = {(r['prog'].tobytes(), r['p']) for r in flat}
    rows = []
    for i in sel:
        r, row = flat[i], dict(visited=(flat[i]['prog'], flat[i]['p']))
        for src in ('end', 'close'):
            b = next((b for b in r[src] if (b.tobytes(), 1 - r['p']) not in seen), None)
            if b is not None:
                seen.add((b.tobytes(), 1 - r['p']))
                row[src] = (b, 1 - r['p'])
        rows.append(row)
    return flat, rows

def residuals(pool, path, rows, workers):
    # abs residual |V - y| per (turn start, set), NaN where the row has no board; and signed means
    D = np.full((len(rows), 3), np.nan)
    signed = []
    for j, sname in enumerate(SETS):
        idx = [i for i, r in enumerate(rows) if sname in r]
        progs, ms = np.array([rows[i][sname][0] for i in idx]), np.array([rows[i][sname][1] for i in idx])
        chunks = np.array_split(np.arange(len(ms)), 4 * workers)
        yv = np.array([r for part in pool.map(residual_job, [(path, progs[c], ms[c]) for c in chunks]) for r in part])
        D[idx, j] = np.abs(yv[:, 1] - yv[:, 0])
        signed.append(float((yv[:, 1] - yv[:, 0]).mean()))
    return D, signed

SETS = ('visited', 'end', 'close')

def summarize(D, signed, B=5000, seed=0):
    # paired bootstrap over turn starts: visited/end/close of a resampled turn start are drawn together
    bs = np.stack([np.nanmean(D[np.random.default_rng(seed + b).integers(len(D), size=len(D))], 0) for b in range(B)])
    m = np.nanmean(D, 0)
    R = {s: dict(n=int((~np.isnan(D[:, j])).sum()), mean_abs=float(m[j]), ci=ci(bs[:, j]), signed=signed[j])
         for j, s in enumerate(SETS)}
    for j in (1, 2):
        R[SETS[j]].update(ratio=float(m[j] / m[0]), ratio_ci=ci(bs[:, j] / bs[:, 0]))
    return R

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--n', type=int, default=900, help='turn starts (= visited size; end/close are paired with them)')
    ap.add_argument('--games', type=int, default=100, help='V_h games the turn starts come from')
    ap.add_argument('--seed0', type=int, default=10000, help='first game seed; 10000+ is disjoint from every training '
                    'game. --seed0 0 --games 32 --n 200 is exactly td_train.heldout()')
    ap.add_argument('--wide60', default=WIDE60)
    ap.add_argument('--out', default=os.path.join(ROOT, 'runs', 'diag'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    nets = dict(NETS, wide60=a.wide60)
    t0 = time.time()
    with Pool(a.workers) as pool:
        flat, rows = build_sets(pool, a.seed0, a.games, a.n)
        sizes = {s: sum(s in r for r in rows) for s in SETS}
        for s, k in sizes.items():
            assert all(legal(r[s][0]) for r in rows if s in r)
            if k < 0.95 * a.n:
                print(f'WARNING: {s} set has {k} boards (< 95% of {a.n})', flush=True)
        print(f'sets built in {time.time() - t0:.0f}s: {sizes}', flush=True)
        gaps = [x for r in flat for x in r['gaps']]
        res = dict(seed0=a.seed0, games=a.games, delta=DELTA, rho_min=RHO_MIN, sizes=sizes,
                   close_calls=dict(turns=len(flat), per_turn=float(np.mean([r['n_close'] for r in flat])),
                                    frac_turns_ge1=float(np.mean([r['n_close'] > 0 for r in flat])),
                                    expected_per_turn=float(np.mean([r['exp_close'] for r in flat])),
                                    mean_gap=float(np.mean(gaps)), win_mass=float(np.mean([r['win_mass'] for r in flat])),
                                    bust_mass=float(np.mean([r['bust'] for r in flat]))), nets={})
        for name, path in nets.items():
            if not os.path.exists(path):
                print(f'WARNING: skipping {name}, {path} not found', flush=True)
                continue
            t1 = time.time()
            R = summarize(*residuals(pool, path, rows, a.workers))
            res['nets'][name] = dict(path=os.path.relpath(path, ROOT), seconds=round(time.time() - t1, 1), **R)
            print(name, json.dumps(R), flush=True)
    res['proceed'] = {k: max(v['end']['ratio'], v['close']['ratio']) >= 1.5 for k, v in res['nets'].items()}
    res['seconds'] = round(time.time() - t0, 1)
    json.dump(res, open(os.path.join(a.out, 'results.json'), 'w'), indent=1)
    print(json.dumps(res['close_calls']), res['proceed'], f"{res['seconds']}s")

if __name__ == '__main__':
    main()
