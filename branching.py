import numpy as np
from numba import njit
import fastrules as fr

# decision-relevant extra boards (branching spec R1/R2): the solver policy pi's forward pass over a solved turn start
# (reach probabilities rho, end distribution mu), close calls, and boards sampled from them; plus the held-out sets (R7)
DELTA, RHO_MIN = 0.02, 0.01
SETS = ('visited', 'end', 'close')

@njit(cache=True)
def _rec(recs, nr, kind, i, chosen, rej, gap, rho):
    recs[nr, 0], recs[nr, 1], recs[nr, 2], recs[nr, 3], recs[nr, 4], recs[nr, 5] = kind, i, chosen, rej, gap, rho
    return nr + 1

@njit(cache=True)
def _forward(order, ptr, cmv, crow, best, stop, go, start, force_go, delta, rho_min):
    # reach probabilities under pi (go iff go > stop; pick the child with max best) from row start (rho = 1); order: parents
    # first. Returns rho, end (stop mass per row), bust mass, close-call records (kind 0 go/stop, 1 pick; row, chosen,
    # rejected, gap, rho) at rows with rho >= rho_min (start's forced go is not a decision)
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
    # pi's forward pass over sol, an exact turn_solver.Solution from a turn start (prog, p)
    def __init__(self, sol, delta=DELTA, rho_min=RHO_MIN):
        assert sol.limit == -1 and sol.keys[0] == 0 and sol.child is not None
        self.sol, self.prog, self.p, self.delta, self.rho_min = sol, sol.prog, int(sol.p), delta, rho_min
        self.order = np.ascontiguousarray(sol.order[::-1])
        self.win = sol.stop >= 1.0
        self.rho, self.end, self.bust, recs = self.forward()
        self.close = merge_close(recs, rho_min)

    def forward(self, start=0, force_go=False, light=False):
        # light: no close-call records and no mass check (sampling from a counterfactual branch)
        s = self.sol
        rho, end, bust, recs = _forward(self.order, *s.child, s.best, s.stop, s.go, start, force_go, self.delta,
                                        np.inf if light else self.rho_min)
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

def merge_close(recs, rho_min=RHO_MIN):
    # one decision per (kind, row, chosen, rejected): rolls offering the same top two options merge, summing rho;
    # kept if rho >= rho_min. Returns tuples (kind, row, chosen, rejected, gap, rho)
    out = {}
    for k, i, c, r, gap, rho in recs:
        key = (int(k), int(i), int(c), int(r))
        out[key] = (gap, out.get(key, (0, 0.0))[1] + rho)
    return [k + v for k, v in out.items() if v[1] >= rho_min]

def legal(b):
    L = fr.LEN[2:]
    h = b[:, 2:]
    return bool((h >= 0).all() and (h <= L).all() and not ((h == L) & (h[::-1] > 0)).any() and (h == L).sum(1).max() < 3)

def candidates(sample, rng, exclude, to_move, k):
    # up to k distinct boards from sample(rng, exclude) (to_move: who moves on them); adds them to exclude
    out = []
    for _ in range(k):
        b = sample(rng, exclude)
        if b is None:
            break
        out.append(b)
        exclude.add((b.tobytes(), to_move))
    return out

def branch(t, k_end, k_close, rng, exclude):
    # R1/R2 for one Turn: up to k_end end boards and k_close close-call boards (opponent to move) -> [(board, source)]
    return [(b, src) for src, k, f in (('end', k_end, t.sample_end), ('close', k_close, t.sample_close))
            for b in candidates(f, rng, exclude, 1 - t.p, k)]

# ---- R7 held-out sets
def game_job(g, k=4, delta=DELTA, rho_min=RHO_MIN):
    # one V_h game with seed g: per turn start, close-call stats and up to k end / close candidate boards excluding this
    # game's turn starts (incl. each turn's actual next start); build_sets takes the first one not excluded globally (R2)
    from td_train import BEST_VH, play
    from turn_solver import solve_turn
    recs = play((BEST_VH, g))[0]
    rng = np.random.default_rng(g)
    starts = {(r[0].tobytes(), r[1]) for r in recs}
    out = []
    for prog, p, *_ in recs:
        t = Turn(solve_turn(prog, p, BEST_VH), delta, rho_min)
        out.append(dict(prog=prog, p=int(p), end=candidates(t.sample_end, rng, set(starts), 1 - p, k),
                        close=candidates(t.sample_close, rng, set(starts), 1 - p, k), n_close=len(t.close),
                        gaps=[r[4] for r in t.close], exp_close=float(sum(r[5] for r in t.close)),
                        win_mass=float(t.end[t.win].sum()), bust=t.bust))
    return out

def build_sets(pool, seed0, n_games, n, delta=DELTA, rho_min=RHO_MIN):
    # n turn starts of V_h games seed0.. (visited) and, paired with each, its end and close board (absent if none survives
    # R2: excluded are all turn starts of these games and every board already taken, globally). Returns (turns, rows)
    from functools import partial
    flat = [r for g in pool.map(partial(game_job, delta=delta, rho_min=rho_min), range(seed0, seed0 + n_games), chunksize=1)
            for r in g]
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
