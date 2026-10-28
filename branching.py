import os
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

# ---- R7 held-out sets
def game_job(g, k=4, delta=DELTA, rho_min=RHO_MIN):
    # one V_h game with seed g: per turn start (reusing play's solve), close-call stats and up to k end / close candidate
    # boards; build_sets drops those that are any game's turn start or already taken (R2)
    from td_train import BEST_VH, play
    rng, out = np.random.default_rng(g), []
    def hook(sol):
        t = Turn(sol, delta, rho_min)
        out.append(dict(prog=t.prog, p=t.p, end=candidates(t.sample_end, rng, set(), 1 - t.p, k),
                        close=candidates(t.sample_close, rng, set(), 1 - t.p, k), n_close=len(t.close),
                        gaps=[r[4] for r in t.close], exp_close=float(sum(r[5] for r in t.close)),
                        win_mass=float(t.end[t.win].sum()), bust=t.bust))
    play((BEST_VH, g), hook)
    return out

def build_sets(pool, seed0, n_games, n, delta=DELTA, rho_min=RHO_MIN):
    # n turn starts of V_h games seed0.. (visited) and, paired with each, its end and close board (absent if none survives
    # R2: excluded are all turn starts of these games and every board already taken, globally).
    # Returns (turns, sets {name: (boards, to_move, turn-start row)})
    from functools import partial
    flat = [r for g in pool.map(partial(game_job, delta=delta, rho_min=rho_min), range(seed0, seed0 + n_games), chunksize=1)
            for r in g]
    sel = np.random.default_rng(seed0).choice(len(flat), min(n, len(flat)), replace=False)
    seen = {(r['prog'].tobytes(), r['p']) for r in flat}
    sets = {k: ([], [], []) for k in SETS}
    for j, i in enumerate(sel):
        r = flat[i]
        for k, b in [('visited', r['prog'])] + [(k, next((b for b in r[k] if (b.tobytes(), 1 - r['p']) not in seen), None))
                                                 for k in ('end', 'close')]:
            if b is not None:
                m = r['p'] if k == 'visited' else 1 - r['p']
                seen.add((b.tobytes(), m))
                for lst, x in zip(sets[k], (b, m, j)):
                    lst.append(x)
    return flat, {k: (np.array(b, np.int8).reshape(-1, 2, 13), np.array(m, np.int64), np.array(r, np.int64))
                  for k, (b, m, r) in sets.items()}

def cached_sets(path, pool, seed0, n_games, n, delta=DELTA, rho_min=RHO_MIN):
    # build_sets' sets, saved to / reused from path (an .npz) when built with the same arguments
    key = np.array([seed0, n_games, n, delta, rho_min], np.float64)
    if os.path.exists(path) and np.array_equal(np.load(path)['key'], key):
        return load_sets(path)
    sets = build_sets(pool, seed0, n_games, n, delta, rho_min)[1]
    np.savez(path + '.tmp.npz', key=key, **{k + s: a for k, v in sets.items() for s, a in zip(('_b', '_m', '_r'), v)})
    os.replace(path + '.tmp.npz', path)
    return sets

def load_sets(path):
    z = np.load(path)
    return {k: (z[k + '_b'], z[k + '_m'], z[k + '_r']) for k in SETS}

def set_keys(sets):
    return {(b.tobytes(), int(m)) for bs, ms, _ in sets.values() for b, m in zip(bs, ms)}

def resid_jobs(src, sets, rows=None, n_chunks=16):
    # td_train.residual_job args over the boards of sets on turn-start rows (None: all), and each board's (set, row)
    tag, B, M, R = [], [], [], []
    for k in SETS:
        b, m, r = sets[k]
        ok = np.ones(len(r), bool) if rows is None else np.isin(r, rows)
        tag += [k] * int(ok.sum())
        B.append(b[ok]), M.append(m[ok]), R.append(r[ok])
    B, M, R = np.concatenate(B), np.concatenate(M), np.concatenate(R)
    return [(src, B[c], M[c]) for c in np.array_split(np.arange(len(M)), n_chunks) if len(c)], np.array(tag), R

def resid_by_set(parts, tag, R):
    # residual_job results (in job order) -> {set: (turn-start rows, V - y)}
    yv = np.array([x for p in parts for x in p]).reshape(-1, 2)
    d = yv[:, 1] - yv[:, 0]
    return {k: (R[tag == k], d[tag == k]) for k in SETS if (tag == k).any()}
