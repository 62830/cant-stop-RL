import numpy as np
from numba import njit
import fastrules as fr

# value_fn(progs int8[n, 2, 13], to_move) -> float64[n]: P(to_move wins) at the start of to_move's turn

class Solution:
    def __init__(self, prog, p, limit, keys, depth, index, best, stop, go, pbust, bust_val):
        self.prog, self.p, self.limit, self.keys, self.depth, self.index = prog, p, limit, keys, depth, index
        self.best, self.stop, self.go, self.pbust, self.bust_val = best, stop, go, pbust, bust_val

    def row(self, tmp):
        k = fr.key(tmp)
        if k not in self.index:
            raise ValueError(f'situation {tmp.tolist()} is outside the solved turn (horizon {self.limit}); re-solve with tmp=')
        return self.index[k]

    def value(self, tmp):
        return self.best[self.row(tmp)]

    def should_go(self, tmp):
        i = self.row(tmp)
        return self.go[i] > self.stop[i]

@njit(cache=True)
def _stop_boards(prog, p, keys):
    n = len(keys)
    boards = np.empty((n, 2, 13), np.int8)
    win = np.zeros(n, np.bool_)
    for i in range(n):
        boards[i] = fr.apply_stop(prog, p, fr.decode(keys[i]))
        win[i] = fr.score(boards[i], p) >= 3
    return boards, win

@njit(cache=True)
def _fill(prog, p, keys, depth, index, stop, bust_val, limit):
    n = len(keys)
    adv = np.zeros(n, np.int64)
    for i in range(n):
        t = fr.decode(keys[i])
        for c in range(2, 13):
            if t[c] > 0:
                adv[i] += t[c] - prog[p, c]
    best, go, pbust = stop.copy(), np.full(n, np.nan), np.full(n, np.nan)
    cv = np.empty((13, 13))
    mv = np.empty((fr.MAX_PICKS, 2), np.int8)
    for i in np.argsort(-adv):
        if depth[i] == limit or stop[i] >= 1.0:
            continue
        t = fr.decode(keys[i])
        ok = fr.legal_moves(prog, p, t)
        cv[:] = -1.0
        g = pb = 0.0
        for r in range(fr.N_ROLLS):
            v = -1.0
            for j in range(fr.roll_moves(ok, r, mv)):
                a, b = mv[j, 0], mv[j, 1]
                if cv[a, b] < 0:
                    cv[a, b] = best[index[fr.move_key(prog, p, t, keys[i], a, b)]]
                v = max(v, cv[a, b])
            if v < 0:
                pb += fr.ROLL_PROB[r]
                v = bust_val
            g += fr.ROLL_PROB[r] * v
        go[i], pbust[i] = g, pb
        best[i] = max(stop[i], g)
    return best, go, pbust

def solve_turn(prog, p, value_fn, depth=None, tmp=None):
    # solves the rest of the turn from tmp (default: turn start); winning stops are not expanded
    # depth k: min-roll-distance horizon, situations first reachable in k rolls must stop (a realizable policy, so a
    # lower bound, monotone in k; exact once k > max distance). From the empty tmp k is raised to 1 (it can't stop).
    k0 = 0 if tmp is None else fr.key(tmp)
    limit = -1 if depth is None else max(depth, int(k0 == 0))
    keys, dep, index = fr.enumerate_turn(prog, p, limit, k0)
    boards, win = _stop_boards(prog, p, keys)
    ask = ~win & (keys != 0)
    v = value_fn(np.concatenate([boards[ask], prog[None]]), 1 - p)
    stop = np.where(win, 1.0, -np.inf)
    stop[ask] = 1 - v[:-1]
    bust_val = 1 - v[-1]
    best, go, pbust = _fill(prog, p, keys, dep, index, stop, bust_val, limit)
    return Solution(prog, p, limit, keys, dep, index, best, stop, go, pbust, bust_val)

def best_move(sol, tmp, roll):
    # roll: 4 dice or a roll index; returns (move (a, b), new tmp) maximizing best, or None on a bust
    if sol.depth[sol.row(tmp)] == sol.limit:
        raise ValueError(f'situation {tmp.tolist()} is on the depth-{sol.limit} horizon (forced stop); re-solve with tmp=')
    r = roll if np.isscalar(roll) else fr.roll_index(roll)
    mv = np.empty((fr.MAX_PICKS, 2), np.int8)
    n = fr.roll_moves(fr.legal_moves(sol.prog, sol.p, tmp), r, mv)
    if n == 0:
        return None
    outs = [((int(a), int(b)), fr.apply_move(sol.prog, sol.p, tmp, a, b)) for a, b in mv[:n]]
    return max(outs, key=lambda o: sol.value(o[1]))
