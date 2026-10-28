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
    tmp = np.zeros(13, np.int8)
    for i in range(n):
        fr.decode_into(keys[i], tmp)
        boards[i] = fr.apply_stop(prog, p, tmp)
        win[i] = fr.stop_wins(prog, p, tmp)
    return boards, win

@njit(cache=True)
def _wins(prog, p, keys):
    win = np.zeros(len(keys), np.bool_)
    tmp = np.zeros(13, np.int8)
    for i in range(len(keys)):
        win[i] = fr.stop_wins(prog, p, fr.decode_into(keys[i], tmp))
    return win

def stop_values(prog, p, value_fn, keys):
    # stop[i] = P(p wins) after banking keys[i] (1 on a win, -inf for the empty tmp); bust_val = P(p wins) after a bust.
    # value_fns with a banked method (value_net.FusedNet) evaluate the banked boards from the keys directly
    if hasattr(value_fn, 'banked'):
        win = _wins(prog, p, keys)
        ask = ~win & (keys != 0)
        v = value_fn.banked(prog, p, keys[ask])
    else:
        boards, win = _stop_boards(prog, p, keys)
        ask = ~win & (keys != 0)
        v = value_fn(np.concatenate([boards[ask], prog[None]]), 1 - p)
    stop = np.where(win, 1.0, -np.inf)
    stop[ask] = 1 - v[:-1]
    return stop, 1 - v[-1]

@njit(cache=True)
def _fill(prog, p, keys, depth, child, stop, bust_val, limit):
    # children advance >= 1, so visiting rows by decreasing total advance finalizes them before their parents
    n = len(keys)
    adv = np.zeros(n, np.int64)
    t = np.zeros(13, np.int8)
    for i in range(n):
        fr.decode_into(keys[i], t)
        for c in range(2, 13):
            if t[c] > 0:
                adv[i] += t[c] - prog[p, c]
    rank = adv.max() - adv
    start = np.zeros(rank.max() + 2, np.int64)
    for i in range(n):
        start[rank[i] + 1] += 1
    start = np.cumsum(start)
    order = np.empty(n, np.int64)
    for i in range(n):
        order[start[rank[i]]] = i
        start[rank[i]] += 1
    ptr, cmv, crow = child
    best, go, pbust = stop.copy(), np.full(n, np.nan), np.full(n, np.nan)
    sv, uv = np.empty(13), np.empty(len(fr.UPAIRS))
    for i in order:
        if depth[i] == limit or stop[i] >= 1.0:
            continue
        sv[:] = -1.0
        for e in range(ptr[i], ptr[i + 1]):
            if cmv[e] < 11:
                sv[cmv[e] + 2] = best[crow[e]]
        for u in range(len(uv)):
            uv[u] = max(sv[fr.UPAIRS[u, 0]], sv[fr.UPAIRS[u, 1]])
        for e in range(ptr[i], ptr[i + 1]):
            if cmv[e] >= 11:
                uv[cmv[e] - 11] = best[crow[e]]
        g = pb = 0.0
        for r in range(fr.N_ROLLS):
            v = -1.0
            for j in range(fr.ROLL_NPAIRS[r]):
                v = max(v, uv[fr.ROLL_UPAIR[r, j]])
            if v < 0:
                pb += fr.ROLL_PROB[r]
                v = bust_val
            g += fr.ROLL_PROB[r] * v
        go[i], pbust[i] = g, pb
        best[i] = max(stop[i], g)
    return best, go, pbust

_graphs = {}  # (p, from turn start) -> (graph key, graph) of p's last such solve; a turn after a bust reuses it

def turn_graph(prog, p, limit, k0):
    # fr.enumerate_turn(..., children=True) (read-only), which depends on the board only via p's row and the blocked columns
    gk = (prog[p].tobytes(), (prog.max(0) == fr.LEN).tobytes(), limit, k0)
    hit = _graphs.get((p, k0 == 0))
    if hit is None or hit[0] != gk:
        keys, depth, index, child = fr.enumerate_turn(prog, p, limit, k0, children=True)
        for a in (keys, depth, index.rows) + child:
            a.flags.writeable = False
        hit = _graphs[(p, k0 == 0)] = (gk, (keys, depth, index, child))
    return hit[1]

def solve_turn(prog, p, value_fn, depth=None, tmp=None):
    # solves the rest of the turn from tmp (default: turn start); winning stops are not expanded
    # depth k: min-roll-distance horizon, situations first reachable in k rolls must stop (a realizable policy, so a
    # lower bound, monotone in k; exact once k > max distance). From the empty tmp k is raised to 1 (it can't stop).
    k0 = 0 if tmp is None else fr.key(tmp)
    limit = -1 if depth is None else max(depth, int(k0 == 0))
    keys, dep, index, child = turn_graph(prog, p, limit, k0)
    stop, bust_val = stop_values(prog, p, value_fn, keys)
    best, go, pbust = _fill(prog, p, keys, dep, child, stop, bust_val, limit)
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
