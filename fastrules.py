import random
import numpy as np
from itertools import product, combinations
from numba import njit
from game import State

# board: prog int8[2, 13] (index = column sum, banked heights); a column is blocked when max(prog[:, c]) == LEN[c]
# turn situation: tmp int8[13] absolute temp heights, 0 = no marker; key(tmp) is a unique int64
LEN = np.array([0, 0, 3, 5, 7, 9, 11, 13, 11, 9, 7, 5, 3], np.int8)
STRIDE = np.zeros(13, np.int64)
STRIDE[2:] = np.cumprod(np.r_[1, LEN[2:12].astype(np.int64) + 1])
MAX_PICKS = 6

def _rolls():
    count = {}
    for d in product(range(1, 7), repeat=4):
        k = tuple(sorted(d))
        count[k] = count.get(k, 0) + 1
    rolls = sorted(count)
    dice = np.array(rolls, np.int8)
    prob = np.array([count[r] for r in rolls]) / 1296
    pairs = np.zeros((len(rolls), 3, 2), np.int8)
    npairs = np.zeros(len(rolls), np.int8)
    for i, (a, b, c, d) in enumerate(rolls):
        ps = sorted({tuple(sorted(p)) for p in [(a + b, c + d), (a + c, b + d), (a + d, b + c)]})
        npairs[i] = len(ps)
        pairs[i, :len(ps)] = ps
    return dice, prob, pairs, npairs

ROLL_DICE, ROLL_PROB, ROLL_PAIRS, ROLL_NPAIRS = _rolls()
# the distinct (sum, sum) pairs over all rolls; a roll's moves depend only on its pairs, so the moves reachable by any
# roll, or a roll's best move, come from per-pair results
UPAIRS = np.array(sorted({tuple(p) for r in range(len(ROLL_DICE)) for p in ROLL_PAIRS[r, :ROLL_NPAIRS[r]].tolist()}), np.int8)
ROLL_UPAIR = np.array([[UPAIRS.tolist().index(ROLL_PAIRS[r, i].tolist()) if i < ROLL_NPAIRS[r] else -1 for i in range(3)]
                       for r in range(len(ROLL_DICE))], np.int64)
ROLL_INDEX = {tuple(d): i for i, d in enumerate(ROLL_DICE.tolist())}
N_ROLLS = len(ROLL_DICE)

def roll_index(dice):
    return ROLL_INDEX[tuple(sorted(dice))]

def _sit_offsets():
    # dense situation numbering: off[c1, c2, c3] (ascending tmp columns, 0-padded) + mixed radix of heights - 1
    off, n = np.full((13, 13, 13), -1, np.int64), 0
    for cols in [()] + [s for k in (1, 2, 3) for s in combinations(range(2, 13), k)]:
        off[tuple(cols) + (0,) * (3 - len(cols))] = n
        n += int(np.prod([LEN[c] for c in cols]))
    return off, n

SIT_OFF, N_SIT = _sit_offsets()

@njit(cache=True)
def slot(tmp):
    # dense index in [0, N_SIT) of a situation with <= 3 markers, -1 if more
    c1 = c2 = c3 = 0
    s, mul = 0, 1
    for c in range(2, 13):
        if tmp[c] > 0:
            if c3 > 0:
                return -1
            s += mul * (tmp[c] - 1)
            mul *= LEN[c]
            if c1 == 0:
                c1 = c
            elif c2 == 0:
                c2 = c
            else:
                c3 = c
    return SIT_OFF[c1, c2, c3] + s

@njit(cache=True)
def child_slot(prog, p, tmp, a, b):
    # slot(apply_move(prog, p, tmp, a, b)), tmp left unchanged
    ta, tb = tmp[a], tmp[b]
    push(prog, p, tmp, a, b)
    s = slot(tmp)
    tmp[a], tmp[b] = ta, tb
    return s

class Index:
    # key -> row of a solved turn, backed by a dense slot -> row array (-1: not in the turn)
    def __init__(self, rows, n):
        self.rows, self.n = rows, n

    def get(self, k):
        s = slot(decode(k))
        return -1 if s < 0 else int(self.rows[s])

    def __contains__(self, k):
        return self.get(k) >= 0

    def __getitem__(self, k):
        i = self.get(k)
        if i < 0:
            raise KeyError(k)
        return i

    def __len__(self):
        return self.n

@njit(cache=True)
def key(tmp):
    k = 0
    for c in range(2, 13):
        k += tmp[c] * STRIDE[c]
    return k

@njit(cache=True)
def decode(k):
    return decode_into(k, np.zeros(13, np.int8))

@njit(cache=True)
def decode_into(k, tmp):
    for c in range(12, 1, -1):
        tmp[c] = k // STRIDE[c]
        k -= tmp[c] * STRIDE[c]
    return tmp

@njit(cache=True)
def n_markers(tmp):
    return np.sum(tmp > 0)

@njit(cache=True)
def blocked(prog, c):
    return max(prog[0, c], prog[1, c]) == LEN[c]

@njit(cache=True)
def score(prog, p):
    return np.sum(prog[p, 2:] == LEN[2:])

@njit(cache=True)
def legal_moves(prog, p, tmp):
    # ok[c, 0]: column c alone can advance 1; ok[a, b] (2 <= a <= b): both pairs can be placed together
    return legal_into(prog, p, tmp, np.zeros((13, 13), np.bool_))

@njit(cache=True)
def legal_into(prog, p, tmp, ok):
    # legal_moves into a reused ok (only ok[c, 0] and ok[a, b >= a] are written)
    m = n_markers(tmp)
    for c in range(2, 13):
        h = tmp[c] if tmp[c] > 0 else prog[p, c]
        free = not blocked(prog, c) and (tmp[c] > 0 or m < 3)
        ok[c, 0] = free and h + 1 <= LEN[c]
        ok[c, c] = free and h + 2 <= LEN[c]
    for a in range(2, 13):
        for b in range(a + 1, 13):
            ok[a, b] = ok[a, 0] and ok[b, 0] and m + (tmp[a] == 0) + (tmp[b] == 0) <= 3
    return ok

@njit(cache=True)
def roll_moves(ok, r, out):
    # fills out[:n] with the distinct legal moves (a, b) for roll index r (b == 0: single column a); n == 0 is a bust
    n = 0
    for i in range(ROLL_NPAIRS[r]):
        a, b = ROLL_PAIRS[r, i, 0], ROLL_PAIRS[r, i, 1]
        if ok[a, b]:
            n = _add(out, n, a, b)
        else:
            if ok[a, 0]:
                n = _add(out, n, a, 0)
            if ok[b, 0]:
                n = _add(out, n, b, 0)
    return n

@njit(cache=True)
def _add(out, n, a, b):
    for i in range(n):
        if out[i, 0] == a and out[i, 1] == b:
            return n
    out[n, 0], out[n, 1] = a, b
    return n + 1

@njit(cache=True)
def apply_move(prog, p, tmp, a, b):
    t = tmp.copy()
    for c in (a, b):
        if c > 0:
            t[c] = (t[c] if t[c] > 0 else prog[p, c]) + 1
    return t

@njit(cache=True)
def move_key(prog, p, tmp, k, a, b):
    # key(apply_move(prog, p, tmp, a, b)) given k == key(tmp)
    k += STRIDE[a] * (1 if tmp[a] > 0 else prog[p, a] + 1)
    if b == a:
        k += STRIDE[a]
    elif b > 0:
        k += STRIDE[b] * (1 if tmp[b] > 0 else prog[p, b] + 1)
    return k

@njit(cache=True)
def picks(prog, p, tmp, r, out):
    # fills out[:n] with the distinct legal resulting tmp configs for roll index r; n == 0 is a bust
    mv = np.empty((MAX_PICKS, 2), np.int8)
    n = roll_moves(legal_moves(prog, p, tmp), r, mv)
    for i in range(n):
        out[i] = apply_move(prog, p, tmp, mv[i, 0], mv[i, 1])
    return n

@njit(cache=True)
def bust(prog, p, tmp, r):
    return roll_moves(legal_moves(prog, p, tmp), r, np.empty((MAX_PICKS, 2), np.int8)) == 0

@njit(cache=True)
def stop_wins(prog, p, tmp):
    # score(apply_stop(prog, p, tmp), p) >= 3
    w = 0
    for c in range(2, 13):
        w += prog[p, c] == LEN[c] or tmp[c] == LEN[c]
    return w >= 3

@njit(cache=True)
def apply_stop(prog, p, tmp):
    # returns the banked board; claims reset the opponent's progress; winner iff score(new, p) >= 3
    new = prog.copy()
    for c in range(2, 13):
        if tmp[c] > 0:
            new[p, c] = tmp[c]
            if tmp[c] == LEN[c]:
                new[1 - p, c] = 0
    return new

@njit(cache=True)
def push(prog, p, tmp, a, b):
    # apply_move in place (undo: tmp[a], tmp[b] = the old values)
    for c in (a, b):
        if c > 0:
            tmp[c] = (tmp[c] if tmp[c] > 0 else prog[p, c]) + 1

@njit(cache=True)
def _enumerate(prog, p, max_depth, k0):
    # child table (CSR over rows, unset past the expanded ones): for row i, entries ptr[i]:ptr[i + 1] are (move, row)
    # with move c - 2 for a single-column move (c, 0) some roll forces, 11 + u for a legal pair move UPAIRS[u]
    rows = np.full(N_SIT, -1, np.int32)
    keys, depth = np.empty(N_SIT, np.int64), np.empty(N_SIT, np.int64)
    ptr = np.zeros(N_SIT + 1, np.int64)
    cap = 16 * N_SIT
    cmv, crow = np.empty(cap, np.int8), np.empty(cap, np.int32)
    rows[slot(decode(k0))] = 0
    keys[0], depth[0], n, m = k0, 0, 1, 0
    reach, ok, tmp = np.zeros((13, 13), np.bool_), np.zeros((13, 13), np.bool_), np.zeros(13, np.int8)
    to = np.full((13, 13), -1, np.int32)
    i = 0
    while i < n and depth[i] != max_depth:
        ptr[i] = m
        decode_into(keys[i], tmp)
        if stop_wins(prog, p, tmp):
            i += 1
            continue
        legal_into(prog, p, tmp, ok)
        reach[:] = False
        for u in range(len(UPAIRS)):
            a, b = UPAIRS[u, 0], UPAIRS[u, 1]
            if ok[a, b]:
                reach[a, b] = True
            else:
                reach[a, 0] |= ok[a, 0]
                reach[b, 0] |= ok[b, 0]
        for a in range(2, 13):
            for b in range(13):
                if reach[a, b]:
                    s = child_slot(prog, p, tmp, a, b)
                    if rows[s] < 0:
                        rows[s] = n
                        keys[n], depth[n] = move_key(prog, p, tmp, keys[i], a, b), depth[i] + 1
                        n += 1
                    to[a, b] = rows[s]
        if m + 11 + len(UPAIRS) > cap:
            cap *= 2
            cmv, crow = np.concatenate((cmv, np.empty_like(cmv))), np.concatenate((crow, np.empty_like(crow)))
        for c in range(2, 13):
            if reach[c, 0]:
                cmv[m], crow[m] = c - 2, to[c, 0]
                m += 1
        for u in range(len(UPAIRS)):
            if ok[UPAIRS[u, 0], UPAIRS[u, 1]]:
                cmv[m], crow[m] = 11 + u, to[UPAIRS[u, 0], UPAIRS[u, 1]]
                m += 1
        i += 1
    ptr[i:n + 1] = m
    return keys[:n].copy(), depth[:n].copy(), rows, (ptr[:n + 1].copy(), cmv[:m].copy(), crow[:m].copy())

def enumerate_turn(prog, p, max_depth=-1, k0=0, children=False):
    # BFS over rolls x picks from tmp k0 (default empty); nodes at roll depth max_depth (-1: none) or where stopping wins are not expanded
    # returns keys (BFS order), roll depth at first discovery, key -> row (Index; .rows is the dense slot -> row array)
    # [, the child table of _enumerate]
    keys, depth, rows, child = _enumerate(prog, p, max_depth, k0)
    return (keys, depth, Index(rows, len(keys))) + ((child,) if children else ())

def from_state(s):
    # game.State -> (prog, player, tmp)
    assert s.num_agents == 2
    prog = np.zeros((2, 13), np.int8)
    for c in range(2, 13):
        prog[:, c] = s.track_state[c]
    tmp = np.zeros(13, np.int8)
    for c, h in zip(s.tmp_track, s.tmp_progress):
        tmp[c] = h
    return prog, s.cur_agent_idx, tmp

def to_state(prog, p, tmp=None, dice=None, wait_for_go=False):
    rs = random.getstate()
    s = State()
    random.setstate(rs)
    s.wait_for_go = wait_for_go
    s.cur_agent_idx = int(p)
    s.track_state = [[int(prog[0, c]), int(prog[1, c])] for c in range(13)]
    s.score = [int(score(prog, q)) for q in range(2)]
    if tmp is not None:
        s.tmp_track = [c for c in range(2, 13) if tmp[c] > 0]
        s.tmp_progress = [int(tmp[c]) for c in s.tmp_track]
    if dice is not None:
        s.dice = list(dice)
    return s
