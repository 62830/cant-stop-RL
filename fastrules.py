import random
import numpy as np
from itertools import product
from numba import njit, types
from numba.typed import Dict, List
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
ROLL_INDEX = {tuple(d): i for i, d in enumerate(ROLL_DICE.tolist())}
N_ROLLS = len(ROLL_DICE)

def roll_index(dice):
    return ROLL_INDEX[tuple(sorted(dice))]

@njit(cache=True)
def key(tmp):
    k = 0
    for c in range(2, 13):
        k += tmp[c] * STRIDE[c]
    return k

@njit(cache=True)
def decode(k):
    tmp = np.zeros(13, np.int8)
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
    ok = np.zeros((13, 13), np.bool_)
    m = n_markers(tmp)
    for c in range(2, 13):
        h = tmp[c] if tmp[c] > 0 else prog[p, c]
        free = tmp[c] > 0 or m < 3
        if blocked(prog, c) or not free:
            continue
        ok[c, 0] = h + 1 <= LEN[c]
        ok[c, c] = h + 2 <= LEN[c]
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
def enumerate_turn(prog, p, max_depth=-1, k0=0):
    # BFS over rolls x picks from tmp k0 (default empty); nodes at roll depth max_depth (-1: none) or where stopping wins are not expanded
    # returns keys (BFS order), roll depth at first discovery, key -> row
    seen = Dict.empty(types.int64, types.int64)
    keys, depth = List.empty_list(types.int64), List.empty_list(types.int64)
    seen[k0] = 0
    keys.append(k0)
    depth.append(0)
    mv = np.empty((MAX_PICKS, 2), np.int8)
    i = 0
    while i < len(keys) and depth[i] != max_depth:
        tmp = decode(keys[i])
        if score(apply_stop(prog, p, tmp), p) >= 3:
            i += 1
            continue
        ok = legal_moves(prog, p, tmp)
        reach = np.zeros((13, 13), np.bool_)
        for r in range(N_ROLLS):
            for j in range(roll_moves(ok, r, mv)):
                reach[mv[j, 0], mv[j, 1]] = True
        for a in range(2, 13):
            for b in range(13):
                if reach[a, b]:
                    k = move_key(prog, p, tmp, keys[i], a, b)
                    if k not in seen:
                        seen[k] = len(keys)
                        keys.append(k)
                        depth.append(depth[i] + 1)
        i += 1
    return np.asarray(keys), np.asarray(depth), seen

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
