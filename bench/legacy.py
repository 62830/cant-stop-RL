# frozen pre-Phase-4.5 solver (typed-Dict index, generic value_fn): the reference for exactness tests and "before" benchmarks
import numpy as np
from numba import njit, types
from numba.typed import Dict, List
import fastrules as fr
from fastrules import decode, score, apply_stop, legal_moves, roll_moves, move_key, N_ROLLS, MAX_PICKS
from turn_solver import Solution

@njit(cache=True)
def enumerate_turn(prog, p, max_depth=-1, k0=0):
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

@njit(cache=True)
def stop_boards(prog, p, keys):
    n = len(keys)
    boards = np.empty((n, 2, 13), np.int8)
    win = np.zeros(n, np.bool_)
    for i in range(n):
        boards[i] = apply_stop(prog, p, decode(keys[i]))
        win[i] = score(boards[i], p) >= 3
    return boards, win

@njit(cache=True)
def fill(prog, p, keys, depth, index, stop, bust_val, limit):
    n = len(keys)
    adv = np.zeros(n, np.int64)
    for i in range(n):
        t = decode(keys[i])
        for c in range(2, 13):
            if t[c] > 0:
                adv[i] += t[c] - prog[p, c]
    best, go, pbust = stop.copy(), np.full(n, np.nan), np.full(n, np.nan)
    cv = np.empty((13, 13))
    mv = np.empty((MAX_PICKS, 2), np.int8)
    for i in np.argsort(-adv):
        if depth[i] == limit or stop[i] >= 1.0:
            continue
        t = decode(keys[i])
        ok = legal_moves(prog, p, t)
        cv[:] = -1.0
        g = pb = 0.0
        for r in range(N_ROLLS):
            v = -1.0
            for j in range(roll_moves(ok, r, mv)):
                a, b = mv[j, 0], mv[j, 1]
                if cv[a, b] < 0:
                    cv[a, b] = best[index[move_key(prog, p, t, keys[i], a, b)]]
                v = max(v, cv[a, b])
            if v < 0:
                pb += fr.ROLL_PROB[r]
                v = bust_val
            g += fr.ROLL_PROB[r] * v
        go[i], pbust[i] = g, pb
        best[i] = max(stop[i], g)
    return best, go, pbust

def stop_values(prog, p, value_fn, keys):
    boards, win = stop_boards(prog, p, keys)
    ask = ~win & (keys != 0)
    v = value_fn(np.concatenate([boards[ask], prog[None]]), 1 - p)
    stop = np.where(win, 1.0, -np.inf)
    stop[ask] = 1 - v[:-1]
    return stop, 1 - v[-1]

def solve_turn(prog, p, value_fn, depth=None, tmp=None):
    k0 = 0 if tmp is None else fr.key(tmp)
    limit = -1 if depth is None else max(depth, int(k0 == 0))
    keys, dep, index = enumerate_turn(prog, p, limit, k0)
    stop, bust_val = stop_values(prog, p, value_fn, keys)
    best, go, pbust = fill(prog, p, keys, dep, index, stop, bust_val, limit)
    return Solution(prog, p, limit, keys, dep, index, best, stop, go, pbust, bust_val)
