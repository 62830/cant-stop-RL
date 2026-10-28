# Phase 4.5 exactness: the fast solver (dense index, child table, turn-graph cache, FusedNet) vs the frozen bench/legacy.py
import os, sys, pickle
import numpy as np
import torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'bench')]
import fastrules as fr
import turn_solver as ts
import legacy
from heuristics import VH
from value_net import NetV, FusedNet, make_net
from test_solver import _boards
from test_value_net import boards as random_boards

VHB = VH('claim_top3', (0.6431, 0.6591), 0.2690)
NET = os.path.join(ROOT, 'runs', 'phase4', 'it0041.pt')

def _real():
    # turn starts of V2 games (early to near-won), plus boards where p owns 2 columns so many stops win
    out = _boards(3)[::3]
    prog = np.zeros((2, 13), np.int8)
    prog[0, [3, 11]] = fr.LEN[[3, 11]]
    prog[1, 2:] = [1, 0, 2, 4, 5, 3, 6, 2, 1, 0, 2]
    return out + [(prog, 0), (prog[::-1].copy(), 1)]

def _nets():
    torch.manual_seed(0)
    return [NetV.load(NET), NetV(make_net((32, 16)))]

def test_situation_index():
    assert fr.N_SIT == 70224
    keys, _, index = fr.enumerate_turn(np.zeros((2, 13), np.int8), 0)
    slots = np.array([fr.slot(fr.decode(k)) for k in keys])
    assert len(set(slots.tolist())) == len(keys) and slots.min() >= 0 and slots.max() < fr.N_SIT
    assert (index.rows[slots] == np.arange(len(keys))).all() and (index.rows >= 0).sum() == len(keys) == len(index)
    prog, p = _boards(1)[8]
    keys, _, index = fr.enumerate_turn(prog, p)
    assert keys[5] in index and index[keys[5]] == 5 and fr.key(fr.decode(keys[0]) + 1) not in index
    assert all(k not in index for k in (-1, fr.STRIDE[12] * (fr.LEN[12] + 1), keys[5] + fr.STRIDE[12] * 100))

def test_fused_matches_netv():
    n = 0
    for v in _nets():
        f = v.fused()
        for prog, p in _real():
            keys = fr.enumerate_turn(prog, p)[0]
            a, ab = legacy.stop_values(prog, p, v, keys)
            b, bb = ts.stop_values(prog, p, f, keys)
            assert np.array_equal(a == 1, b == 1) and np.array_equal(np.isinf(a), np.isinf(b))
            fin = np.isfinite(a) & (a < 1)
            np.testing.assert_allclose(b[fin], a[fin], rtol=0, atol=1e-5)
            assert abs(ab - bb) < 1e-5
            n += fin.sum()
        progs = random_boards(500, 3)
        np.testing.assert_array_equal(f(progs, 0), v(progs, 0))
    print(f'{n} net stop values checked')
    assert n > 10 ** 5
    f2 = pickle.loads(pickle.dumps(f))
    assert isinstance(f2, FusedNet) and all(np.array_equal(x, y) for x, y in zip(f.w, f2.w))

def test_solver_matches_legacy():
    v = NetV.load(NET)
    f = v.fused()
    for prog, p in _real()[::2]:
        for depth, tmp in ((None, None), (2, None), (None, fr.decode(fr.enumerate_turn(prog, p)[0][7]))):
            ts._graphs.clear()
            a, b = legacy.solve_turn(prog, p, VHB, depth, tmp), ts.solve_turn(prog, p, VHB, depth, tmp)
            assert np.array_equal(a.keys, b.keys) and np.array_equal(a.depth, b.depth)
            assert np.abs(a.best - b.best).max() <= 1e-9 and np.array_equal(np.isnan(a.go), np.isnan(b.go))
            np.testing.assert_allclose(b.go, a.go, rtol=0, atol=1e-9)
            np.testing.assert_array_equal(b.pbust, a.pbust)
        a, b = legacy.solve_turn(prog, p, v), ts.solve_turn(prog, p, f)
        assert np.abs(a.best - b.best).max() <= 1e-5
        np.testing.assert_allclose(b.go, a.go, rtol=0, atol=1e-5)

def test_graph_cache_across_bust():
    # same mover row, the opponent moved (no new block): the cached graph gives the cold solve; a new block misses
    prog, p = _boards(1)[6]
    later = prog.copy()
    q = 1 - p
    c = next(c for c in range(2, 13) if later[p, c] == 0 and later[q, c] < fr.LEN[c] - 1)
    later[q, c] += 1
    ts.solve_turn(prog, p, VHB)
    hit = ts.solve_turn(later, p, VHB)
    assert ts._graphs[(p, True)][1][0] is hit.keys
    ts._graphs.clear()
    np.testing.assert_array_equal(hit.best, ts.solve_turn(later, p, VHB).best)
    claimed = later.copy()
    claimed[q, c] = fr.LEN[c]
    sol = ts.solve_turn(claimed, p, VHB)
    assert sol.keys is not hit.keys
    np.testing.assert_array_equal(sol.best, legacy.solve_turn(claimed, p, VHB).best)

def test_fused_fallback_and_snapshot():
    torch.manual_seed(1)
    prog, p = _boards(1)[8]
    for hidden in ((128,), (64, 32, 16)):
        v = NetV(make_net(hidden))
        assert v.fused() is v
        a, b = legacy.solve_turn(prog, p, v), ts.solve_turn(prog, p, v.fused())
        np.testing.assert_array_equal(a.best, b.best)
    v = NetV(make_net((32, 16)))
    f = v.fused()
    progs = random_boards(50, 4)
    before = f(progs, 0)
    with torch.no_grad():
        v.net[0].weight += 1.0
    np.testing.assert_array_equal(f(progs, 0), before)

def test_graph_cache_keeps_turn_start():
    prog, p = _boards(1)[6]
    ts._graphs.clear()
    start = ts.solve_turn(prog, p, VHB)
    ts.solve_turn(prog, p, VHB, tmp=fr.decode(start.keys[7]))
    assert ts.solve_turn(prog, p, VHB).keys is start.keys
    assert not start.keys.flags.writeable and not start.index.rows.flags.writeable
