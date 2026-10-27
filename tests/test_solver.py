import os, sys, random, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fastrules as fr
from turn_solver import solve_turn, best_move
from game import Game
from baselines import V2

FRAC = fr.LEN[2:].astype(float)

def heur(progs, m):
    f = progs[:, :, 2:] / FRAC
    return 1 / (1 + np.exp(-3 * (f[:, m].sum(1) - f[:, 1 - m].sum(1))))

def const(c):
    return lambda progs, m: np.full(len(progs), c)

def _tmp(d):
    t = np.zeros(13, np.int8)
    for c, h in d.items():
        t[c] = h
    return t

def _boards(n_games=3, seed=0):
    random.seed(seed)
    out = []
    for _ in range(n_games):
        for s, _ in Game([V2(), V2()]).play(record=True)[:-1]:
            prog, p, tmp = fr.from_state(s)
            if not tmp.any() and not s.wait_for_go and not s.end_turn:
                out.append((prog, p))
    return out

def test_closed_form():
    # p owns 3 and 4, needs 2 more on column 2 to win; every other column is blocked
    prog = np.zeros((2, 13), np.int8)
    prog[1, 5:] = fr.LEN[5:]
    prog[0, 3:5] = fr.LEN[3:5]
    prog[0, 2] = 1
    c = 0.3
    sol = solve_turn(prog, 0, const(c))
    q4, q = 1 / 1296, 171 / 1296  # four 1s; at least two 1s
    b2 = q + (1 - q) * (1 - c)
    exact = q4 + (q - q4) * b2 + (1 - q) * (1 - c)
    assert len(sol.keys) == 3 and abs(sol.best[0] - exact) < 1e-12
    assert abs(sol.value(_tmp({2: 2})) - b2) < 1e-12 and sol.should_go(_tmp({2: 2}))
    assert sol.value(_tmp({2: 3})) == 1.0

def test_bust_odds():
    prog = np.zeros((2, 13), np.int8)
    for cols, pub in (((6, 7, 8), 0.92), ((2, 3, 12), 0.44)):
        tmp = _tmp({c: 1 for c in cols})
        sol = solve_turn(prog, 0, const(0.5), depth=1, tmp=tmp)
        s = fr.to_state(prog, 0, tmp)
        n = 0
        for d in np.ndindex(6, 6, 6, 6):
            s.set_dice([x + 1 for x in d])
            n += s.get_legal_actions()[0].is_null()
        print(f'{cols}: success {1 - sol.pbust[0]:.4f} (enum {1 - n / 1296:.4f}, published ~{pub})')
        assert abs(sol.pbust[0] - n / 1296) < 1e-12 and abs(1 - n / 1296 - pub) < 0.01

def test_depth_monotone():
    late = np.zeros((2, 13), np.int8)
    late[0, 2:] = fr.LEN[2:] - 2
    late[0, 7] = fr.LEN[7]
    late[1, 2:] = fr.LEN[2:] - 3
    late[1, [6, 8]] = fr.LEN[[6, 8]]
    late[0, [6, 8]] = 0
    exact = solve_turn(late, 0, heur)
    D = exact.depth.max()
    vals = [solve_turn(late, 0, heur, k).best[0] for k in range(D + 2)]
    print('late-game depth', D, np.round(vals, 4))
    assert np.all(np.diff(vals) >= -1e-12) and abs(vals[D] - exact.best[0]) < 1e-12 and vals[D + 1] == vals[D]
    prog, p = _boards(1)[10]
    vals = [solve_turn(prog, p, heur, k).best[0] for k in range(5)] + [solve_turn(prog, p, heur).best[0]]
    print('mid-game k=0..4, exact', np.round(vals, 4))
    assert np.all(np.diff(vals) >= -1e-12)

def test_monte_carlo():
    prog, p = _boards(1)[12]
    sol = solve_turn(prog, p, heur)
    rng = random.Random(0)
    leaves = []
    for _ in range(2000):
        tmp = np.zeros(13, np.int8)
        while True:
            s = fr.to_state(prog, p, tmp, dice=[rng.randint(1, 6) for _ in range(4)])
            legal = s.get_legal_actions()
            if legal[0].is_null():
                leaves.append(sol.bust_val)
                break
            outs = [_tmp(dict(zip(*s.unsafe_pick(a)))) for a in legal]
            tmp = max(outs, key=sol.value)
            if not sol.should_go(tmp):
                leaves.append(sol.stop[sol.row(tmp)])
                break
    m, se = np.mean(leaves), np.std(leaves) / np.sqrt(len(leaves))
    print(f'MC {m:.4f} +- {se:.4f} vs best(empty) {sol.best[0]:.4f}')
    assert abs(m - sol.best[0]) < 3 * se

def test_best_move_matches_state():
    prog, p = _boards(1)[12]
    sol = solve_turn(prog, p, heur)
    rng = random.Random(1)
    for _ in range(300):
        tmp = fr.decode(sol.keys[rng.randrange(len(sol.keys))])
        dice = [rng.randint(1, 6) for _ in range(4)]
        s = fr.to_state(prog, p, tmp, dice=dice)
        legal = s.get_legal_actions()
        bm = best_move(sol, tmp, dice)
        if legal[0].is_null():
            assert bm is None
        else:
            ref = max(sol.value(_tmp(dict(zip(*s.unsafe_pick(a))))) for a in legal)
            assert sol.value(bm[1]) == ref

def test_timing():
    prog = np.zeros((2, 13), np.int8)
    solve_turn(prog, 0, heur)
    t = time.time()
    sol = solve_turn(prog, 0, heur)
    t_empty = time.time() - t
    times, sizes = [], []
    boards = _boards(3)
    for prog, p in boards[::max(1, len(boards) // 20)]:
        t = time.time()
        sizes.append(len(solve_turn(prog, p, heur).keys))
        times.append(time.time() - t)
    t = time.time()
    solve_turn(np.zeros((2, 13), np.int8), 0, heur, 4)
    t_k4 = time.time() - t
    print(f'exact empty {t_empty:.3f}s ({len(sol.keys)} situations); k=4 empty {t_k4:.3f}s; '
          f'{len(times)} game boards median {np.median(times):.3f}s max {max(times):.3f}s, median size {int(np.median(sizes))}')
    assert t_empty < 1
