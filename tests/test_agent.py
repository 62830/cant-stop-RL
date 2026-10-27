import os, sys, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fastrules as fr
from game import State
from baselines import V2
from heuristics import VH, KINDS, features, fit
from solver_agent import SolverAgent, tmp_of
from turn_solver import solve_turn

V = VH('top3', (4.0,))

def _play(agents, seed, check=None):
    random.seed(seed)
    s = State()
    for a in agents:
        a.reset()
    while not s.is_terminal():
        legal = s.get_legal_actions()
        a = agents[s.cur_agent_idx].act(s)
        assert a in legal
        if check:
            check(agents[s.cur_agent_idx], s, a, legal)
        s, _ = s.update(a)
    return s.get_winner()

def test_legal_actions():
    for i, kw in enumerate([{}, dict(depth=2), dict(depth=4), dict(v2_picks=True), dict(v2_gostop=True)]):
        assert _play([SolverAgent(V, **kw), V2()][::1 - 2 * (i % 2)], i) in (0, 1)

def test_picks_are_argmax():
    n = [0]
    def check(agent, s, a, legal):
        if not isinstance(agent, SolverAgent) or s.wait_for_go or len(legal) < 2:
            return
        prog, p, tmp = fr.from_state(s)
        sol = solve_turn(prog, p, V)
        vals = [sol.value(tmp_of(*s.unsafe_pick(b))) for b in legal]
        assert sol.value(tmp_of(*s.unsafe_pick(a))) == max(vals)
        n[0] += 1
    _play([SolverAgent(V), SolverAgent(V)], 7, check)
    assert n[0] > 5

def test_gostop_matches_solver():
    def check(agent, s, a, legal):
        if isinstance(agent, SolverAgent) and s.wait_for_go:
            prog, p, tmp = fr.from_state(s)
            assert a.is_go() == solve_turn(prog, p, V).should_go(tmp)
    _play([SolverAgent(V), V2()], 3, check)

def test_antisymmetric():
    rng = np.random.default_rng(0)
    progs = np.zeros((200, 2, 13), np.int8)
    for c in range(2, 13):
        progs[:, :, c] = rng.integers(0, fr.LEN[c] + 1, (200, 2))
        top = progs[:, :, c].max(1) == fr.LEN[c]
        progs[top, :, c] = np.where(progs[top, :, c] == fr.LEN[c], fr.LEN[c], 0)
    swapped = progs[:, ::-1]
    for kind, th in (('top3', (4.0,)), ('sum', (2.0,)), ('claim_top3', (3.0, 1.5))):
        v = VH(kind, th)
        for p in (0, 1):
            assert np.allclose(v(progs, p) + v(progs, 1 - p), 1)
            assert np.allclose(v(progs, p), v(swapped, 1 - p))

def test_features_and_fit():
    prog = np.zeros((1, 2, 13), np.int8)
    prog[0, 0, [7, 2, 3, 4]] = [13, 1, 1, 4]
    assert np.allclose(features(prog, 0, 'top3'), 1 + 1 / 3 + 4 / 7)
    assert np.allclose(features(prog, 1, 'claim_top3'), [[-(1 + 1 / 3 + 4 / 7), -1]])
    rng = np.random.default_rng(0)
    X = rng.normal(size=(20000, 1))
    y = (rng.random(20000) < 1 / (1 + np.exp(-(2 * X[:, 0] + 0.3)))).astype(float)
    th, b = fit(X, y)
    assert abs(th[0] - 2) < 0.1 and abs(b - 0.3) < 0.1
    assert set(KINDS) == {'top3', 'sum', 'claim_top3'}
