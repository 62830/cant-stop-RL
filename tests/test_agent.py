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
    n = [0]
    def check(agent, s, a, legal):
        if isinstance(agent, SolverAgent) and s.wait_for_go:
            prog, p, tmp = fr.from_state(s)
            assert a.is_go() == solve_turn(prog, p, V).should_go(tmp)
            n[0] += 1
    _play([SolverAgent(V), V2()], 3, check)
    assert n[0] > 5

def test_depth_resolve():
    # whenever the depth-2 agent re-solves mid-turn, its decision matches a fresh depth-2 solve from that tmp
    random.seed(5)
    agent, s, n = SolverAgent(V, depth=2), State(), 0
    agents = [agent, V2()]
    while not s.is_terminal():
        before = agent.n_solves
        a = agents[s.cur_agent_idx].act(s)
        if s.cur_agent_idx == 0 and agent.n_solves > before and s.tmp_track:
            prog, p, tmp = fr.from_state(s)
            sol = solve_turn(prog, p, V, 2, tmp)
            if s.wait_for_go:
                assert a.is_go() == sol.should_go(tmp)
            else:
                legal = s.get_legal_actions()
                assert sol.value(tmp_of(*s.unsafe_pick(a))) == max(sol.value(tmp_of(*s.unsafe_pick(b))) for b in legal)
            n += 1
        s, _ = s.update(a)
    assert n > 3

def test_pick_after_winning_stop():
    # a delegated 'go' from a winning stop must not crash the solver's pick
    prog = np.zeros((2, 13), np.int8)
    prog[0, 3:5] = fr.LEN[3:5]
    s = fr.to_state(prog, 0, tmp_of([2], [3]), dice=[1, 1, 3, 4])
    assert SolverAgent(V).act(s) in s.get_legal_actions()

def test_antisymmetric():
    # valid boards: heights < LEN, or one player claimed the column and the other is reset to 0
    rng = np.random.default_rng(0)
    progs = np.zeros((200, 2, 13), np.int8)
    for c in range(2, 13):
        progs[:, :, c] = rng.integers(0, fr.LEN[c], (200, 2))
        claim = rng.random(200) < 0.15
        who = rng.integers(0, 2, 200)
        progs[claim, who[claim], c] = fr.LEN[c]
        progs[claim, 1 - who[claim], c] = 0
    assert not (progs[:, :, 2:] == fr.LEN[2:]).all(1).any()
    swapped = progs[:, ::-1]
    for kind, th, b in (('top3', (4.0,), 0), ('sum', (2.0,), 0), ('claim_top3', (3.0, 1.5), 0), ('claim_top3', (0.6431, 0.6591), 0.269)):
        v = VH(kind, th, b)
        for p in (0, 1):
            if b == 0:
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
