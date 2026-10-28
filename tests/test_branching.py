import os, sys, random
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fastrules as fr
import game
from branching import Turn, DELTA, legal, merge_close, _forward
from eval import _game, _pairs
from heuristics import VH
from solver_agent import SolverAgent
from td_train import BEST_VH, ZERO, play
from turn_solver import solve_turn

def _turns():
    recs = play((BEST_VH, 3))[0]
    return [Turn(solve_turn(prog, p, BEST_VH)) for prog, p, *_ in recs[::max(1, len(recs) // 4)]]

TURNS = _turns()
BR = (1, 2, DELTA, 0.01)

def test_mass_sums():
    for t in TURNS:  # Turn.forward asserts it too, also for forced-go / mid-turn starts
        assert abs(t.end.sum() + t.bust - 1) < 1e-9
        ok = np.isfinite(t.sol.go) & ~t.win & (np.arange(len(t.rho)) > 0)  # an expanded, non-winning mid-turn row
        i = int(np.argmax(np.where(ok, t.rho, -1)))
        assert ok[i]
        rho, end, bust, _ = t.forward(i, True)
        assert abs(end.sum() + bust - 1) < 1e-9

def test_sampled_boards_legal():
    rng = np.random.default_rng(0)
    for t in TURNS:
        for b in [t.sample_end(rng, set()) for _ in range(20)] + [t.sample_close(rng, set()) for _ in range(5)]:
            if b is None:
                continue
            assert legal(b) and not fr.score(b, t.p) >= 3
            assert (b[1 - t.p] <= t.prog[1 - t.p]).all() and (b[t.p] >= t.prog[t.p]).all()  # banked by p; opponent to move
    assert not legal(np.array([[0, 0, 3] + [0] * 10, [0, 0, 1] + [0] * 10], np.int8))

def test_close_call_detected():
    # hand-built: at a reached situation where pi goes, set stop just under go -> a go/stop close call rejecting its stop board
    t = TURNS[0]
    s = t.sol
    i = next((i for i in np.argsort(-t.rho) if i > 0 and s.go[i] > s.stop[i] + DELTA and t.rho[i] >= 0.05), None)
    assert i is not None, 'no reached situation where pi clearly goes; pick another test game'
    stop = s.stop.copy()
    stop[i] = s.go[i] - DELTA / 2
    recs = merge_close(_forward(t.order, *s.child, s.best, stop, s.go, 0, False, DELTA, 0.01)[3])
    rec = next((r for r in recs if r[0] == 0 and r[1] == i), None)
    assert rec is not None, f'hand-built go/stop close call at row {i} not detected'
    assert rec[2] == 1 and abs(rec[4] - DELTA / 2) < 1e-12
    b = t.rejected_board(np.random.default_rng(0), rec, set())
    assert (b == fr.apply_stop(t.prog, t.p, fr.decode(s.keys[i]))).all()

def _reference_play(v, seed):
    # td_train.play before branching: (board, to_move, y) at each turn start
    random.seed(seed)
    agent, s, rec = SolverAgent(v), game.State(), []
    while not s.is_terminal():
        if not s.tmp_track and not s.wait_for_go and not s.end_turn:
            prog, p, _ = fr.from_state(s)
            rec.append((prog, p, agent.solution(prog, p, ZERO).value(ZERO)))
        s, _ = s.update(agent.act(s))
    return rec

def _same(a, b):
    return len(a) == len(b) and all((x[0] == y[0]).all() and x[1:3] == y[1:3] for x, y in zip(a, b))

def test_k0_identical_and_branching_tags():
    seed = 11
    ref = _reference_play(BEST_VH, seed)
    k0 = play((BEST_VH, seed, (0, 0, DELTA, 0.01)))
    assert _same(ref, k0[0]) and _same(ref, play((BEST_VH, seed))[0]) and k0[3] == [] and k0[2] == len(ref)
    recs, _, solves, extra, gaps = play((BEST_VH, seed, BR))
    assert _same(ref, recs)  # branching never perturbs the game
    assert len(gaps) == len(recs) and solves == len(recs) + len(extra)
    seen = {(r[0].tobytes(), r[1]) for r in recs}
    for b, m, y, tag in extra:
        assert tag in ('end', 'close') and legal(b) and (b.tobytes(), m) not in seen
        seen.add((b.tobytes(), m))
        assert y == solve_turn(b, m, BEST_VH).value(ZERO)
    n = {k: sum(e[3] == k for e in extra) for k in ('end', 'close')}
    assert 0 < n['end'] <= len(recs) and 0 < n['close'] <= 2 * len(recs)
    # each extra board ends a turn of the game: p banked (a claim can reset the opponent's column) or busted; opponent to move
    for b, m, *_ in extra:
        assert any(p == 1 - m and (b[1 - p] <= prog[1 - p]).all() and (b[p] >= prog[p]).all()
                   for prog, p, *_ in recs)

def test_paired_eval():
    v = VH('top3', (4.0,))
    mk = lambda: SolverAgent(v)
    assert _pairs((mk, mk, range(4))) == [0.5] * 4  # deterministic self-play: each pair splits exactly
    dice = []
    roll = game.State.roll_dice
    def rec(self):
        roll(self)
        dice[-1].append(tuple(self.dice))
    game.State.roll_dice = rec
    try:
        for a_first in (True, False):
            dice.append([])
            _game(mk(), mk(), 5, a_first)
    finally:
        game.State.roll_dice = roll
    assert dice[0] == dice[1] and len(dice[0]) > 10
