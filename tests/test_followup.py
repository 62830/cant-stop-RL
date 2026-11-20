import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'analysis')]
from fastrules import LEN
from td_train import BEST_VH
from turn_solver import solve_turn
import strategy_profile as sp
import strategy_followup as sf

def test_fresh_top():
    t = np.zeros((4, 13), np.int8)
    t[0, 2], t[1, 2], t[2, 2], t[3, 2] = 2, 3, 3, 3  # tops on the 2nd decision, still topped on the 3rd; new turn on the 4th
    assert sp.fresh_top(t, np.array([0, 0, 0, 1])).tolist() == [False, True, False, True]

def test_race_count():
    own, opp = np.zeros(13, np.int8), np.zeros(13, np.int8)
    assert sf.race_count(own, opp) == 26 + 33 + 33  # 7, 6, 8 from empty
    own[7], opp[6] = 13, 11  # own 7 topped counts 0; 6 claimed by the opponent is out
    assert sf.race_count(own, opp) == 0 + 33 + 35
    assert (np.argsort(sf.STEP[2:8] * LEN[2:8]) == np.argsort([sp.turns_to_top(c) for c in range(2, 8)])).all()

def test_rule_value_and_offers():
    prog = np.zeros((2, 13), np.int8)
    prog[0, [6, 7, 8]] = (8, 9, 9)
    prog[1, [4, 10]] = (5, 5)
    sol = solve_turn(prog, 0, BEST_VH)
    tp = sp.turn_policy(sol)
    T, pb = tp['T'], np.nan_to_num(sol.pbust, nan=1.0)
    rv = lambda r, x=0.0: sf.rule_value(sol.child, sol.stop, sol.bust_val, sol.order, pb, T, prog[1], r, x)
    assert abs(rv(0) - sol.best[0]) < 1e-12 and rv(1) <= sol.best[0] + 1e-12 and max(rv(r, 0.15) for r in (2, 3, 4, 5)) <= sol.best[0] + 1e-12
    tot, ev = sf._offers(prog, 0, sol.index.rows, sol.best, ~tp['act_stop'], tp['reach'], T, 50, 0.5)
    assert 0 < tot[1] < tot[0] and tot[2] >= 0 and (ev[:, 3] >= 0).all()
    oc = sf._outcomes(sol.child, sol.best, tp['act_stop'], sol.order, T)
    assert abs(oc[0, 0] - tp['bustm'].sum()) < 1e-9
