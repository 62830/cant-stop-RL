import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'analysis')]
import fastrules as fr
from td_train import BEST_VH
from turn_solver import solve_turn
import strategy_profile as sp

def _empty():
    sol = solve_turn(np.zeros((2, 13), np.int8), 0, BEST_VH)
    return sol, sp.turn_policy(sol)

def test_reach_is_a_distribution_and_rule_value_matches():
    sol, tp = _empty()
    assert abs(sp.check_turn(sol, tp) - 1) < 1e-9  # also checks the forced optimal rule reproduces best[0]
    assert (tp['reach'][tp['dec']] > 0).all() and abs(sum(fr.ROLL_PROB[r] for r in range(fr.N_ROLLS) if tp['first'][r] < 0)) < 1e-12
    worse = sp.rule_value(sol, np.where(sp.feats(sol, tp)['steps'] >= 3, 1, 2))
    assert worse < sol.best[0]

def test_mirror_symmetric_value_gives_mirror_symmetric_opening():
    sol, tp = _empty()  # V_h only sees per-column fractions, so column c <-> 14 - c is exact
    T = sp.tmps(sol)
    mir = np.array([sol.index[fr.key(np.r_[t[:2], t[2:][::-1]].astype(np.int8))] for t in T])
    assert np.abs(sol.best - sol.best[mir]).max() < 1e-12
    cb = sp.summarize_turn(sol, tp)['col_banked']  # tied picks are broken in move order, not symmetrically
    assert max(abs(cb[c] - cb[14 - c]) for c in range(2, 13)) < 2e-3
