import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'analysis')]
import fastrules as fr
from heuristics import VH
from td_train import BEST_VH
from turn_solver import solve_turn
from headroom import Roller, regrets, GO, STOP

PROG = np.zeros((2, 13), np.int8)
PROG[0, [6, 7, 9]], PROG[1, [5, 8]] = (3, 6, 2), (4, 5)
OTHER = VH('top3', (4.0,), 0.0)

def roller(Ks, seed=1, ev=OTHER):
    return Roller(PROG, 0, Ks, seed, BEST_VH, OTHER, ev)

def test_k0_turn_start_is_evaluator():
    r = roller([0])
    v, c = r.cont(PROG, 0, 1)
    assert v[0] == 1 - OTHER(PROG[None], 1)[0] and c[0] == 0

def test_crn_and_self_regret():
    tmp = np.zeros(13, np.int8)
    tmp[7] = 8
    a, b = roller([2, None]), roller([2, None])
    Q = np.array([[np.r_[a.q(tmp, act, j)] for j in range(8)] for act in (GO, STOP, GO)])
    assert np.array_equal(Q[0], Q[2]) and np.array_equal(Q[0], [np.r_[b.q(tmp, GO, j)] for j in range(8)])
    for k in range(4):
        g = regrets(Q[:, :, k], int(Q[:, :, k].mean(1).argmax()))
        assert g['naive'] == 0 and g['x_best']

def test_k0_rollout_mean_is_solved_value():
    # X's own V as evaluator: finishing the turn from its start and scoring with V is unbiased for the solved value
    # and the control-variate estimate is then exactly the solved value, rollout by rollout
    r = roller([0], ev=BEST_VH)
    q = np.array([np.r_[r.q(np.zeros(13, np.int8), GO, j)] for j in range(400)])
    v = solve_turn(PROG, 0, BEST_VH).value(np.zeros(13, np.int8))
    assert abs(q[:, 0].mean() - v) < 4 * q[:, 0].std() / 20 and np.allclose(q[:, 1], v, atol=1e-12)

def test_cv_unbiased_and_tighter():
    tmp = np.zeros(13, np.int8)
    tmp[7] = 8
    r = roller([2], ev=lambda b, m: (BEST_VH(b, m) + OTHER(b, m)) / 2)
    q = np.array([np.r_[r.q(tmp, GO, j)] for j in range(300)])
    d = q[:, 0] - q[:, 1]
    assert abs(d.mean()) < 4 * d.std() / np.sqrt(300) and q[:, 1].std() < q[:, 0].std()
