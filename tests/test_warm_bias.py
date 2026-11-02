import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'analysis')]
from heuristics import VH
from td_train import BEST_VH
from warm_bias import NAMES, game_job, position_metrics, boot_weights

def test_reference_and_identity():
    other = VH('top3', (4.0,), 0.0)
    srcs = dict(zip(NAMES, (other, BEST_VH, BEST_VH, other, other, other)))  # T := H, S74.. := S
    starts, decs = game_job((srcs, 'H', 7))
    reg, ch = np.array([d[3] for d in decs]), np.array([d[2] for d in decs])
    assert (reg[:, 0] == 0).all() and (reg >= -1e-12).all() and (ch[:, 1] == ch[:, 2]).all()
    sol = np.array([s[3] for s in starts])
    V = {x: sol[:, j] for j, x in enumerate(NAMES)}
    r = position_metrics(V, np.zeros(len(sol), int), 1, boot_weights([1], 10))
    assert r['S']['mean_abs']['est'] == 0 and r['S74']['mean_abs']['est'] == 0
    assert abs(r['beta']['est'] - 1) < 1e-12 and abs(r['ratio_T_H']['est'] - 1) < 1e-12
