import os, sys, copy, random
from functools import partial
from itertools import product
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fastrules as fr
import game
from game import Game
from baselines import V2
from solver_agent import SolverAgent
from td_train import BEST_VH
from eval import roll_luck, _game, evaluate_paired, evaluate_paired_detail

def _roll_states(n_games=6, seed=0):
    # (state, mover's P(bust)) at every chance event of a few solver-vs-v2 games
    random.seed(seed)
    ref, out = SolverAgent(BEST_VH), []
    for g in range(n_games):
        for s, _ in Game([SolverAgent(BEST_VH), V2()][::1 - 2 * (g % 2)]).play(record=True)[:-1]:
            if not s.wait_for_go and not s.end_turn:
                prog, p, tmp = fr.from_state(s)
                sol = ref.solution(prog, p, tmp)
                out.append((s, tmp.any(), sol.pbust[sol.row(tmp)]))
    return out

def test_roll_luck_unbiased():
    # mean of A_t over all 1296 dice outcomes is E_t: turn starts, the riskiest mid-turn states and random others
    sts = _roll_states()
    start = [x for x in sts if not x[1]]
    mid = sorted([x for x in sts if x[1]], key=lambda x: -x[2])
    rng = random.Random(1)
    risky = fr.to_state(np.zeros((2, 13), np.int8), 1, np.array([0, 0, 1, 1] + [0] * 8 + [2], np.int8))  # 2, 3, 12 open
    pick = rng.sample(start, 7) + mid[:6] + rng.sample(mid[6:], 6) + [(risky, True, None)]
    assert mid[0][2] > 0.3 and len(pick) == 20
    ref, dev = SolverAgent(BEST_VH), []
    for s, _, _ in pick:
        s = copy.deepcopy(s)
        vals = []
        for d in product(range(1, 7), repeat=4):
            s.dice = list(d)
            vals.append(roll_luck(ref, s))
        assert len({e for _, e in vals}) == 1
        dev.append(abs(np.mean([a for a, _ in vals]) - vals[0][1]))
    print(f"max |mean A - E| = {max(dev):.2e}")
    assert max(dev) < 1e-9

def test_luck_only_at_rolls():
    # with A = R the adjustment leaves every winner unchanged and there is exactly one chance event per dice roll
    rolls = [0]
    roll = game.State.roll_dice
    def rec(self):
        rolls[0] += 1
        roll(self)
    mk = partial(SolverAgent, BEST_VH)
    for seed in range(3):
        for a_first in (True, False):
            w0 = _game(mk(), V2(), seed, a_first)[0]
            game.State.roll_dice = rec
            try:
                rolls[0] = 0
                a = mk()
                w, luck, n = _game(a, V2(), seed, a_first, (a, SolverAgent(BEST_VH)))  # A doubles as R
            finally:
                game.State.roll_dice = roll
            assert w == w0 and n == rolls[0] > 10 and np.isfinite(luck)

def test_small_n():
    # fewer pairs than jobs (empty jobs), n = 1 (finite zero-width CI); ref path returns the adjusted (p, (lo, hi))
    for n in range(1, 21):
        p, (lo, hi) = evaluate_paired(V2, V2, n, workers=4, seed=n)
        assert lo <= p <= hi and np.isfinite([lo, hi]).all()
    d = evaluate_paired_detail(V2, partial(SolverAgent, BEST_VH), 3, workers=2, seed=1, ref=BEST_VH)
    assert d['result'].shape == (3, 2) and (d['n_events'] > 0).all()
    assert evaluate_paired(V2, partial(SolverAgent, BEST_VH), 3, workers=2, seed=1, ref=BEST_VH) == d['adj']
