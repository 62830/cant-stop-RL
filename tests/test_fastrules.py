import os, sys, random, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fastrules as fr
from game import Game, Action
from baselines import V2, RandomAgent

HIST = [1, 66, 836, 2814, 5351, 7763, 9518, 10218, 9659, 8079, 6054, 4073, 2440, 1281, 579, 224, 68, 13, 1]

def test_rolls():
    assert len(fr.ROLL_DICE) == 126 and abs(fr.ROLL_PROB.sum() - 1) < 1e-12

def test_count_fixture():
    prog = np.zeros((2, 13), np.int8)
    fr.enumerate_turn(prog, 0)
    t = time.time()
    keys, depth = fr.enumerate_turn(prog, 0)
    print(f'enumerate_turn: {time.time() - t:.3f}s')
    assert len(keys) == 69038 and len(set(keys.tolist())) == 69038
    assert np.bincount(depth).tolist() == HIST
    assert all(fr.key(fr.decode(k)) == k for k in keys[::97])

def _states(n_games=300, seed=0):
    random.seed(seed)
    out = []
    for g in range(n_games):
        agents = [V2(), RandomAgent()] if g % 2 else [RandomAgent(), V2()]
        out += [s for s, _ in Game(agents).play(record=True)[:-1] if not s.end_turn]
    return random.Random(seed).sample(out, 2000)

def _tmp(track, progress):
    t = np.zeros(13, np.int8)
    for c, h in zip(track, progress):
        t[c] = h
    return tuple(t)

def test_differential():
    out = np.empty((fr.MAX_PICKS, 13), np.int8)
    rng = random.Random(1)
    n_pick = n_stop = n_bust = 0
    for s in _states():
        prog, p, tmp = fr.from_state(s)
        t = fr.to_state(prog, p, tmp)
        assert t.track_state == s.track_state and t.score == s.score and t.tmp_track == sorted(s.tmp_track)
        if s.wait_for_go:
            s.update(Action('stop'))
            assert np.array_equal(fr.from_state(s)[0], fr.apply_stop(prog, p, tmp))
            assert (fr.score(fr.apply_stop(prog, p, tmp), p) >= 3) == s.is_terminal()
            n_stop += 1
            continue
        for dice in [s.dice] + [[rng.randint(1, 6) for _ in range(4)] for _ in range(3)]:
            s.set_dice(dice)
            legal = s.get_legal_actions()
            ref = set() if legal[0].is_null() else {_tmp(*s.unsafe_pick(a)) for a in legal}
            n = fr.picks(prog, p, tmp, fr.roll_index(dice), out)
            got = {tuple(o) for o in out[:n]}
            assert len(got) == n and got == ref, (s, dice)
            assert fr.bust(prog, p, tmp, fr.roll_index(dice)) == (not ref)
            n_pick += 1
            n_bust += not ref
    print(f'pick checks {n_pick} (busts {n_bust}), stop checks {n_stop}')
    assert n_bust > 0 and n_stop > 100
