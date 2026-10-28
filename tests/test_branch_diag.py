import os, sys
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'analysis'))
import fastrules as fr
from branch_diag import Turn, BEST_VH, DELTA, legal, merge_close, _forward
from td_train import play

def _turns():
    recs = play((BEST_VH, 3))[0]
    return [Turn(prog, p, BEST_VH) for prog, p, *_ in recs[::max(1, len(recs) // 4)]]

TURNS = _turns()

def test_mass_sums():
    for t in TURNS:  # Turn.forward asserts it too, also for forced-go / mid-turn starts
        assert abs(t.end.sum() + t.bust - 1) < 1e-9
        i = int(np.argmax(np.where(np.arange(len(t.rho)) > 0, t.rho, 0)))
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
    i = next(i for i in np.argsort(-t.rho) if i > 0 and s.go[i] > s.stop[i] + DELTA and t.rho[i] >= 0.05)
    stop = s.stop.copy()
    stop[i] = s.go[i] - DELTA / 2
    recs = merge_close(_forward(t.order, *t.child, s.best, stop, s.go, 0, False, DELTA, 0.01)[3])
    rec = next(r for r in recs if r[0] == 0 and r[1] == i)
    assert rec[2] == 1 and abs(rec[4] - DELTA / 2) < 1e-12
    b = t.rejected_board(np.random.default_rng(0), rec, set())
    assert (b == fr.apply_stop(t.prog, t.p, fr.decode(s.keys[i]))).all()
