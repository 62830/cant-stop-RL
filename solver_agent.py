import numpy as np
import fastrules as fr
from game import Action
from turn_solver import solve_turn
from baselines import V2

def tmp_of(track, progress):
    t = np.zeros(13, np.int8)
    t[track] = progress
    return t

class SolverAgent:
    # solves each turn once, re-solving from the current tmp when its row wasn't expanded (missing, horizon, win leaf);
    # depth k makes it a receding-horizon agent: picks are scored against horizon stop values, then it re-solves there.
    # v2_picks / v2_gostop delegate that decision to v2 (ablations)
    def __init__(self, value_fn, depth=None, v2_picks=False, v2_gostop=False):
        self.value_fn, self.depth, self.v2_picks, self.v2_gostop = value_fn, depth, v2_picks, v2_gostop
        self.v2 = V2()
        self.reset()

    def reset(self):
        self.sol, self.board, self.n_solves = None, None, 0

    def solution(self, prog, p, tmp):
        # a solution in which tmp is expandable (its go value is computed)
        board = (prog.tobytes(), p)
        k = fr.key(tmp)
        s = self.sol
        if s is None or board != self.board or k not in s.index or np.isnan(s.go[s.index[k]]):
            self.sol = solve_turn(prog, p, self.value_fn, self.depth, tmp if k else None)
            self.board = board
            self.n_solves += 1
        return self.sol

    def act(self, state):
        legal = state.get_legal_actions()
        if len(legal) == 1:
            return legal[0]
        go = state.is_wait_for_go()
        if (self.v2_gostop if go else self.v2_picks):
            return self.v2.act(state)
        prog, p, tmp = fr.from_state(state)
        sol = self.solution(prog, p, tmp)
        if go:
            return Action('go') if sol.should_go(tmp) else Action('stop')
        if np.isnan(sol.go[sol.row(tmp)]):
            return legal[0]  # going on from a winning stop (delegated go): every pick still wins
        return max(legal, key=lambda a: sol.value(tmp_of(*state.unsafe_pick(a))))
