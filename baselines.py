import random
from fraction_eval import Fraction_Eval_Agent_v2

class RandomAgent:
    def act(self, state):
        return random.choice(state.get_legal_actions())

    def reset(self):
        pass

V2 = Fraction_Eval_Agent_v2

class V2Tuned(Fraction_Eval_Agent_v2):
    threshold = 1.0

    def fraction_eval(self, state):
        # same as v2 but with a configurable go threshold
        tmp_track = state.get_tmp_track()
        tmp_progress = state.get_tmp_progress()
        track_len = state.get_track_len()
        track_state = state.get_track_state()
        cur_agent_idx = state.get_cur_agent_idx()

        fraction_score = 0.0
        for i in range(len(tmp_track)):
            l = track_len[tmp_track[i]]
            p = tmp_progress[i] - track_state[tmp_track[i]][cur_agent_idx]
            fraction_score += p / l
            if tmp_progress[i] == l:
                return False
        return fraction_score < self.threshold
