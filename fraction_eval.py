import random 
from game import Action, State

class Fraction_Eval_Agent:
    def __init__(self):
        # choose random track, on average 3 tracks takes 7.5 turns to top
        # so every turn we should complete 3/7.5 = 0.4 track
        pass

    def fraction_eval(self, state):
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
        return fraction_score < 0.6

    def act(self, state):
        legal_actions = state.get_legal_actions()
        if len(legal_actions) == 1:
            return legal_actions[0]
        
        if state.is_wait_for_go():
            if self.fraction_eval(state):
                return Action('go')
            else:
                return Action('stop')

        a = random.choice(legal_actions)
        return a

    def reset(self):
        pass

class Fraction_Eval_Agent_v2:
    def __init__(self):
        # choose the track with highest relative progress, on average 3 tracks takes 7.5 turns to top
        # so every turn we should complete 3/7.5 = 0.4 track, in reality 0.8 is the strongest
        # if finish a track, stop
        pass

    def fraction_eval(self, state):
        # to go -> true
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
        return fraction_score < 0.8

    def act(self, state):
        legal_actions = state.get_legal_actions()
        if len(legal_actions) == 1:
            return legal_actions[0]
        
        if state.is_wait_for_go():
            if self.fraction_eval(state):
                return Action('go')
            else:
                return Action('stop')

        dice = state.get_dice()
        track_len = state.get_track_len()
        track_state = state.get_track_state()
        cur_agent_idx = state.get_cur_agent_idx()

        def track_score(track):
            l = track_len[track]
            cur = track_state[track][cur_agent_idx]
            return cur / l

        action_score = []
        for a in legal_actions:
            dice_idx = a.get_dice()
            if len(dice_idx) == 2:
                action_score.append(track_score(dice[dice_idx[0]] + dice[dice_idx[1]]))
            else:
                t1, t2 = dice[dice_idx[0]] + dice[dice_idx[1]], dice[dice_idx[2]] + dice[dice_idx[3]]
                action_score.append(max(track_score(t1), track_score(t2)))
        a = legal_actions[action_score.index(max(action_score))]
        return a

    def reset(self):
        pass