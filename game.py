import random
import copy

class State:
    def __init__(self, num_agents = 2):
        self.num_agents = num_agents
        self.cur_agent_idx = 0
        self.mx_tmp_track_len = 3
        self.score = [0 for _ in range(self.num_agents)]
        self.track_len = [0, 0, 3, 5, 7, 9, 11, 13, 11, 9, 7, 5, 3]
        self.track_state = [ [0 for _ in range(self.num_agents)] for _ in range(len(self.track_len)) ]

        self.tmp_track = []
        self.tmp_progress = []
        self.wait_for_go = False
        # set after 'stop'; the turn is then ended by a forced 'null', like a bust
        self.end_turn = False
        self.dice = [1, 1, 1, 1]
        self.roll_dice()

    def get_cur_agent_idx(self):
        return self.cur_agent_idx
    def get_num_agents(self):
        return self.num_agents
    def get_score(self):
        return self.score
    def get_track_len(self):
        return self.track_len
    def get_track_state(self):
        return self.track_state
    def get_tmp_track(self):
        return self.tmp_track
    def get_tmp_progress(self):
        return self.tmp_progress
    def get_mx_tmp_track_len(self):
        return self.mx_tmp_track_len
    def get_dice(self):
        return self.dice

    def is_terminal(self):
        return any([score >= 3 for score in self.score])

    def is_wait_for_go(self):
        return self.wait_for_go

    def set_dice(self, dice):
        # can't bother to check
        # set after go
        self.dice = dice

    def roll_dice(self):
        # roll 4 6-sided dice
        self.dice = [random.randint(1, 6) for _ in range(4)]

    def is_track_topped(self, track_idx):
        return max(self.track_state[track_idx]) == self.track_len[track_idx]

    def unsafe_pick(self, action):
        # make a pick action and return the tmp_track and tmp_progress after the action
        # without checking if the action is valid
        dice_idx = action.get_dice()
        tmp_track = self.tmp_track.copy()
        tmp_progress = self.tmp_progress.copy()
        for i in range(0,len(dice_idx),2):
            track_idx = self.dice[dice_idx[i]] + self.dice[dice_idx[i+1]]
            if track_idx in tmp_track:
                tmp_progress[tmp_track.index(track_idx)] += 1
            else:
                tmp_track.append(track_idx)
                tmp_progress.append(self.track_state[track_idx][self.cur_agent_idx] + 1)
        return tmp_track, tmp_progress

    def is_valid_tmp(self, tmp_track, tmp_progress):
        if len(tmp_track) > self.mx_tmp_track_len:
            return False
        for track_idx, progress in zip(tmp_track, tmp_progress):
            if progress > self.track_len[track_idx]:
                return False
            if progress <= self.track_state[track_idx][self.cur_agent_idx]:
                return False
            if self.is_track_topped(track_idx):
                return False
        return True

    def is_valid_pick(self, action):
        return self.is_valid_tmp(*self.unsafe_pick(action))

    def update(self, action):
        reward = [0 for _ in range(self.num_agents)]
        R_WIN = 1
        R_LOSE = -1
        R_NULL = 0
        R_COLUMN = 0  # subsumed by the potential-based shaping in episode_to_transitions

        if self.get_legal_actions()[0].is_null():
            # null ends a turn (bust or after stop): switch agent and roll fresh dice
            assert action.is_null()
            self.wait_for_go = False
            self.end_turn = False
            self.cur_agent_idx = (self.cur_agent_idx + 1) % self.num_agents
            self.tmp_track = []
            self.tmp_progress = []
            self.roll_dice()
            reward[self.cur_agent_idx] += R_NULL
            return self, reward

        if self.wait_for_go:
            # 'go' or 'stop' action is expected
            assert action.is_go() or action.is_stop()
            if action.is_go():
                self.roll_dice()
            else:
                for track_idx, progress in zip(self.tmp_track, self.tmp_progress):
                    self.track_state[track_idx][self.cur_agent_idx] = progress
                    if progress == self.track_len[track_idx]:
                        self.score[self.cur_agent_idx] += 1
                        reward[self.cur_agent_idx] += R_COLUMN
                        if self.score[self.cur_agent_idx] >= 3:
                            reward[self.cur_agent_idx] += R_WIN
                            for j in range(self.num_agents):
                                if j != self.cur_agent_idx:
                                    reward[j] += R_LOSE
                        self.track_state[track_idx] = [0 for _ in range(self.num_agents)]
                        self.track_state[track_idx][self.cur_agent_idx] = progress
                # end the turn via a following 'null' (unless this stop won -> terminal)
                self.end_turn = True
            self.wait_for_go = False
            return self, reward
        
        # expect to pick dice
        assert action.is_pick()
        tmp_track, tmp_progress = self.unsafe_pick(action)
        assert self.is_valid_tmp(tmp_track, tmp_progress)
        self.tmp_track = tmp_track
        self.tmp_progress = tmp_progress
        self.wait_for_go = True
        return self, reward
    
    def track_str(self, track_idx, agents_char = 'ox$#'):
        # suppose track len 5, agent 0 at 2, agent 1 at 3, agent 2 at 0
        # 'ox_ ox_ _x_ ___ ___' o is 0, x is 1, _ is space
        s = ''
        in_tmp = track_idx in self.tmp_track
        tmp_progress = self.tmp_progress[self.tmp_track.index(track_idx)] if in_tmp else -1
        for i in range(1, self.track_len[track_idx]+1):
            for j in range(self.num_agents):
                if self.track_state[track_idx][j] == i:
                    s += agents_char[j]
                elif in_tmp and tmp_progress == i and j == self.cur_agent_idx:
                    s += '*'
                else:
                    s += '_'
            s += '|'
        return s

    def get_legal_actions(self):
        if self.is_terminal():
            return [Action('null')]
        if self.end_turn:
            return [Action('null')]
        if self.wait_for_go:
            return [Action('go'), Action('stop')]
        picks = [Action('01'), Action('02'), Action('03'), Action('12'), Action('13'), Action('23'),
                    Action('0123'), Action('0213'), Action('0312')]
        legal_picks = [pick for pick in picks if self.is_valid_pick(pick)]
        if Action('0123') in legal_picks:
            # remove 01, 23 if they're in legal_picks
            legal_picks = [pick for pick in legal_picks if pick.value not in [3, 8]]
        if Action('0213') in legal_picks:
            legal_picks = [pick for pick in legal_picks if pick.value not in [4, 7]]
        if Action('0312') in legal_picks:
            legal_picks = [pick for pick in legal_picks if pick.value not in [5, 6]]
        return legal_picks if len(legal_picks) > 0 else [Action('null')]

    def get_reward(self):
        # return [ 1 for score in self.score if score >= 3 else -1]
        return [int(score >= 3)*2-1 for score in self.score]

    def get_winner(self):
        for i, score in enumerate(self.score):
            if score >= 3:
                return i
        return -1 
    
    def __str__(self):
        # print every track, dice, and score
        s = ''
        for i in range(len(self.track_len)):
            if self.track_len[i] == 0:
                continue
            s += f'{i:02d}: {self.track_str(i)}\n'

        s += f'dice: {self.dice}\n'
        s += f'score: {self.score}\n'
        s += f'cur_agent: {self.cur_agent_idx}\n'
        s += f'wait_for_go: {self.wait_for_go}\n'
        s += f'end_turn: {self.end_turn}\n'
        s += f'tmp_track: {self.tmp_track}\n'
        s += f'tmp_progress: {self.tmp_progress}\n'
        s += f'legal_actions: {self.get_legal_actions()}\n'

        return s
    
    __repr__ = __str__

class Action:
    act2str = ['null', 'go', 'stop', '01', '02', '03', '12', '13', '23', '0123', '0213', '0312']
    str2act = {v: k for k, v in enumerate(act2str)}
    act2dice = {3: [0, 1], 4: [0, 2], 5: [0, 3], 6: [1, 2], 7: [1, 3], 8: [2, 3], 
                9: [0, 1, 2, 3], 10: [0, 2, 1, 3], 11: [0, 3, 1, 2]}
    def __init__(self, action = 0):
        '''
        Pass in either none, int, or str
        Define the action space
        1: go
        2: stop
        3-8: pick dice 01 02 03 12 13 23
        9-11: pick dice (01 23) (02 13) (03 12)
        '''
        self.value = action if type(action) == int else self.str2act[action]
        assert 0 <= self.value <= 11

    def get_dice(self):
        return self.act2dice[self.value]
    
    def is_null(self):
        return self.value == 0
    
    def is_go(self):
        return self.value == 1
    
    def is_stop(self):
        return self.value == 2
    
    def is_pick(self):
        return 3 <= self.value <= 11
    
    def is_pick_2(self):
        return 3 <= self.value <= 8
    
    def is_pick_4(self):
        return 9 <= self.value <= 11

    def __str__(self):
        return self.act2str[self.value]

    __repr__ = __str__
    
    def __eq__(self, other):
        return self.value == other.value

class Game:
    def __init__(self, agents, state = None):
        self.state = state if state else State(num_agents = len(agents))
        self.agents = agents

    def play(self, record = False, verbose = False):
        episode = []
        while not self.state.is_terminal() :
            cur_agent = self.agents[self.state.get_cur_agent_idx()]
            action = cur_agent.act(self.state)
            if verbose:
                print(self.state)
                print(f'agent {self.state.get_cur_agent_idx()} action: {action}')
            if record:
                episode.append((copy.deepcopy(self.state), action))
            self.state, _ = self.state.update(action)

        if verbose:
            print(self.state)
        if record:
            episode.append((self.state, None))
            return episode
        else:
            return self.state.score

    def reset(self):
        self.state = State(num_agents = len(self.agents))
        for agent in self.agents:
            agent.reset()

class RLGame:
    def __init__(self, agents):
        self.agents = agents
        self.state = State(num_agents = len(agents))
        self.reward = [0 for _ in range(len(agents))]
        self.steps = 0

    def play(self, verbose = False):
        while not self.state.is_terminal() and self.steps < 500:
            agent_idx = self.state.get_cur_agent_idx()
            agent = self.agents[agent_idx]
            action = agent.act(self.state, self.reward[agent_idx], False)
            self.reward[agent_idx] = 0

            if verbose:
                print(self.state)
                print(f'agent {self.state.get_cur_agent_idx()} action: {action}')

            self.state, reward = self.state.update(action)
            for i in range(len(self.agents)):
                self.reward[i] += reward[i]
            # self.reward[agent_idx] += -0.0001 # small penalty for long game
            self.steps += 1

        # send the final state to all agents 
        for agent_idx in range(len(self.agents)):
            self.agents[agent_idx].act(self.state, self.reward[agent_idx], True)

    def reset(self):
        self.state = State(num_agents = len(self.agents))
        self.reward = [0 for _ in range(len(self.agents))]
        for agent in self.agents:
            agent.reset()
        self.steps = 0
