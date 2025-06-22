import random
from game import Action, State
import copy

class Agent:
    def __init__(self, verbose = False):
        self.verbose = verbose
    def act(self, state):
        legal_actions = state.get_legal_actions()
        a = random.choice(legal_actions)
        if self.verbose:
            print(state)
            print(f'agent action: {a}')
        return a
    def reset(self):
        pass

class HumanAgent:
    def __init__(self):
        pass
    def act(self, state):
        print(state)
        print('input your action: ')
        action = input()
        return Action(action)
    def reset(self):
        pass

def episode_to_transitions(episode):
    # episode is a list of (s, a, a_prob, r, done)
    # transitions need (s, a, a_prob, r, s', done)
    transitions = []
    for i in range(len(episode)):
        state, action, action_prob, reward, done = episode[i]
        next_state = episode[i + 1][0] if i + 1 < len(episode) else episode[i][0]
        transitions.append((state, action, action_prob, reward, next_state, done))
    return transitions
    
class RLAgent:
    def __init__(self, model, verbose = False, transition_buffer = None):
        self.model = model
        self.verbose = verbose
        self.episode = []
        self.transition_buffer = transition_buffer

    def act(self, state, reward, done):
        legal_actions = state.get_legal_actions()
        action, action_prob, value = self.model.predict(state, legal_actions)
        if self.verbose:
            print(state)
            print(f'agent action: {action}, action_prob: {action_prob}, value: {value}')

        if self.transition_buffer is not None:
            self.episode.append((copy.deepcopy(state), action, action_prob, reward, done))
        return action
    
    def reset(self):
        self.model.reset()
        if self.transition_buffer is not None:
            transitions = episode_to_transitions(self.episode)
            for transition in transitions:
                self.transition_buffer.add(transition)
        self.episode.clear()
        
            

