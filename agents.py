import random
from game import Action, State
import copy

class Agent:
    def __init__(self, verbose = False):
        self.verbose = verbose
    def act(self, state, reward=None, done=False):
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

def episode_to_transitions(episode, gamma=0.99, lam=0.95):
    # episode is a list of (s, a, a_prob, v, r, done)
    # transitions need (s, a, a_prob, v, s', done, adv)

    transitions = []
    advantage = 0.0
    for i in reversed(range(len(episode) - 1)):
        state, action, action_prob, value, _, _ = episode[i]
        next_state, _, _, next_value, reward, done = episode[i + 1]
        delta = reward + gamma * next_value * (1 - done) - value
        advantage = delta + gamma * lam * advantage * (1 - done)
        transitions.append((state, action, action_prob, value, next_state, done, advantage))
    return transitions
    
class RLAgent:
    def __init__(self, model, verbose = False, transition_buffer = None):
        self.model = model
        self.verbose = verbose
        self.episode = []
        self.transition_buffer = transition_buffer

    def act(self, state, reward, done, deterministic=False):
        legal_actions = state.get_legal_actions()
        action, action_prob, value = self.model.predict(state, legal_actions, deterministic=deterministic)
        if self.verbose:
            print(state)
            print(f'agent action: {action}, action_prob: {action_prob}, value: {value}')

        if self.transition_buffer is not None:
            self.episode.append((copy.deepcopy(state), action, action_prob, value, reward, done))
        return action

    def load_weights(self, weights):
        self.model.load_weights(weights)
    
    def reset(self):
        self.model.reset()
        if self.transition_buffer is not None:
            transitions = episode_to_transitions(self.episode)
            for transition in transitions:
                self.transition_buffer.add(transition)
        self.episode.clear()
        
class MixedAgent:
    def __init__(self, agents, verbose = False):
        self.agents = agents
        self.verbose = verbose
        self.agent_idx = 0

    def act(self, state, reward, done):
        agent = self.agents[self.agent_idx]
        action = agent.act(state, reward, done)
        if self.verbose:
            print(f'agent {self.agent_idx} action: {action}')
        return action

    def reset(self):
        self.agents[self.agent_idx].reset()
        self.agent_idx = (self.agent_idx + 1) % len(self.agents)

    def load_weights(self, weights):
        for agent in self.agents:
            if hasattr(agent, 'load_weights'):
                agent.load_weights(weights)


