import random
from game import Action, State
import copy
import itertools

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


class ExpectimaxAgent:
    def __init__(self, model, verbose = False, transition_buffer = None):
        self.model = model
        self.verbose = verbose
        self.episode = []
        self.transition_buffer = transition_buffer

    def go_expectimax(self, state):
        # stop: -V(opponent POV state)
        # go: sum dice V(state after dice)

        def get_value(state):
            return self.model.predict(state, state.get_legal_actions(), deterministic=True)[2]

        stop_state, _ = copy.deepcopy(state).update(Action('stop'))
        stop_value = -get_value(stop_state)

        go_state, _ = copy.deepcopy(state).update(Action('go'))
        bust_value = None
        sum_value = 0
        for dice in itertools.product(range(1,7), repeat=4):
            go_state.set_dice(list(dice))
            if go_state.get_legal_actions()[0].is_null():
                if bust_value is None:
                    bust_value = get_value(go_state)
                sum_value += bust_value
            else:
                sum_value += get_value(go_state)
        go_value = sum_value / 1296

        if self.verbose:
            breakpoint()

        if go_value > stop_value:
            return Action('go')
        else:
            return Action('stop')


    def act(self, state, reward, done, deterministic=False):
        if state.is_wait_for_go():
            legal_actions = state.get_legal_actions()
            action, action_prob, value = self.model.predict(state, legal_actions, deterministic=deterministic)
            if self.verbose:
                print("what would vanilla act do?")
                print(action, action_prob, value)
            return self.go_expectimax(state)

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
    # An opponent that samples one member per episode by weight. Members keep their
    # own (frozen) weights -- load_weights is intentionally a no-op so the league
    # can't be overwritten by an opponent-sync call.
    def __init__(self, agents, weights=None, verbose=False):
        self.agents = agents
        self.weights = weights if weights is not None else [1.0] * len(agents)
        assert len(self.weights) == len(self.agents)
        self.verbose = verbose
        self.cur = 0
        self._sample()

    def _sample(self):
        self.cur = random.choices(range(len(self.agents)), weights=self.weights, k=1)[0]

    def act(self, state, reward, done, deterministic=False):
        agent = self.agents[self.cur]
        try:
            action = agent.act(state, reward, done, deterministic=deterministic)
        except TypeError:
            action = agent.act(state, reward, done)  # heuristics without deterministic kwarg
        if self.verbose:
            print(f'member {self.cur} action: {action}')
        return action

    def reset(self):
        self.agents[self.cur].reset()
        self._sample()

    def load_weights(self, weights):
        pass  # members are frozen; do not broadcast


def build_opponent_pool(model_factory, self_weights, heuristics=None, heuristic_frac=0.0, verbose=False):
    # Nash-preserving opponent mixture for self-play.
    #
    # self_weights: list of frozen learner checkpoints (recent + past). The pool is
    # dominated by these (fictitious self-play), which is what lets self-play move
    # toward the game's Nash equilibrium; uniform-over-past has the cleanest
    # average-strategy convergence, recency-skew trains faster but can cycle.
    #
    # heuristics (list of fixed agents e.g. random / v2) get only heuristic_frac of
    # the mass, annealed toward 0. A Nash policy already beats them, so as their
    # weight -> 0 the best-response fixed point coincides with Nash; a little mass
    # just aids coverage and stops the learner forgetting how to punish weak play.
    # heuristic_frac=0 => strict fictitious self-play.
    heuristics = heuristics or []
    assert self_weights, 'need at least one self checkpoint'
    if not heuristics:
        heuristic_frac = 0.0

    members, weights = [], []
    self_mass = (1.0 - heuristic_frac) / len(self_weights)
    for w in self_weights:
        a = RLAgent(model_factory())
        a.load_weights(w)
        members.append(a)
        weights.append(self_mass)
    if heuristics:
        h_mass = heuristic_frac / len(heuristics)
        for h in heuristics:
            members.append(h)
            weights.append(h_mass)
    return MixedAgent(members, weights=weights, verbose=verbose)


