import numpy as np
from agents import RLAgent, Agent
from fraction_eval import Fraction_Eval_Agent,Fraction_Eval_Agent_v2
from game import RLGame, Game
from models import SmallModel
from tqdm import tqdm
from util import STATE_LEN, ACTION_LEN

# Baseline agent
baselines = {
    'random': Agent(),
    'fraction_eval': Fraction_Eval_Agent(),
    'fraction_eval_v2': Fraction_Eval_Agent_v2()
}

class Wrap:
    # wrap agent for RLGame to use in Game
    def __init__(self, agent):
        self.agent = agent
    def act(self, state):
        # Game agent.act(state)
        # RLGame agent.act(state, reward, done)
        return self.agent.act(state, 0, False)
    def reset(self):
        self.agent.reset()

def rl_vs_baseline(wrapped_rl_agent, baseline_name, num_episodes=1000, verbose=False):
    baseline = baselines[baseline_name]
    agents = [wrapped_rl_agent, baseline]
    game01 = Game(agents)
    game10 = Game(agents[::-1])

    score = [0, 0]
    win = [0, 0]

    for _ in range(num_episodes // 2):
        for i,g in enumerate([game01, game10]):
            episode = g.play(record=True, verbose=verbose)
            g.reset()
            final_state = episode[-1][0]
            final_score = final_state.get_score()
            if i == 1:
                final_score = final_score[::-1]
            if final_score[0] > final_score[1]:
                win[0] += 1
            else:
                win[1] += 1
            score[0] += final_score[0]
            score[1] += final_score[1]
    print(f'RL Agent vs {baseline_name} Agent: {win[0]} wins, {win[1]} losses')
    print(f'RL Agent vs {baseline_name} Agent: avg {score[0] / num_episodes} points, avg {score[1] / num_episodes} points')

def example_game(wrapped_rl_agent, baseline_name, save_path=None):
    baseline = baselines[baseline_name]
    agents = [wrapped_rl_agent, baseline]
    game = Game(agents)
    episode = game.play(record=True, verbose=False)
    # save the episode to a file
    if save_path:
        with open(save_path, 'w') as f:
            for state, action in episode:
                f.write(f'{state}\n{action}\n')

def test(rl_agent):
    wrapped_rl_agent = Wrap(rl_agent)
    for baseline_name in baselines:
        example_game(wrapped_rl_agent, baseline_name, save_path=f'example_{baseline_name}.txt')
    for baseline_name in baselines:
        rl_vs_baseline(wrapped_rl_agent, baseline_name, num_episodes=100, verbose=False)

if __name__ == "__main__":
    # Example usage
    rl_agent = RLAgent(SmallModel(STATE_LEN, ACTION_LEN))
    test(rl_agent)
    
# rl_agent = RLAgent(SmallModel(273, 12))
# rl_agent.model.load_path('model_weights/rl_agent.pth')
# rl_vs_baseline(rl_agent, 'random', num_episodes=10, verbose=False)
# test(rl_agent)

