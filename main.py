from game import RLGame, Game
from agents import RLAgent, HumanAgent, Agent, ExpectimaxAgent
from models import DummyModel, SmallModel
from transition_buffer import TransitionBuffer
from util import run_game, state_to_array, STATE_LEN, ACTION_LEN

if __name__ == "__main__":
    verbose = True
    # buffer = TransitionBuffer()
    # agents = [RLAgent(SmallModel(STATE_LEN, ACTION_LEN), verbose, buffer), 
              # RLAgent(SmallModel(STATE_LEN, ACTION_LEN), verbose, buffer)]
    # game = RLGame(agents)
    # run_game(game, num_episodes=1, verbose=verbose)
    # print(f'buffer size: {len(buffer)}')
    agents = [ExpectimaxAgent(SmallModel(STATE_LEN, ACTION_LEN), verbose),
              RLAgent(SmallModel(STATE_LEN, ACTION_LEN), verbose)]
    agents[0].model.load_path("model_weights/rl_agent_v13_253.pth")
    agents[1].model.load_path("model_weights/rl_agent_v13_253.pth")
    game = RLGame(agents)
    run_game(game, num_episodes=1, verbose=verbose)



