from game import RLGame, Game
from agents import RLAgent, HumanAgent, Agent
from models import DummyModel, SmallModel
from transition_buffer import TransitionBuffer
from util import run_game, state_to_array, STATE_LEN, ACTION_LEN

if __name__ == "__main__":
    verbose = True
    buffer = TransitionBuffer()
    agents = [RLAgent(SmallModel(STATE_LEN, ACTION_LEN), verbose, buffer), 
              RLAgent(SmallModel(STATE_LEN, ACTION_LEN), verbose, buffer)]
    game = RLGame(agents)
    run_game(game, num_episodes=1, verbose=verbose)
    print(f'buffer size: {len(buffer)}')