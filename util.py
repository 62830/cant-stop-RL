import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from copy import deepcopy

STATE_LEN = 273  # Length of the state array
ACTION_LEN = 12 # Number of possible actions

def run_game(game, num_episodes=1000, verbose=False, progress=True):
    wins = [0, 0]
    for _ in (tqdm(range(num_episodes)) if progress else range(num_episodes)):
        game.play(verbose=verbose)
        winner = game.state.get_winner()
        if winner != -1:
            wins[winner] += 1
        game.reset()
    if verbose:
        print(f'Wins: {wins[0]} - {wins[1]}')
    return wins

def _rollout_worker(task):
    # Runs in a separate process. Each worker plays n_episodes of self-play with
    # its own model copies and returns (wins, transitions). Only the learner
    # records transitions, so the buffer stays on-policy (see fix #1).
    import torch
    torch.set_num_threads(1)  # batch-1 inference is faster single-threaded; avoids oversubscription across workers
    from agents import RLAgent
    from game import RLGame
    from transition_buffer import TransitionBuffer

    model_cls, weights_learner, weights_opponent, learner_first, n_episodes = task
    buffer = TransitionBuffer()
    learner = RLAgent(model_cls(STATE_LEN, ACTION_LEN), transition_buffer=buffer)
    opponent = RLAgent(model_cls(STATE_LEN, ACTION_LEN))
    learner.model.load_weights(weights_learner)
    opponent.model.load_weights(weights_opponent)

    players = [learner, opponent]
    if not learner_first:
        players = players[::-1]
    game = RLGame(players)
    wins = run_game(game, num_episodes=n_episodes, progress=False)
    if not learner_first:
        wins = wins[::-1]
    return wins, list(buffer)

def run_game_parallel(model_cls, weights_learner, weights_opponent, swap_first=False, num_episodes=1000, num_workers=8):
    # Fan the rollout out over num_workers processes and merge the results.
    # Returns (total_wins, transitions) where transitions come from the learner only.

    def build_tasks(episodes, learner_first, workers):
        per = episodes // workers
        rem = episodes % workers
        tasks = [(model_cls, weights_learner, weights_opponent, learner_first, per + (1 if i < rem else 0))
                for i in range(workers)]
        tasks = [t for t in tasks if t[4] > 0]
        return tasks

    if swap_first:
        assert num_workers >= 2
        tasks = build_tasks((num_episodes+1)//2, True, (num_workers+1)//2)
        tasks.extend(build_tasks(num_episodes//2, False, num_workers//2))
    else:
        tasks = build_tasks(num_episodes, True, num_workers)

    with Pool(len(tasks)) as pool:
        results = pool.map(_rollout_worker, tasks)
    total_wins = [0, 0]
    transitions = []
    for wins, trans in results:
        total_wins[0] += wins[0]
        total_wins[1] += wins[1]
        transitions.extend(trans)
    return total_wins, transitions

def state_to_array(state):
    def threshold_encode(cut, length):
        arr = np.zeros(length, dtype=int)
        arr[:cut] = 1
        return arr
    def track_state_to_array(track_state, track_len, agent_idx):
        arr = []
        for i in range(len(track_state)):
            arr.append(threshold_encode(track_state[i][agent_idx], track_len[i]))
        return np.concatenate(arr)
    def tmp_track_to_array(tmp_track, tmp_progress, track_len):
        arr = []
        for i in range(len(track_len)):
            if i in tmp_track:
                idx = tmp_track.index(i)
                progress = tmp_progress[idx]
            else:
                progress = 0
            arr.append(threshold_encode(progress, track_len[i]))
        return np.concatenate(arr)
    def dice_to_array(dice):
        arr = np.zeros(24, dtype=int)
        for i, d in enumerate(dice):
            arr[i * 6 + d - 1] = 1
        return arr

    my_idx = state.get_cur_agent_idx()
    track_state = state.get_track_state()
    track_len = state.get_track_len()
    tmp_track = state.get_tmp_track()
    tmp_progress = state.get_tmp_progress()

    my_track_state_array = track_state_to_array(track_state, track_len, my_idx)
    oppo_track_state_array = track_state_to_array(track_state, track_len, 1 - my_idx)
    tmp_track_array = tmp_track_to_array(tmp_track, tmp_progress, track_len)
    if state.is_wait_for_go():
        dice_array = np.zeros(24, dtype=int)
    else:
        dice = state.get_dice()
        dice_array = dice_to_array(dice)

    return np.concatenate([my_track_state_array, oppo_track_state_array, tmp_track_array, dice_array])