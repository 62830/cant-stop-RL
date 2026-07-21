import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from copy import deepcopy

STATE_LEN = 273  # Length of the state array
ACTION_LEN = 12 # Number of possible actions



def run_game(game, num_episodes=1000, verbose=False):
    # for _ in tqdm(range(num_episodes)):
    wins = [0, 0]
    for _ in tqdm(range(num_episodes)):
        game.play(verbose=verbose)
        winner = game.state.get_winner()
        if winner != -1:
            wins[winner] += 1
        game.reset()
    if verbose:
        print(f'Wins: {wins[0]} - {wins[1]}')
    return wins

def run_game_parallel(game, num_episodes=1000, verbose=False, num_workers=4):
    per_worker = num_episodes // num_workers
    games = [deepcopy(game) for _ in range(num_workers)]
    with Pool(num_workers) as pool:
        results = list(tqdm(pool.imap(lambda g: run_game(g, per_worker, verbose), games), total=num_workers))
    total_wins = [0, 0]
    for result in results:
        total_wins[0] += result[0]
        total_wins[1] += result[1]
    if verbose:
        print(f'Total Wins: {total_wins[0]} - {total_wins[1]}')

    # copy transitions from each game's agents' transition buffers to the main game's transition buffer
    for g in games:
        for agent in g.agents:
            if hasattr(agent, 'transition_buffer'):
                for transition in agent.transition_buffer.transitions:
                    game.agents[0].transition_buffer.add(transition)
    return total_wins

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