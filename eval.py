import math
import random
from multiprocessing import Pool
from game import Game

def wilson(wins, n, z=1.96):
    p = wins / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h

def _play(args):
    make_a, make_b, n_games, a_first, seed = args
    random.seed(seed)
    a, b = make_a(), make_b()
    wins = 0
    for _ in range(n_games):
        a.reset()
        b.reset()
        agents = [a, b] if a_first else [b, a]
        score = Game(agents).play()
        wins += score[0 if a_first else 1] >= 3
    return wins

def evaluate(make_a, make_b, n=2000, workers=8, seed=0):
    # returns win rate of A and 95% Wilson CI (lo, hi); A moves first in half the games
    chunks = 2 * workers
    jobs = [(make_a, make_b, n // chunks + (i < n % chunks), i % 2 == 0, seed * 1000003 + i) for i in range(chunks)]
    with Pool(workers) as pool:
        wins = sum(pool.map(_play, jobs))
    return wins / n, wilson(wins, n)

if __name__ == '__main__':
    from baselines import RandomAgent, V2, V2Tuned
    for name, a, b in [('v2 vs v2', V2, V2), ('v2tuned vs v2', V2Tuned, V2), ('random vs v2', RandomAgent, V2)]:
        p, (lo, hi) = evaluate(a, b)
        print(f'{name}: {p:.3f}  95% CI [{lo:.3f}, {hi:.3f}]')
