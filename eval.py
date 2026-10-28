import os, math, random
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
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

def _game(a, b, seed, a_first):
    # 1 if A wins the game with dice seed `seed`
    random.seed(seed)
    a.reset()
    b.reset()
    score = Game([a, b] if a_first else [b, a]).play()
    return float(score[0 if a_first else 1] >= 3)

def _pairs(args):
    make_a, make_b, seeds = args
    a, b = make_a(), make_b()
    return [(_game(a, b, g, True) + _game(a, b, g, False)) / 2 for g in seeds]

def evaluate_paired(make_a, make_b, n=1000, workers=8, seed=0):
    # n pairs: pair g plays dice seed g with A first, then the same seed with B first (identical decisions consume identical
    # dice). Returns A's mean pair score (0 / 0.5 / 1 per pair) and a 95% normal CI over pairs
    seeds = [seed * 1000003 + g for g in range(n)]
    with Pool(workers) as pool:
        s = [x for part in pool.map(_pairs, [(make_a, make_b, seeds[i::2 * workers]) for i in range(2 * workers)]) for x in part]
    m = sum(s) / n
    h = 1.96 * math.sqrt(sum((x - m) ** 2 for x in s) / max(n - 1, 1) / n)
    return m, (m - h, m + h)

if __name__ == '__main__':
    from baselines import RandomAgent, V2, V2Tuned
    for name, a, b in [('v2 vs v2', V2, V2), ('v2tuned vs v2', V2Tuned, V2), ('random vs v2', RandomAgent, V2)]:
        p, (lo, hi) = evaluate(a, b)
        print(f'{name}: {p:.3f}  95% CI [{lo:.3f}, {hi:.3f}]')
