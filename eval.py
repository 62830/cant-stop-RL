import os, math, random, pickle
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from multiprocessing import Pool
import numpy as np
import fastrules as fr
from game import Game, State
from solver_agent import SolverAgent, tmp_of

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

def roll_luck(ref, s):
    # chance event at s (dice just rolled, a pick or bust pending): (A_t, E_t) for the mover under the reference solver
    # ref (a SolverAgent): E_t = go value at the current tmp, A_t = best pick value after the roll (bust value on a bust).
    # None where go is undefined (going on from a winning stop)
    prog, p, tmp = fr.from_state(s)
    sol = ref.solution(prog, p, tmp)
    e = sol.go[sol.row(tmp)]
    if np.isnan(e):
        return None
    legal = s.get_legal_actions()
    a = sol.bust_val if legal[0].is_null() else max(sol.value(tmp_of(*s.unsafe_pick(x))) for x in legal)
    return a, e

def _game(a, b, seed, a_first, ref=None):
    # (1 if A wins the game with dice seed `seed`, sum of A's luck L_t = A_t - E_t over chance events, number of events).
    # ref: (R for A's turns, R for B's turns), each a SolverAgent solving with the reference value_fn (the mover itself
    # when it is one, reusing its solves)
    random.seed(seed)
    a.reset()
    b.reset()
    if ref is None:
        score = Game([a, b] if a_first else [b, a]).play()
        return float(score[0 if a_first else 1] >= 3), 0.0, 0
    for r in ref:
        r.reset()
    agents, refs, me = ([a, b], ref, 0) if a_first else ([b, a], ref[::-1], 1)
    s, luck, n = State(), 0.0, 0
    while not s.is_terminal():
        act = agents[s.cur_agent_idx].act(s)
        if not s.wait_for_go and not s.end_turn:
            r = roll_luck(refs[s.cur_agent_idx], s)
            if r is not None:
                luck += (r[0] - r[1]) * (1 if s.cur_agent_idx == me else -1)
                n += 1
        s, _ = s.update(act)
    return float(s.get_winner() == me), luck, n

def _is_ref(agent, ref):
    return (type(agent) is SolverAgent and agent.depth is None and not (agent.v2_picks or agent.v2_gostop)
            and pickle.dumps(agent.value_fn) == pickle.dumps(ref))

def _pairs(args):
    make_a, make_b, seeds, ref = args
    a, b = make_a(), make_b()
    if ref is not None:
        r = SolverAgent(ref)
        ref = tuple(x if _is_ref(x, ref) else r for x in (a, b))
    return [(_game(a, b, g, True, ref), _game(a, b, g, False, ref)) for g in seeds]

def _ci(x):
    m, h = x.mean(), 1.96 * x.std(ddof=1) / math.sqrt(len(x))
    return float(m), (float(m - h), float(m + h))

def evaluate_paired(make_a, make_b, n=1000, workers=8, seed=0, ref=None, detail=False):
    # n pairs: pair g plays dice seed g with A first, then the same seed with B first (identical decisions consume identical
    # dice). Returns A's mean pair score (0 / 0.5 / 1 per pair) and a 95% normal CI over pairs.
    # ref (a value_fn): also luck-adjust with the reference solver R = solve_turn(., ., ref): each game scores
    # Y = result - sum_t (A_t - E_t) over chance events (rolls), A's view; an unbiased control variate. Returns a dict
    # (raw and adjusted mean, CI, SD; luck diagnostics; with detail, per-game arrays [n, 2] for A first / B first)
    seeds = [seed * 1000003 + g for g in range(n)]
    jobs = [(make_a, make_b, seeds[i::2 * workers], ref) for i in range(2 * workers)]
    with Pool(workers) as pool:
        parts = pool.map(_pairs, jobs)
    out = np.empty((n, 2, 3))
    for i, part in enumerate(parts):
        out[i::2 * workers] = part
    res, luck, events = out[..., 0], out[..., 1], out[..., 2]
    raw = res.mean(1)
    if ref is None:
        return _ci(raw)
    adj = (res - luck).mean(1)
    d = dict(n=n, raw=_ci(raw), adj=_ci(adj), sd_raw=float(raw.std(ddof=1)), sd_adj=float(adj.std(ddof=1)),
             luck=_ci(luck.ravel()), luck_sd=float(luck.std(ddof=1)), events=float(events.mean()))
    if detail:
        d.update(result=res, luck_sum=luck, n_events=events)
    return d

if __name__ == '__main__':
    from baselines import RandomAgent, V2, V2Tuned
    for name, a, b in [('v2 vs v2', V2, V2), ('v2tuned vs v2', V2Tuned, V2), ('random vs v2', RandomAgent, V2)]:
        p, (lo, hi) = evaluate(a, b)
        print(f'{name}: {p:.3f}  95% CI [{lo:.3f}, {hi:.3f}]')
