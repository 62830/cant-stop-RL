import os, math, random
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from multiprocessing import Pool
import numpy as np
import fastrules as fr
from game import Game, State

Z_POWER = 1.96 + 0.84  # alpha 0.05 two-sided, 80% power

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
    # None when going on from a winning stop (no go value; checked before solving, which would not expand it)
    from solver_agent import tmp_of
    prog, p, tmp = fr.from_state(s)
    if fr.stop_wins(prog, p, tmp):
        return None
    sol = ref.solution(prog, p, tmp)
    legal = s.get_legal_actions()
    a = sol.bust_val if legal[0].is_null() else max(sol.value(tmp_of(*s.unsafe_pick(x))) for x in legal)
    return a, sol.go[sol.row(tmp)]

def _game(a, b, seed, a_first, ref=None):
    # (1 if A wins the game with dice seed `seed`, sum of A's luck L_t = A_t - E_t over chance events, number of events).
    # ref: None, or (R for A's turns, R for B's turns), SolverAgents solving with the reference value_fn
    random.seed(seed)
    agents, me = ([a, b], 0) if a_first else ([b, a], 1)
    for x in agents + list(ref or ()):
        x.reset()
    refs = ref if ref is None or a_first else ref[::-1]
    s, luck, n = State(), 0.0, 0
    while not s.is_terminal():
        act = agents[s.cur_agent_idx].act(s)
        if refs and not s.wait_for_go and not s.end_turn:
            r = roll_luck(refs[s.cur_agent_idx], s)
            if r is not None:
                luck += (r[0] - r[1]) * (1 if s.cur_agent_idx == me else -1)
                n += 1
        s, _ = s.update(act)
    return float(s.get_winner() == me), luck, n

def _pairs(args):
    # an agent built around the very ref object (pickled in the same job, so identity survives) doubles as R
    make_a, make_b, seeds, ref = args
    a, b = make_a(), make_b()
    if ref is not None:
        from solver_agent import SolverAgent
        r = SolverAgent(ref)
        ref = tuple(x if type(x) is SolverAgent and x.value_fn is ref and x.depth is None and not (x.v2_picks or x.v2_gostop)
                    else r for x in (a, b))
    return [(_game(a, b, g, True, ref), _game(a, b, g, False, ref)) for g in seeds]

def _sd(x):
    return float(math.sqrt(((x - x.mean()) ** 2).sum() / max(len(x) - 1, 1)))

def _ci(x):
    m, h = float(x.mean()), 1.96 * _sd(x) / math.sqrt(len(x))
    return m, (m - h, m + h)

def n_needed(sd, mde):
    # pairs for a minimum detectable edge mde (win-rate fraction) at Z_POWER
    return (Z_POWER * sd / mde) ** 2

def paired_stats(res, luck, events):
    # res, luck, events [n, 2] (A first, B first; A's view). CIs over pairs. luck_first: the first mover's mean luck
    raw, adj = res.mean(1), (res - luck).mean(1)
    return dict(n=len(res), raw=_ci(raw), adj=_ci(adj), sd_raw=_sd(raw), sd_adj=_sd(adj), luck=_ci(luck.mean(1)),
                luck_first=_ci((luck[:, 0] - luck[:, 1]) / 2), luck_sd_game=_sd(luck.ravel()), events=float(events.mean()))

def _run_pairs(make_a, make_b, n, workers, seed, ref):
    seeds = [seed * 1000003 + g for g in range(n)]
    with Pool(workers) as pool:
        parts = pool.map(_pairs, [(make_a, make_b, seeds[i::2 * workers], ref) for i in range(2 * workers)])
    out = np.empty((n, 2, 3))
    for i, part in enumerate(parts):
        if part:
            out[i::2 * workers] = part
    return out[..., 0], out[..., 1], out[..., 2]

def evaluate_paired(make_a, make_b, n=1000, workers=8, seed=0, ref=None):
    # n pairs: pair g plays dice seed g with A first, then the same seed with B first (identical decisions consume identical
    # dice). Returns A's mean pair score (0 / 0.5 / 1 per pair) and a 95% normal CI over pairs.
    # ref (a value_fn): the luck-adjusted score instead (see evaluate_paired_detail), same estimand, less variance
    d = evaluate_paired_detail(make_a, make_b, n, workers, seed, ref)
    return d['adj' if ref is not None else 'raw']

def evaluate_paired_detail(make_a, make_b, n=1000, workers=8, seed=0, ref=None):
    # paired_stats plus per-game arrays result, luck_sum, n_events [n, 2]. With ref, luck adjustment by the reference solver
    # R = solve_turn(., ., ref): each game also scores Y = result - sum_t (A_t - E_t) over chance events (every roll), A's
    # view; E[A_t - E_t] = 0 at each roll, so Y is an unbiased control variate. Without ref luck is 0 (adj = raw)
    res, luck, events = _run_pairs(make_a, make_b, n, workers, seed, ref)
    return dict(paired_stats(res, luck, events), result=res, luck_sum=luck, n_events=events)

if __name__ == '__main__':
    from baselines import RandomAgent, V2, V2Tuned
    for name, a, b in [('v2 vs v2', V2, V2), ('v2tuned vs v2', V2Tuned, V2), ('random vs v2', RandomAgent, V2)]:
        p, (lo, hi) = evaluate(a, b)
        print(f'{name}: {p:.3f}  95% CI [{lo:.3f}, {hi:.3f}]')
