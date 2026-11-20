import os, sys, json, time, argparse
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from collections import Counter
from multiprocessing import Pool
import numpy as np
from numba import njit
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import fastrules as fr
from fastrules import LEN
from turn_solver import solve_turn

# How the solver agent plays: exact first-turn policy (opening) and self-play decision logs (claims, starts, bail-out).
# python analysis/strategy_profile.py selfplay --model ens|S|vh --games N ; opening ; games
OUT = os.path.join(ROOT, 'runs', 'strategy')
CACHE = os.path.join(OUT, 'cache')
ARCH = os.environ.get('CANT_STOP_ARCHIVE', '/home/identity_element/Documents/code_playground/python/cant_stop_archive').rstrip('/') + '/'  # checkpoint root
MODELS = dict(ens=[ARCH + f'cant_stop_ens/runs/{c}' for c in ('ens_s2/ckpt/it0337.pt', 'ens_s4/ckpt/it0283.pt', 'ens_s5/ckpt/it0284.pt')],
              S=[ARCH + 'cant_stop_scratch/runs/scratch30/ckpt/it0297.pt'], vh='vh')
COLS = np.arange(2, 13)
ONE = lambda progs, m: np.ones(len(progs))  # opponent always wins after a bank: solving maximizes P(win this turn)
NOPP = np.zeros(13, np.int8)

def load(model):
    import torch
    torch.set_num_threads(1)
    from value_net import NetV, Ensemble
    from td_train import BEST_VH
    src = MODELS[model]
    return BEST_VH if src == 'vh' else NetV.load(src[0]).fused() if len(src) == 1 else Ensemble(src)

# ---- exact turn analysis
@njit(cache=True)
def _row_moves(ptr, cmv, crow, i, best, urow, srow):
    # urow[u]: the row the policy reaches from row i when the roll offers pair UPAIRS[u] (-1: none legal)
    srow[:] = -1
    for e in range(ptr[i], ptr[i + 1]):
        if cmv[e] < 11:
            srow[cmv[e] + 2] = crow[e]
    for u in range(len(fr.UPAIRS)):
        a, b = srow[fr.UPAIRS[u, 0]], srow[fr.UPAIRS[u, 1]]
        urow[u] = a if b < 0 or (a >= 0 and best[a] >= best[b]) else b
    for e in range(ptr[i], ptr[i + 1]):
        if cmv[e] >= 11:
            urow[cmv[e] - 11] = crow[e]

@njit(cache=True)
def _roll_row(urow, r, best):
    j = -1
    for k in range(fr.ROLL_NPAIRS[r]):
        x = urow[fr.ROLL_UPAIR[r, k]]
        if x >= 0 and (j < 0 or best[x] > best[j]):
            j = x
    return j

@njit(cache=True)
def _reach(child, best, act_stop, order):
    # forward pass of the policy (picks maximize best, act_stop[i]: bank at row i); returns reach, stop mass, bust mass
    # per row and the row chosen by each first roll (-1: bust)
    ptr, cmv, crow = child
    n = len(best)
    reach, stopm, bustm = np.zeros(n), np.zeros(n), np.zeros(n)
    first = np.full(fr.N_ROLLS, -1, np.int64)
    urow, srow = np.empty(len(fr.UPAIRS), np.int64), np.empty(13, np.int64)
    reach[0] = 1.0
    for t in range(n - 1, -1, -1):
        i = order[t]
        if reach[i] == 0:
            continue
        if act_stop[i]:
            stopm[i] = reach[i]
            continue
        _row_moves(ptr, cmv, crow, i, best, urow, srow)
        for r in range(fr.N_ROLLS):
            j = _roll_row(urow, r, best)
            if i == 0:
                first[r] = j
            if j < 0:
                bustm[i] += reach[i] * fr.ROLL_PROB[r]
            else:
                reach[j] += reach[i] * fr.ROLL_PROB[r]
    return reach, stopm, bustm, first

@njit(cache=True)
def _forced(child, stop, bust_val, order, force):
    # exact value of a stop rule: force[i] 1 bank, 2 go (picks re-optimized for the rule); winning rows always bank
    ptr, cmv, crow = child
    val = stop.copy()
    urow, srow = np.empty(len(fr.UPAIRS), np.int64), np.empty(13, np.int64)
    for i in order:
        if stop[i] >= 1.0 or force[i] == 1:
            continue
        _row_moves(ptr, cmv, crow, i, val, urow, srow)
        g = 0.0
        for r in range(fr.N_ROLLS):
            j = _roll_row(urow, r, val)
            g += fr.ROLL_PROB[r] * (bust_val if j < 0 else val[j])
        val[i] = g
    return val

def tmps(sol):
    return np.array([fr.decode(k) for k in sol.keys])

def turn_policy(sol):
    # reach analysis of a full (unbounded) turn solve; decision rows: reached, not the root, not a winning bank
    act_stop = ~(sol.go > sol.stop)  # nan go (winning leaf) -> bank
    act_stop[0] = False
    reach, stopm, bustm, first = _reach(sol.child, sol.best, act_stop, sol.order)
    T = tmps(sol)
    adv = np.where(T > 0, T - sol.prog[sol.p], 0)
    return dict(reach=reach, stopm=stopm, bustm=bustm, first=first, T=T, adv=adv, act_stop=act_stop,
                win=sol.stop >= 1.0, dec=(reach > 0) & (sol.stop < 1.0) & (np.arange(len(reach)) > 0))

def feats(sol, tp):
    # rel_steps: sum over markers of steps / column length; topped: markers at the top; pbust: P(next roll busts)
    T, adv = tp['T'], tp['adv']
    on = T > 0
    return dict(markers=on.sum(1), steps=adv.sum(1), rel_steps=(adv / np.maximum(LEN, 1)).sum(1), topped=(on & (T == LEN)).sum(1),
                pbust=np.nan_to_num(sol.pbust), outer=on[:, [2, 3, 11, 12]].sum(1), on7=on[:, 7].astype(int))

def rule_value(sol, force):
    force = force.astype(np.int8)
    force[0] = 2  # the empty turn start can't bank
    return float(_forced(sol.child, sol.stop, sol.bust_val, sol.order, force)[0])

def summarize_turn(sol, tp):
    f = feats(sol, tp)
    stopm, bustm, reach, dec = tp['stopm'], tp['bustm'], tp['reach'], tp['dec']
    T = tp['T']
    on = T > 0
    win = tp['win']
    out = dict(v=float(sol.best[0]), p_bust=float(bustm.sum()), p_bank=float(stopm[~win].sum()), p_win=float(stopm[win].sum()),
               exp_steps_banked=float((stopm * f['steps']).sum()), exp_rel_banked=float((stopm * f['rel_steps']).sum()),
               mean_steps_at_bank=float((stopm * f['steps']).sum() / stopm.sum()),
               markers_at_bank={int(m): float(stopm[f['markers'] == m].sum()) for m in (1, 2, 3)},
               col_banked={int(c): float(stopm[on[:, c]].sum()) for c in COLS},
               col_mean_height_if_banked={int(c): float((stopm * T[:, c])[on[:, c]].sum() / max(stopm[on[:, c]].sum(), 1e-12)) for c in COLS})
    sets = Counter()
    for i in np.flatnonzero(stopm):
        sets[tuple(int(c) for c in np.flatnonzero(on[i]))] += stopm[i]
    out['top_banked_sets'] = [[list(k), float(v)] for k, v in sets.most_common(12)]
    # P(bank) at decisions, by markers x steps (reach-weighted)
    tab = {}
    for m in (1, 2, 3):
        for s in range(1, int(f['steps'][dec].max()) + 1):
            k = dec & (f['markers'] == m) & (f['steps'] == s)
            if reach[k].sum() > 1e-6:
                tab[f'{m},{s}'] = [float(reach[k].sum()), float(stopm[k].sum() / reach[k].sum())]
    out['stop_by_markers_steps'] = tab
    return out

def check_turn(sol, tp):
    tot = tp['stopm'].sum() + tp['bustm'].sum()
    v = (tp['stopm'] * np.where(tp['stopm'] > 0, sol.stop, 0)).sum() + tp['bustm'].sum() * sol.bust_val
    assert abs(tot - 1) < 1e-9 and abs(v - sol.best[0]) < 1e-9, (tot, v, sol.best[0])
    assert abs(rule_value(sol, np.where(tp['act_stop'], 1, 2)) - sol.best[0]) < 1e-9
    return float(tot)

# ---- self-play
def _dist(prog, q):
    # q's steps to a win: 3 - claimed columns cheapest open columns' remaining steps (opponent-claimed columns are out)
    rem = sorted(int(LEN[c] - prog[q, c]) for c in COLS if not (prog[1 - q, c] == LEN[c]))
    return sum(rem[:3])

def play_game(V, seed):
    rng = np.random.default_rng(seed)
    prog, p, turn = np.zeros((2, 13), np.int8), 0, 0
    D, Tn, claims = [], [], []
    mv = np.empty((fr.MAX_PICKS, 2), np.int8)
    while True:
        sol = solve_turn(prog, p, V)
        thr = solve_turn(prog, 1 - p, ONE).best[0]  # opponent's best P(win next turn) on this board
        vn = [float(x) for x in (V(prog[None], p)[0], V(prog[None], 1 - p)[0])]
        base = dict(turn=turn, p=p, v_me=float(sol.best[0]), vnet_me=vn[0], vnet_opp=vn[1], threat=float(thr),
                    won_me=int(fr.score(prog, p)), won_opp=int(fr.score(prog, 1 - p)), dist_me=_dist(prog, p), dist_opp=_dist(prog, 1 - p))
        tmp, first, out = np.zeros(13, np.int8), None, None
        while out is None:
            r = fr.roll_index(rng.integers(1, 7, 4))
            n = fr.roll_moves(fr.legal_moves(prog, p, tmp), r, mv)
            if first is None:
                avail = sorted({int(c) for a, b in mv[:n] for c in (a, b) if c})
            if n == 0:
                out = 'bust'
                break
            cands = [fr.apply_move(prog, p, tmp, a, b) for a, b in mv[:n]]
            j = int(np.argmax([sol.value(t) for t in cands]))
            if first is None:
                first = sorted({int(c) for c in mv[j] if c})
            tmp = cands[j]
            if fr.stop_wins(prog, p, tmp):
                out = 'win'
                break
            i = sol.row(tmp)
            go = bool(sol.go[i] > sol.stop[i])
            D.append(dict(base, prog=prog.copy(), tmp=tmp.copy(), stop=float(sol.stop[i]), go=float(sol.go[i]),
                          pbust=float(sol.pbust[i]), bank=not go))
            if not go:
                out = 'bank'
        adv = int(np.where(tmp > 0, tmp - prog[p], 0).sum()) if out != 'bust' else 0
        Tn.append(dict(base, first=first or [], avail=avail, out=out, banked_steps=adv, prog=prog.copy()))
        if out != 'bust':
            new = fr.apply_stop(prog, p, tmp)
            for c in COLS:
                if new[p, c] == LEN[c] and prog[p, c] < LEN[c]:
                    claims.append((int(c), p, turn))
            prog = new
        if out == 'win':
            return dict(D=D, T=Tn, claims=claims, winner=p, turns=turn + 1, final=prog)
        p, turn = 1 - p, turn + 1

_V = {}

def _job(args):
    model, seeds = args
    if model not in _V:
        _V[model] = load(model)
    return [play_game(_V[model], s) for s in seeds]

def selfplay(model, n, workers, seed0=7000):
    t = time.time()
    seeds = [seed0 + g for g in range(n)]
    with Pool(workers) as pool:
        games = [g for part in pool.imap(_job, [(model, seeds[i:i + 10]) for i in range(0, n, 10)]) for g in part]
    dkeys = ['turn', 'p', 'v_me', 'vnet_me', 'vnet_opp', 'threat', 'won_me', 'won_opp', 'dist_me', 'dist_opp', 'stop', 'go', 'pbust', 'bank']
    tkeys = ['turn', 'p', 'v_me', 'vnet_me', 'vnet_opp', 'threat', 'won_me', 'won_opp', 'dist_me', 'dist_opp', 'banked_steps']
    arr = {}
    D = [(g, d) for g, x in enumerate(games) for d in x['D']]
    T = [(g, d) for g, x in enumerate(games) for d in x['T']]
    for k in dkeys:
        arr['d_' + k] = np.array([d[k] for _, d in D])
    arr['d_game'], arr['d_prog'], arr['d_tmp'] = np.array([g for g, _ in D]), np.array([d['prog'] for _, d in D]), np.array([d['tmp'] for _, d in D])
    for k in tkeys:
        arr['t_' + k] = np.array([d[k] for _, d in T])
    arr['t_game'], arr['t_prog'] = np.array([g for g, _ in T]), np.array([d['prog'] for _, d in T])
    arr['t_out'] = np.array([d['out'] for _, d in T])
    arr['t_first'] = np.array([np.isin(COLS, d['first']) for _, d in T])
    arr['t_avail'] = np.array([np.isin(COLS, d['avail']) for _, d in T])
    arr['g_winner'], arr['g_turns'] = np.array([x['winner'] for x in games]), np.array([x['turns'] for x in games])
    arr['g_final'] = np.array([x['final'] for x in games])
    arr['c_game'] = np.array([g for g, x in enumerate(games) for _ in x['claims']])
    arr['c_col'], arr['c_p'], arr['c_turn'] = (np.array([c[k] for x in games for c in x['claims']]) for k in range(3))
    os.makedirs(CACHE, exist_ok=True)
    np.savez_compressed(os.path.join(CACHE, f'selfplay_{model}.npz'), **arr)
    print(f'{model}: {n} games, {len(D)} decisions, {time.time() - t:.0f}s')

# ---- opening
def tree_rule(sol, tp, depth):
    from sklearn.tree import DecisionTreeClassifier, export_text
    f = feats(sol, tp)
    X = np.stack(list(f.values()), 1)
    dec, reach = tp['dec'], tp['reach']
    clf = DecisionTreeClassifier(max_depth=depth, min_weight_fraction_leaf=0.002, random_state=0)
    clf.fit(X[dec], tp['act_stop'][dec], sample_weight=reach[dec])
    pred = clf.predict(X).astype(bool)  # the rule acts on every row
    pred[0] = False  # the empty turn start can't bank
    return export_text(clf, feature_names=list(f), decimals=3), pred

def rule_report(sol, tp, pred):
    # agreement: reach-weighted over the policy's decisions; myopic cost: reach x |go - stop| summed over disagreements;
    # exact cost: V(optimal) - V(rule as the stop policy, picks re-optimized), both in win-probability points
    dec, reach = tp['dec'], tp['reach']
    w = reach[dec]
    agree = pred[dec] == tp['act_stop'][dec]
    gap = np.abs(sol.go - sol.stop)[dec]
    v_rule = rule_value(sol, np.where(pred, 1, 2))
    return dict(agreement=float(w[agree].sum() / w.sum()), myopic_cost_pts=float(100 * (w * gap)[~agree].sum()),
                exact_cost_pts=float(100 * (sol.best[0] - v_rule)), v_rule=v_rule, v_opt=float(sol.best[0]))

def simple_rules(sol, tp):
    f = feats(sol, tp)
    m3, top, rel, pb = f['markers'] == 3, f['topped'] > 0, f['rel_steps'], f['pbust']
    ratio = rel * pb / np.maximum(1 - pb, 1e-9)
    R = {'bank iff a marker is at the top of its column': top}
    R |= {f'bank iff 3 markers and steps >= {k}': m3 & (f['steps'] >= k) for k in range(8, 21, 2)}
    R |= {f'bank iff 3 markers and rel >= {x:.1f}': m3 & (rel >= x - 1e-9) for x in np.arange(1.2, 2.21, 0.2)}
    R |= {f'bank iff topped, or 3 markers and rel >= {x:.1f}': top | (m3 & (rel >= x - 1e-9)) for x in np.arange(1.2, 2.21, 0.2)}
    R |= {f'bank iff topped, or 3 markers and rel * pbust / (1 - pbust) >= {c:.2f}': top | (m3 & (ratio >= c)) for c in (0.2, 0.25, 0.3, 0.35, 0.4)}
    return [dict(rule=k, **rule_report(sol, tp, v)) for k, v in R.items()]

def outcomes_table(sol, tp):
    # value of every one-roll result from the empty turn: one marker +1 / +2, or two markers +1 each
    V = {}
    for a in COLS:
        for b in [0] + list(COLS):
            if b and b < a:
                continue
            t = fr.apply_move(sol.prog, sol.p, NOPP, a, b)
            if t[a] > LEN[a] or (b and t[b] > LEN[b]) or fr.key(t) not in sol.index:
                continue
            V['+'.join(str(c) for c in (a, b) if c)] = float(sol.value(t))
    return dict(sorted(V.items(), key=lambda kv: -kv[1]))

def first_roll_choices(sol, tp):
    # for each first roll: options (column sets) and choice; summaries of how often the choice uses 7 / 6-8 / outer
    mv = np.empty((fr.MAX_PICKS, 2), np.int8)
    rows, col_first = [], np.zeros(13)
    for r in range(fr.N_ROLLS):
        n = fr.roll_moves(fr.legal_moves(sol.prog, sol.p, NOPP), r, mv)
        opts = [tuple(int(c) for c in m if c) for m in mv[:n]]
        vals = [float(sol.value(fr.apply_move(sol.prog, sol.p, NOPP, a, b))) for a, b in mv[:n]]
        if not n:
            continue
        j = int(np.argmax(vals))
        for c in set(opts[j]):
            col_first[c] += fr.ROLL_PROB[r]
        rows.append(dict(dice=fr.ROLL_DICE[r].tolist(), prob=float(fr.ROLL_PROB[r]), options=[list(o) for o in opts],
                         values=vals, choice=list(opts[j])))
    return rows, {int(c): float(col_first[c]) for c in COLS}

def trade_stats(rows):
    # among first rolls that offer a choice: how often the chosen option beats alternatives of a given kind
    grp = lambda o: '7' if 7 in o else '6/8' if set(o) & {6, 8} else 'outer(2,3,11,12)' if set(o) <= {2, 3, 11, 12} else 'mid(4,5,9,10)'
    out = {}
    for r in rows:
        if len(r['options']) < 2:
            continue
        groups = {grp(o) for o in r['options']}
        if len(groups) < 2:
            continue
        key = ' vs '.join(sorted(groups))
        d = out.setdefault(key, Counter())
        d[grp(r['choice'])] += r['prob']
        d['_total'] += r['prob']
    return {k: {g: round(v / d['_total'], 3) for g, v in d.items() if g != '_total'} | dict(prob=round(d['_total'], 4)) for k, d in out.items()}

def opening_for(V, prog, p, trees=True):
    sol = solve_turn(prog, p, V)
    tp = turn_policy(sol)
    check_turn(sol, tp)
    out = dict(board=prog.tolist(), p=p, **summarize_turn(sol, tp))
    f, dec, reach = feats(sol, tp), tp['dec'], tp['reach']
    k = dec & (f['topped'] > 0)
    out['p_bank_given_topped'] = float(tp['stopm'][k].sum() / max(reach[k].sum(), 1e-12))
    k = dec & (f['topped'] == 0)
    out['p_bank_not_topped'] = float(tp['stopm'][k].sum() / max(reach[k].sum(), 1e-12))  # per decision, like the above
    out['p_turn_plain_bank'] = float(tp['stopm'][k].sum())  # share of turns ending in a bank with nothing topped
    rb = np.round(f['rel_steps'] / 0.25) * 0.25
    out['bank_by_rel'] = {nm: [dict(x=float(x), reach=float(reach[k & (rb == x)].sum()),
                                    p_bank=float(tp['stopm'][k & (rb == x)].sum() / reach[k & (rb == x)].sum()))
                               for x in np.unique(rb[k]) if reach[k & (rb == x)].sum() > 0]
                          for nm, k in (('topped', dec & (f['topped'] > 0)), ('not_topped', dec & (f['topped'] == 0)))}
    sr = simple_rules(sol, tp)
    out['simple_rules_best'] = sorted(sr, key=lambda x: x['exact_cost_pts'])[:6]
    if trees:
        out['simple_rules'] = sr
        out['trees'] = {}
        for d in (1, 2, 3):
            txt, pred = tree_rule(sol, tp, d)
            out['trees'][d] = dict(text=txt, **rule_report(sol, tp, pred))
    return sol, tp, out

def bust_odds():
    # P(the next roll busts) with markers on exactly these 3 columns (below their tops), nothing blocked
    from itertools import combinations
    prog, out = np.zeros((2, 13), np.int8), {}
    for cs in combinations(range(2, 13), 3):
        t = np.zeros(13, np.int8)
        t[list(cs)] = 1
        out['-'.join(map(str, cs))] = float(sum(fr.ROLL_PROB[r] for r in range(fr.N_ROLLS) if fr.bust(prog, 0, t, r)))
    return out

def opening(models):
    res = dict(bust_odds=bust_odds())
    for model in models:
        V = load(model)
        prog = np.zeros((2, 13), np.int8)
        sol, tp, o = opening_for(V, prog, 0)
        o['one_roll_values'] = outcomes_table(sol, tp)
        rows, o['first_marker_prob'] = first_roll_choices(sol, tp)
        o['trades'] = trade_stats(rows)
        o['first_roll_choices'] = rows
        o['check_terminal_mass'] = check_turn(sol, tp)
        # mirror check: bank probability per column vs its mirror, and the empty-board value from either seat
        cb = o['col_banked']
        o['mirror_gap_col_banked'] = max(abs(cb[c] - cb[14 - c]) for c in COLS)
        o['v_p1_empty'] = float(solve_turn(prog, 1, V).best[0])
        res[model] = dict(first_mover=o)
        # second mover: most common boards after the first mover's turn 1 in self-play
        f = os.path.join(CACHE, f'selfplay_{model}.npz')
        if os.path.exists(f):
            z = np.load(f)
            k = (z['t_turn'] == 1)
            boards = [b.tobytes() for b in z['t_prog'][k]]
            cnt = Counter(boards)
            sec = []
            for b, c in cnt.most_common(6):
                pb = np.frombuffer(b, np.int8).reshape(2, 13).copy()
                _, _, s = opening_for(V, pb, 1, trees=False)
                sec.append(dict(share=c / len(boards), opp_banked={int(cc): int(pb[0, cc]) for cc in COLS if pb[0, cc]}, **s))
            # sample: expected banked steps of the second mover vs how much the first mover banked
            rng = np.random.default_rng(0)
            idx = rng.choice(np.flatnonzero(k), min(300, k.sum()), replace=False)
            samp = []
            for i in idx:
                pb = z['t_prog'][i]
                s = solve_turn(pb, 1, V)
                tp2 = turn_policy(s)
                f2 = feats(s, tp2)
                samp.append([int(pb[0].sum()), float((tp2['stopm'] * f2['steps']).sum()), float(tp2['bustm'].sum()), float(s.best[0])])
            samp = np.array(samp)
            by = {}
            for lo, hi in ((0, 0), (1, 3), (4, 5), (6, 7), (8, 99)):
                m = (samp[:, 0] >= lo) & (samp[:, 0] <= hi)
                if m.any():
                    by[f'{lo}-{hi}'] = dict(n=int(m.sum()), exp_steps_banked=float(samp[m, 1].mean()), p_bust=float(samp[m, 2].mean()), v=float(samp[m, 3].mean()))
            res[model]['second_mover'] = dict(common=sec, by_opp_steps=by)
        print(model, 'opening done')
    return res

# ---- whole-game summaries
def hit_prob():
    # P(a roll offers column c in some pair)
    return {int(c): float(sum(fr.ROLL_PROB[r] for r in range(fr.N_ROLLS) if c in fr.ROLL_PAIRS[r, :fr.ROLL_NPAIRS[r]])) for c in COLS}

def turns_to_top(c):
    return float(turns_left(c)[0])

def turns_left(c, iters=2000):
    # T[h]: expected turns to top column c from height h playing it alone (all other columns ignored), stopping optimally (value iteration).
    # per roll: +2 if some split gives (c, c), else +1 if some pair sums to c, else bust
    p2 = sum(fr.ROLL_PROB[r] for r in range(fr.N_ROLLS) if any(tuple(x) == (c, c) for x in fr.ROLL_PAIRS[r, :fr.ROLL_NPAIRS[r]]))
    p1 = hit_prob()[c] - p2
    L = int(LEN[c])
    T = np.zeros(L + 3)
    for _ in range(iters):
        new = T.copy()
        for h in range(L):
            W = np.zeros(L + 3)  # W[t]: turns left after this turn when holding tmp height t (>= L: topped)
            for t in range(L - 1, h, -1):
                W[t] = min(T[t], p2 * W[min(t + 2, L)] + p1 * W[t + 1] + (1 - p1 - p2) * T[h])
            new[h] = 1 + p2 * W[min(h + 2, L)] + p1 * W[h + 1] + (1 - p1 - p2) * T[h]
        if np.abs(new - T).max() < 1e-12:
            break
        T = new
    return T[:LEN[c] + 1]

def mean_or_none(x):
    return float(x.mean()) if len(x) else None

def rate_table(x, y, bins):
    idx = np.digitize(x, bins) - 1
    return [dict(lo=float(bins[b]), hi=float(bins[b + 1]), n=int((idx == b).sum()), rate=float(y[idx == b].mean()) if (idx == b).any() else None)
            for b in range(len(bins) - 1)]

def games_summary(model):
    z = dict(np.load(os.path.join(CACHE, f'selfplay_{model}.npz')))
    G = len(z['g_winner'])
    win_by_c = z['c_p'] == z['g_winner'][z['c_game']]
    hp = hit_prob()
    cost = {c: LEN[c] / hp[c] for c in COLS}  # rolls needed to top at one step per roll that offers c
    tt = {int(c): turns_to_top(c) for c in COLS}  # difficulty used for the normalised rate
    claim_any = {int(c): float(np.isin(np.arange(G), z['c_game'][z['c_col'] == c]).mean()) for c in COLS}
    claim_win = {int(c): float(len(set(z['c_game'][(z['c_col'] == c) & win_by_c])) / G) for c in COLS}
    norm = {c: claim_any[c] * tt[c] / np.mean([claim_any[k] * tt[k] for k in claim_any]) for c in claim_any}
    order = {}
    firsts = Counter()
    for g in range(G):
        k = np.flatnonzero(z['c_game'] == g)
        k = k[np.argsort(z['c_turn'][k], kind='stable')]
        for rank, i in enumerate(k):
            order.setdefault(int(z['c_col'][i]), []).append(rank + 1)
        if len(k):
            firsts[int(z['c_col'][k[0]])] += 1
    wins = Counter(tuple(int(c) for c in COLS if z['g_final'][g, z['g_winner'][g], c] == LEN[c]) for g in range(G))
    wins3 = Counter({k: v for k, v in wins.items() if len(k) == 3})
    out = dict(model=model, games=G, first_player_win=float((z['g_winner'] == 0).mean()), mean_turns=float(z['g_turns'].mean()),
               hit_prob=hp, rolls_to_top={int(c): float(v) for c, v in cost.items()}, turns_to_top=tt,
               claim_any=claim_any, claim_winner=claim_win, claim_norm=norm,
               mean_claim_rank={c: float(np.mean(v)) for c, v in sorted(order.items())},
               first_claimed={int(c): firsts[c] / G for c in COLS},
               top_win_sets=[[list(k), v / G] for k, v in wins3.most_common(12)],
               win_with_4_columns=sum(v for k, v in wins.items() if len(k) > 3) / G)
    # mirror-pooled
    out['claim_any_pooled'] = {f'{c}/{14 - c}' if c != 7 else '7': (claim_any[c] + claim_any[14 - c]) / 2 for c in range(2, 8)}
    out['claim_norm_pooled'] = {f'{c}/{14 - c}' if c != 7 else '7': (norm[c] + norm[14 - c]) / 2 for c in range(2, 8)}
    # starts: P(first pick uses c | c offered on the turn's first roll), by phase (turn number)
    first, avail, turn = z['t_first'], z['t_avail'], z['t_turn']
    phases = dict(opening=(0, 3), early=(4, 9), mid=(10, 19), late=(20, 999))
    st = {}
    for ph, (lo, hi) in phases.items():
        m = (turn >= lo) & (turn <= hi) & avail.any(1)
        st[ph] = dict(n=int(m.sum()), share={int(c): float(first[m, i].mean()) for i, c in enumerate(COLS)},
                      take_if_offered={int(c): mean_or_none(first[m, i][avail[m, i]]) for i, c in enumerate(COLS)})
    # continuing vs fresh: does the first pick go on a column where the mover already has progress?
    tp, p = z['t_prog'], z['t_p']
    own = tp[np.arange(len(p)), p][:, 2:] > 0
    m = avail.any(1) & first.any(1)
    st['first_on_existing_progress'] = float((first[m] & own[m]).any(1).mean())
    st['offered_existing_progress'] = float((avail[m] & own[m]).any(1).mean())
    out['starts'] = st
    # bail-out: go/stop decisions (winning banks are never decisions); split by whether banking would top a column
    bank, thr, v, tmp = z['d_bank'], z['d_threat'], z['d_v_me'], z['d_tmp']
    I = np.arange(len(bank))
    mine = z['d_prog'][I, z['d_p']]
    top = ((tmp == LEN) & (tmp > 0)).any(1)
    steps = np.where(tmp > 0, tmp - mine, 0).sum(1)
    vb, tb = np.linspace(0, 1, 11), np.array([0, .02, .05, .1, .15, .2, .3, .4, .5, .7, 1.0001])
    for nm, x, bins in (('threat', thr, tb), ('v', v, vb), ('opp_dist', z['d_dist_opp'], np.array([1, 2, 3, 4, 5, 7, 10, 15, 20]))):
        out[f'bank_vs_{nm}'] = dict(all=rate_table(x, bank, bins), top=rate_table(x[top], bank[top], bins), no_top=rate_table(x[~top], bank[~top], bins))
    late = bank & ~top & (z['d_turn'] >= 2)  # turns 0-1 (deep opening banks) excluded; "rate" = mean steps banked
    out['steps_at_bank_vs_v'] = rate_table(v[late], steps[late], vb)
    out['steps_at_bank_vs_threat'] = rate_table(thr[bank & ~top], steps[bank & ~top], tb)
    out['bank_by_score'] = [dict(won_me=i, won_opp=j, n=int(m.sum()), bank=mean_or_none(bank[m]), bank_no_top=mean_or_none(bank[m & ~top]),
                                 bank_if_top=mean_or_none(bank[m & top]), n_top=int((m & top).sum()))
                            for i in range(3) for j in range(3) for m in [(z['d_won_me'] == i) & (z['d_won_opp'] == j)] if m.any()]
    vb2, tb2 = np.array([0, .2, .35, .5, .65, .8, 1]), np.array([0, .05, .15, .3, .5, 1.0001])
    cell = (np.digitize(v, vb2) - 1) * 10 + np.digitize(thr, tb2) - 1
    out['bank_no_top_grid'] = dict(v_bins=vb2.tolist(), threat_bins=tb2.tolist(),
                                   n=[[int(((cell == 10 * i + j) & ~top).sum()) for j in range(5)] for i in range(6)],
                                   rate=[[float(bank[(cell == 10 * i + j) & ~top].mean()) if ((cell == 10 * i + j) & ~top).any() else None
                                          for j in range(5)] for i in range(6)])
    out['threat_at_no_top_bank_quantiles'] = {q: float(np.quantile(thr[bank & ~top], q)) for q in (0.5, 0.9, 0.99, 0.999)}
    tn = z['t_out']
    out['turn_outcomes'] = {k: float((tn == k).mean()) for k in ('bank', 'bust', 'win')}
    out['turn_bank_vs_threat'] = rate_table(z['t_threat'], tn == 'bank', tb)
    # per turn: did it end in a bank that topped a column / one that did not (bins of the turn-start threat and V)
    tk = z['t_game'] * 1000 + z['t_turn']
    bt, bn = np.isin(tk, (z['d_game'] * 1000 + z['d_turn'])[bank & top]), np.isin(tk, (z['d_game'] * 1000 + z['d_turn'])[bank & ~top])
    for nm, x, bins in (('threat', z['t_threat'], tb), ('v', z['t_v_me'], vb)):
        out[f'turn_vs_{nm}'] = dict(bank_top=rate_table(x, bt, bins), bank_no_top=rate_table(x, bn, bins), win=rate_table(x, tn == 'win', bins),
                                    bust=rate_table(x, tn == 'bust', bins))
    out['n_decisions'], out['n_turns'], out['share_decisions_top'] = int(len(bank)), int(len(tn)), float(top.mean())
    return out

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['selfplay', 'opening', 'games'])
    ap.add_argument('--model', nargs='+', default=['ens'])
    ap.add_argument('--games', type=int, default=3000)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.cmd == 'selfplay':
        for m in a.model:
            selfplay(m, a.games, a.workers)
    elif a.cmd == 'opening':
        f = os.path.join(OUT, 'opening.json')
        res = json.load(open(f)) if os.path.exists(f) else {}
        json.dump(res | opening(a.model), open(f, 'w'), indent=1)
    else:
        f = os.path.join(OUT, 'games.json')
        res = json.load(open(f)) if os.path.exists(f) else {}
        json.dump(res | {m: games_summary(m) for m in a.model}, open(f, 'w'), indent=1)
