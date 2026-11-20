import os, sys, json, time, argparse
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from multiprocessing import Pool
import numpy as np
from numba import njit
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, 'analysis')]
import fastrules as fr
from fastrules import LEN
from turn_solver import solve_turn
import strategy_profile as sp

# Follow-ups to the strategy profile: declined topping picks (re-solved logged turn starts) and countable bail-out / race count.
# python analysis/strategy_followup.py tops --turns N ; summary
OUT, CACHE = sp.OUT, sp.CACHE
HIT = np.array([0, 0] + [sp.hit_prob()[c] for c in range(2, 13)])

# ---- declined tops: every (row, roll) of the policy's turn where some pick tops a new column
@njit(cache=True)
def _newtop(T, x, i):
    for c in range(2, 13):
        if T[x, c] == LEN[c] and T[i, c] < LEN[c]:
            return True
    return False

@njit(cache=True)
def _offers(prog, p, rows, best, expand, reach, T, M, u0):
    # pass over reached rolling rows x rolls offering a topping pick. tot: [P(offer), P(offer declined), sum P x value lost by
    # topping when declined]; systematic sample of M offers by probability: (row, roll, chosen, best topping, best other or -1)
    tot = np.zeros(3)
    ev = np.full((M, 5), -1, np.int64)
    mv = np.empty((fr.MAX_PICKS, 2), np.int8)
    for ps in range(2):
        cum, q = 0.0, 0
        for i in range(len(best)):
            if reach[i] == 0 or not expand[i]:
                continue
            ok = fr.legal_moves(prog, p, T[i])
            for r in range(fr.N_ROLLS):
                m = fr.roll_moves(ok, r, mv)
                j = k = l = -1
                for e in range(m):
                    x = rows[fr.slot(fr.apply_move(prog, p, T[i], mv[e, 0], mv[e, 1]))]
                    if j < 0 or best[x] > best[j]:
                        j = x
                    if _newtop(T, x, i):
                        if k < 0 or best[x] > best[k]:
                            k = x
                    elif l < 0 or best[x] > best[l]:
                        l = x
                if k < 0:
                    continue
                w = reach[i] * fr.ROLL_PROB[r]
                if ps == 0:
                    tot[0] += w
                    if best[j] > best[k]:  # exact value ties count as taken (the top is as good)
                        tot[1] += w
                        tot[2] += w * (best[j] - best[k])
                else:
                    cum += w
                    while q < M and (q + u0) / M * tot[0] < cum:
                        ev[q] = (i, r, j, k, l)
                        q += 1
    return tot, ev

@njit(cache=True)
def _outcomes(child, best, act_stop, order, T):
    # under the policy, per row: [P(bust), P(the turn ends with column c topped) for c in 2..12]
    ptr, cmv, crow = child
    out = np.zeros((len(best), 12))
    urow, srow = np.empty(len(fr.UPAIRS), np.int64), np.empty(13, np.int64)
    for i in order:
        if act_stop[i] and i > 0:
            for c in range(2, 13):
                out[i, c - 1] = T[i, c] == LEN[c]
            continue
        sp._row_moves(ptr, cmv, crow, i, best, urow, srow)
        for r in range(fr.N_ROLLS):
            j = sp._roll_row(urow, r, best)
            if j < 0:
                out[i, 0] += fr.ROLL_PROB[r]
            else:
                out[i] += fr.ROLL_PROB[r] * out[j]
    return out

@njit(cache=True)
def declines(rule, x, T, i, k, l, pbust, opp):
    # whether to take the best non-topping pick l over the best topping pick k at row i. 1: never (always top);
    # 2: iff topping uses the last marker and P(bust) after l <= x; 3: same, with "l keeps >= 2 live markers on 6/7/8" for the
    # bust test; 4, 5: rules 2, 3 unless the opponent is past halfway on the column the top would claim
    if rule == 1 or (T[k] > 0).sum() < 3:
        return False
    if rule >= 4:
        for c in range(2, 13):
            if T[k, c] == LEN[c] and T[i, c] < LEN[c] and 2 * opp[c] > LEN[c]:
                return False
    if rule % 2 == 0:
        return pbust[l] <= x
    n = 0
    for c in range(6, 9):
        n += T[l, c] > 0 and T[l, c] < LEN[c]
    return n >= 2

@njit(cache=True)
def rule_value(child, stop, bust_val, order, pbust, T, opp, rule, x):
    # exact turn value when topping picks follow the rule (rule 0: free), other picks and bank/go re-optimized
    ptr, cmv, crow = child
    val = stop.copy()
    srow = np.empty(13, np.int64)
    ut, un = np.empty(len(fr.UPAIRS), np.int64), np.empty(len(fr.UPAIRS), np.int64)
    for i in order:
        if stop[i] >= 1.0:
            continue
        srow[:] = -1
        ut[:] = -1
        un[:] = -1
        for e in range(ptr[i], ptr[i + 1]):
            if cmv[e] < 11:
                srow[cmv[e] + 2] = crow[e]
        for u in range(len(fr.UPAIRS)):
            for x_ in (srow[fr.UPAIRS[u, 0]], srow[fr.UPAIRS[u, 1]]):
                if x_ >= 0:
                    if _newtop(T, x_, i):
                        if ut[u] < 0 or val[x_] > val[ut[u]]:
                            ut[u] = x_
                    elif un[u] < 0 or val[x_] > val[un[u]]:
                        un[u] = x_
        for e in range(ptr[i], ptr[i + 1]):
            if cmv[e] >= 11:
                u, x_ = cmv[e] - 11, crow[e]
                ut[u], un[u] = (x_, -1) if _newtop(T, x_, i) else (-1, x_)
        g = 0.0
        for r in range(fr.N_ROLLS):
            k = l = -1
            for a in range(fr.ROLL_NPAIRS[r]):
                u = fr.ROLL_UPAIR[r, a]
                if ut[u] >= 0 and (k < 0 or val[ut[u]] > val[k]):
                    k = ut[u]
                if un[u] >= 0 and (l < 0 or val[un[u]] > val[l]):
                    l = un[u]
            if k < 0 and l < 0:
                v = bust_val
            elif k < 0:
                v = val[l]
            elif l < 0 or stop[k] >= 1.0:
                v = val[k]
            elif rule == 0:
                v = max(val[k], val[l])
            else:
                v = val[l] if declines(rule, x, T, i, k, l, pbust, opp) else val[k]
            g += fr.ROLL_PROB[r] * v
        val[i] = max(stop[i], g) if i > 0 else g
    return val[0]

XS = (0.12, 0.14, 0.16, 0.20)
RULES = [(1, 0.0)] + [(2, x) for x in XS] + [(3, 0.0)] + [(4, x) for x in XS] + [(5, 0.0)]

def turn_events(prog, p, V, rng, M=60):
    sol = solve_turn(prog, p, V)
    tp = sp.turn_policy(sol)
    T, act = tp['T'], tp['act_stop']
    pb = np.nan_to_num(sol.pbust, nan=1.0)
    rv = [rule_value(sol.child, sol.stop, sol.bust_val, sol.order, pb, T, prog[1 - p], r, x) for r, x in [(0, 0.0)] + RULES]
    assert abs(rv[0] - sol.best[0]) < 1e-9
    tot, ev = _offers(prog, p, sol.index.rows, sol.best, ~act, tp['reach'], T, M, rng.random())
    tot = np.r_[tot, sol.best[0] - np.array(rv[1:])]
    if tot[0] == 0:
        return None, tot
    I, R, J, K, L = ev.T
    oc = _outcomes(sol.child, sol.best, act, sol.order, T)
    c = ((T[K] == LEN) & (T[I] < LEN)).argmax(1)  # the column the best topping pick tops (lowest if two)
    n = np.arange(M)
    hc = lambda x: np.where(T[x, c] > 0, T[x, c], prog[p, c])  # height on c (banked when no marker)
    adv = lambda x: np.where(T[x] > 0, T[x] - prog[p], 0).sum(1)
    L_ = np.maximum(L, 0)
    has_l = L >= 0
    rule = np.array([[has_l[e] and sol.stop[K[e]] < 1 and declines(r, x, T, I[e], K[e], L[e], pb, prog[1 - p]) for e in n] for r, x in RULES])
    return dict(w=np.full(M, tot[0] / M), roll=R, c=c, declined=sol.best[J] > sol.best[K],
                has_alt=has_l, gap=np.where(has_l, sol.best[L_] - sol.best[K], np.nan), v=np.full(M, sol.best[0]),
                k_bank=act[K], k_win=sol.stop[K] >= 1, l_bank=act[L_], pb_l=np.where(has_l, sol.pbust[L_], np.nan), pb_k=sol.pbust[K],
                need_i=LEN[c] - hc(I), need_l=LEN[c] - hc(L_), markers_i=(T[I] > 0).sum(1), markers_k=(T[K] > 0).sum(1),
                markers_l=(T[L_] > 0).sum(1), d_l=adv(L_) - adv(I), d_k=adv(K) - adv(I), l_ncols=(T[L_] != T[I]).sum(1),
                l_opp=((T[L_] != T[I]) & (prog[1 - p] > 0)).any(1), k_opp=prog[1 - p, c] > 0, steps_i=adv(I),
                topped_i=((T[I] == LEN) & (T[I] > 0)).sum(1), later_top=oc[L_, c - 1], later_bust=oc[L_, 0],
                rule=rule.T, tmp_i=T[I], tmp_k=T[K], tmp_l=T[L_]), tot

_V = {}

def _job(args):
    model, idx = args
    if model not in _V:
        _V[model] = sp.load(model)
    z = np.load(os.path.join(CACHE, f'selfplay_{model}.npz'))
    res, totals = [], []
    for t in idx:
        e, tot = turn_events(z['t_prog'][t], int(z['t_p'][t]), _V[model], np.random.default_rng(int(t)))
        totals.append(np.r_[t, tot])
        if e is not None:
            res.append(e | dict(turn_idx=np.full(len(e['w']), t)))
    return {k: np.concatenate([r[k] for r in res]) for k in res[0]} if res else {}, np.array(totals)

def tops(model, n, workers, seed=0):
    t0 = time.time()
    z = np.load(os.path.join(CACHE, f'selfplay_{model}.npz'))
    N = len(z['t_p'])
    idx = np.sort(np.random.default_rng(seed).choice(N, min(n, N), replace=False))
    with Pool(workers) as pool:
        parts = list(pool.imap_unordered(_job, [(model, idx[i:i + 25]) for i in range(0, len(idx), 25)]))
    E = {k: np.concatenate([x[0][k] for x in parts if x[0]]) for k in next(x[0] for x in parts if x[0])}
    E['totals'] = np.concatenate([x[1] for x in parts])
    np.savez_compressed(os.path.join(CACHE, f'tops_{model}.npz'), **E)
    print(f'{model}: {len(idx)} turns, {len(E["w"])} events, {time.time() - t0:.0f}s')

@njit(cache=True)
def pbust_of(prog, p, tmp):
    s = 0.0
    for r in range(fr.N_ROLLS):
        if fr.bust(prog, p, tmp, r):
            s += fr.ROLL_PROB[r]
    return s

def wquantile(x, w, q):
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return x[o][np.searchsorted(cw, q * cw[-1])]

def tops_summary(model):
    E = dict(np.load(os.path.join(CACHE, f'tops_{model}.npz')))
    z = dict(np.load(os.path.join(CACHE, f'selfplay_{model}.npz')))
    tot, t = E['totals'], E['turn_idx']
    w, c = E['w'], E['c']
    d = E['declined'] & (np.nan_to_num(E['gap']) > 0)  # ties count as taken (none in the cached runs)
    nw = ~E['k_win'] & E['has_alt']  # a real choice: the top doesn't win and some non-topping pick exists
    J = np.arange(len(z['t_p']))
    co = race_count(z['t_prog'][J, 1 - z['t_p']], z['t_prog'][J, z['t_p']])[t]
    share = lambda m: float(w[m].sum() / w.sum())
    rate = lambda m: float(w[m & d].sum() / w[m].sum())
    mean = lambda x, m: float((w * x)[m].sum() / w[m].sum())
    out = dict(model=model, turns=len(tot), sampled_offers=int(len(w)), offers_per_turn=float(tot[:, 1].mean()),
               declined_per_turn=float(tot[:, 2].mean()), declined_share=float(tot[:, 2].sum() / tot[:, 1].sum()),
               declined_share_real_choice=rate(nw), real_choice_share=share(nw), winning_top_share=share(E['k_win']),
               value_lost_by_topping_pts=float(100 * tot[:, 3].sum() / tot[:, 2].sum()),
               gap_quantiles_pts={q: float(100 * wquantile(E['gap'][d & nw], w[d & nw], q)) for q in (0.1, 0.5, 0.9)})
    pooled = np.minimum(c, 14 - c)
    groups = dict(column={f'{a}/{14 - a}' if a < 7 else '7': nw & (pooled == a) for a in range(2, 8)},
                  opp_lanes={str(k): nw & (z['t_won_opp'][t] == k) for k in range(3)},
                  own_lanes={str(k): nw & (z['t_won_me'][t] == k) for k in range(2)},
                  opp_count={f'{a}-{b}': nw & (co >= a) & (co < b) for a, b in ((0, 15), (15, 30), (30, 60), (60, 90), (90, 999))},
                  threat={f'{a}-{b}': nw & (z['t_threat'][t] >= a) & (z['t_threat'][t] < b) for a, b in ((0, .05), (.05, .15), (.15, .3), (.3, .5), (.5, 1.01))},
                  would_bank_after_topping={'yes': nw & E['k_bank'], 'no': nw & ~E['k_bank']},
                  markers_after_topping={str(k): nw & (E['markers_k'] == k) for k in (1, 2, 3)},
                  steps_to_top_before_roll={str(k): nw & (E['need_i'] == k) for k in (1, 2)})
    out['decline_rate_by'] = {g: {k: dict(share=share(m), declined=rate(m)) for k, m in G.items() if w[m].sum() > 0} for g, G in groups.items()}
    # (a) live vs dead: P(next roll busts) after the top, after the other pick, and on the topped tmp with c one below its top
    P, pp, n = z['t_prog'][t], z['t_p'][t], np.arange(len(w))
    low = E['tmp_k'].copy()
    low[n, c] -= 1
    okl = low[n, c] > P[n, pp, c]  # still a marker there
    pb_live = np.array([pbust_of(P[e], pp[e], low[e]) if okl[e] else 0.0 for e in n])
    m3 = nw & (E['markers_k'] == 3)
    out['live_vs_topped'] = {nm: dict(p_bust_after_top=mean(E['pb_k'], m), p_bust_after_other=mean(E['pb_l'], m),
                                      p_bust_same_but_c_live=mean(pb_live, m & okl), share=share(m))
                             for nm, m in (('declined', nw & d), ('taken', nw & ~d), ('declined, 3 markers', m3 & d), ('taken, 3 markers', m3 & ~d))}
    # (b) what the other pick does; (c) where the top stays; what happens after declining
    out['declined_vs_taken'] = {nm: dict(other_two_columns=mean(E['l_ncols'] == 2, m), other_steps=mean(E['d_l'], m), top_steps=mean(E['d_k'], m),
                                         other_contested=mean(E['l_opp'], m), top_contested=mean(E['k_opp'], m), hit_prob_c=mean(HIT[c], m),
                                         c_still_one_step_after_other=mean(E['need_l'] == 1, m), would_bank_after_topping=mean(E['k_bank'], m),
                                         steps_this_turn=mean(E['steps_i'], m), markers_after_top=mean(E['markers_k'], m),
                                         threat=mean(z['t_threat'][t], m))
                                for nm, m in (('declined', nw & d), ('taken', nw & ~d))}
    out['after_declining'] = dict(tops_c_later=mean(E['later_top'], nw & d), busts=mean(E['later_bust'], nw & d),
                                  banks_without_c=1 - mean(E['later_top'] + E['later_bust'], nw & d),
                                  bank_right_after_topping_when_taken=mean(E['k_bank'], nw & ~d))
    # rules: agreement on real choices (sampled offers); exact cost per turn (all turns) and per game (x turns per player).
    # a gate switches the rule on only in turns where the opponent is close (else always top); costs combine per turn
    tpp = z['g_turns'].mean() / 2
    ti = tot[:, 0].astype(int)
    tco = race_count(z['t_prog'][ti, 1 - z['t_p'][ti]], z['t_prog'][ti, z['t_p'][ti]])
    bust = lambda x: f'the other pick leaves P(bust) <= {x:.0%}'
    names = {1: 'always top', 2: bust, 3: lambda x: 'the other pick keeps 2 live markers on 6/7/8'}
    gates = {'always': (np.ones(len(tot), bool), np.ones(len(w), bool))} | \
            {f"opponent's count <= {K}": (tco <= K, co <= K) for K in (15, 20, 25, 30, 40)} | \
            {'opponent has 2 lanes': (z['t_won_opp'][ti] == 2, z['t_won_opp'][t] == 2)}
    out['rules'] = []
    for q, (r, x) in enumerate(RULES):
        for g, (gt, ge) in gates.items():
            if r == 1 and g != 'always':
                continue
            nm = 'always top' if r == 1 else 'decline iff topping uses your last marker and ' + names[2 if r % 2 == 0 else 3](x) + \
                (", unless the opponent is past halfway on that column" if r >= 4 else '')
            cost = np.where(gt, tot[:, 4 + q], tot[:, 4])
            pred = E['rule'][:, q] & ge
            out['rules'].append(dict(rule=nm, gate=g, agreement=float((w * (pred == d))[nw].sum() / w[nw].sum()),
                                     exact_cost_pts_per_turn=float(100 * cost.mean()), exact_cost_pts_per_game=float(100 * cost.mean() * tpp),
                                     se_pts_per_game=float(100 * cost.std() / np.sqrt(len(cost)) * tpp),
                                     se_vs_always_top=float(100 * (cost - tot[:, 4]).std() / np.sqrt(len(cost)) * tpp)))
    cand = np.flatnonzero(nw & d & (E['gap'] > 0.02) & (E['markers_k'] == 3))
    out['examples'] = [example(z, int(t[e]), E['tmp_i'][e], int(E['roll'][e]), model)
                       for e in np.random.default_rng(1).choice(cand, min(4, len(cand)), replace=False)]
    return out

def example(z, ti, tmp, r, model):
    prog, p = z['t_prog'][ti], int(z['t_p'][ti])
    sol = solve_turn(prog, p, sp.load(model))
    mv = np.empty((fr.MAX_PICKS, 2), np.int8)
    opts = []
    for a, b in mv[:fr.roll_moves(fr.legal_moves(prog, p, tmp), r, mv)]:
        t = fr.apply_move(prog, p, tmp, a, b)
        x = sol.row(t)
        opts.append(dict(move=[int(a), int(b)], value=float(sol.best[x]), tops=bool(((t == LEN) & (tmp < LEN))[2:].any()),
                         banks_after=bool(not sol.go[x] > sol.stop[x]), p_bust_next=float(sol.pbust[x])))
    nz = lambda x: {int(c): int(x[c]) for c in range(2, 13) if x[c]}
    return dict(me=nz(prog[p]), opp=nz(prog[1 - p]), tmp=nz(tmp), dice=fr.ROLL_DICE[r].tolist(), v_turn_start=float(sol.best[0]),
                threat=float(z['t_threat'][ti]), own_count=float(race_count(prog[p], prog[1 - p])), opp_count=float(race_count(prog[1 - p], prog[p])),
                options=sorted(opts, key=lambda o: -o['value']))

# ---- race count: half-turns to finish your best 3 columns, each played alone
STEP = np.array([0, 0, 14, 8, 5, 4, 3, 2, 3, 4, 5, 8, 14])  # ~ 2 x exact turns per step (turns_to_top / LEN), rounded

def race_count(own, opp, w=STEP):
    # own, opp: [..., 13] banked heights; sum of the 3 smallest w * remaining steps over columns opp hasn't claimed
    rem = np.where(opp == LEN, np.inf, (LEN - own) * w)[..., 2:]
    return np.sort(rem, -1)[..., :3].sum(-1)

def proxy_fit(z):
    from scipy.stats import spearmanr
    from scipy.optimize import minimize
    P, p, v = z['t_prog'], z['t_p'], z['t_v_me']
    I = np.arange(len(p))
    me, op = P[I, p], P[I, 1 - p]
    late = z['t_turn'] >= 2
    exact = np.r_[0, 0, [sp.turns_to_top(c) / LEN[c] for c in range(2, 13)]]
    lg = lambda x: 1 / (1 + np.exp(-x))
    res = {}
    for nm, w in (('exact turns/step', exact), ('half-turns/step (2,3,4,5,8,14)', STEP), ('unweighted steps', np.r_[0, 0, np.ones(11)])):
        cm, co = race_count(me, op, w), race_count(op, me, w)
        for form, x in (('log ratio', np.log((co + 1) / (cm + 1))), ('difference', co - cm)):
            f = lambda q: ((lg(q[0] + q[1] * x[late]) - v[late]) ** 2).mean()  # least squares off the opening (turns 0-1 pile up at x = 0)
            q = minimize(f, [0.2, 0.5], method='Nelder-Mead').x
            e = np.abs(lg(q[0] + q[1] * x) - v)
            res[f'{nm}, {form}'] = dict(a=float(q[0]), b=float(q[1]), mae_pts=float(100 * e.mean()), mae_pts_turn2plus=float(100 * e[late].mean()),
                                        spearman=float(spearmanr(x, v)[0]))
    lm, lo = z['t_won_me'], z['t_won_opp']
    tab = {(a, b): v[(lm == a) & (lo == b)].mean() for a in range(3) for b in range(3)}
    e = np.abs(np.array([tab[k] for k in zip(lm, lo)]) - v)
    res['lanes topped (3x3 table of mean V)'] = dict(mae_pts=float(100 * e.mean()), mae_pts_turn2plus=float(100 * e[late].mean()),
                                                     spearman=float(spearmanr(lm - lo, v)[0]))
    # the mental table: V by opponent count / own count
    cm, co = race_count(me, op), race_count(op, me)
    r = co / cm
    edges = [0, 0.25, 0.4, 0.5, 0.67, 0.8, 0.9, 1.11, 1.25, 1.5, 2, 2.5, 4, np.inf]
    table = []
    for lo_, hi in zip(edges[:-1], edges[1:]):
        m = (r >= lo_) & (r < hi) & late
        if m.sum() >= 100:
            table.append(dict(lo=lo_, hi=float(hi) if np.isfinite(hi) else None, mid=float(np.sqrt(max(lo_, 0.15) * min(hi, 6))), n=int(m.sum()), v=float(v[m].mean()), q10=float(np.quantile(v[m], .1)), q90=float(np.quantile(v[m], .9))))
    pred = np.interp(np.log(r), np.log([t['mid'] for t in table]), [t['v'] for t in table])
    res['ratio table'] = dict(rows=table, mae_pts=float(100 * np.abs(pred - v).mean()), mae_pts_turn2plus=float(100 * np.abs(pred - v)[late].mean()))
    res['turns_to_top'] = {int(c): sp.turns_to_top(c) for c in range(2, 13)}
    res['weights_reproduce_order'] = bool((np.argsort(STEP[2:8] * LEN[2:8]) == np.argsort(exact[2:8] * LEN[2:8])).all())
    return res, cm, co

def bailout_cells(z, min_n=200):
    # go/stop decisions and turns by opponent lanes x own lanes (opponent on 2: by steps left on their closest open column)
    I = np.arange(len(z['d_p']))
    tmp, bank = z['d_tmp'], z['d_bank']
    top = ((tmp == LEN) & (tmp > 0)).any(1)  # any column topped this turn; fresh: the pick just made topped one
    fresh = sp.fresh_top(tmp, z['d_game'] * 1000 + z['d_turn'])
    old = top & ~fresh
    co = race_count(z['d_prog'][I, 1 - z['d_p']], z['d_prog'][I, z['d_p']])
    J = np.arange(len(z['t_p']))
    tco = race_count(z['t_prog'][J, 1 - z['t_p']], z['t_prog'][J, z['t_p']])
    out_, steps = z['t_out'], z['t_banked_steps']
    def cell(dm, tm):
        d = dict(n_dec=int(dm.sum()), n_top=int((dm & fresh).sum()), n_no_top=int((dm & ~top).sum()), n_turns=int(tm.sum()),
                 p_bank_top=float(bank[dm & fresh].mean()) if (dm & fresh).sum() >= 30 else None,
                 p_bank_no_top=float(bank[dm & ~top].mean()),
                 p_bank_topped_earlier=float(bank[dm & old].mean()) if (dm & old).sum() >= 30 else None,
                 steps_banked=float(np.where(out_ == 'bank', steps, 0)[tm].mean()), bust=float((out_[tm] == 'bust').mean()),
                 win=float((out_[tm] == 'win').mean()))
        return d if dm.sum() >= min_n and tm.sum() >= 30 else None
    grid = {}
    for o in range(3):
        splits = [('all', None)] + ([(f'{a}-{b} steps to 3rd', (a, b)) for a, b in ((1, 1), (2, 2), (3, 3), (4, 99))] if o == 2 else [])
        for m in range(3):
            for nm, rg in splits:
                dm, tm = (z['d_won_opp'] == o) & (z['d_won_me'] == m), (z['t_won_opp'] == o) & (z['t_won_me'] == m)
                if rg:
                    dm &= (z['d_dist_opp'] >= rg[0]) & (z['d_dist_opp'] <= rg[1])
                    tm &= (z['t_dist_opp'] >= rg[0]) & (z['t_dist_opp'] <= rg[1])
                c = cell(dm, tm)
                if c:
                    grid[f'opp {o}, me {m}, {nm}'] = dict(opp=o, me=m, split=nm, **c)
    by_count = []
    for lo, hi in ((0, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 40), (40, 60), (60, 90), (90, 999)):
        dm, tm = (co >= lo) & (co < hi), (tco >= lo) & (tco < hi)
        c = cell(dm, tm)
        if c:
            by_count.append(dict(lo=lo, hi=hi, threat=float(z['d_threat'][dm].mean()), **c))
    return dict(grid=grid, by_opp_count=by_count)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('cmd', choices=['tops', 'summary'])
    ap.add_argument('--model', default='ens')
    ap.add_argument('--turns', type=int, default=100000)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    if a.cmd == 'tops':
        tops(a.model, a.turns, a.workers)
    else:
        z = dict(np.load(os.path.join(CACHE, f'selfplay_{a.model}.npz')))
        res = dict(proxy=proxy_fit(z)[0], bailout=bailout_cells(z))
        if os.path.exists(os.path.join(CACHE, f'tops_{a.model}.npz')):
            res['tops'] = tops_summary(a.model)
        if a.model != 'S' and os.path.exists(os.path.join(CACHE, 'tops_S.npz')):  # quick check on the single net
            S = tops_summary('S')
            res['tops_check_S'] = {k: S[k] for k in ('turns', 'offers_per_turn', 'declined_share', 'declined_share_real_choice',
                                                     'value_lost_by_topping_pts', 'after_declining', 'live_vs_topped')} | \
                dict(rules=[r for r in S['rules'] if r['gate'] in ('always', "opponent's count <= 25")])
        json.dump(res, open(os.path.join(OUT, 'followup.json'), 'w'), indent=1)
