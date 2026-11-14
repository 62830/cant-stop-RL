# Does an ensemble of 3 from-scratch TD nets (M1..M3 = ens_s2, ens_s4, ens_s5; mean logit) beat single nets? Luck-adjusted paired gates
# (eval.evaluate_paired_detail, ref=BEST_VH) and forward-pass distances on P_H. Steps: rule | dist | gate NAME | verdict.
# Writes runs/ensemble/results.json (+ gate_NAME.npz per-game arrays); plots: plot_ensemble.py
import os, sys, json, glob, math, time, argparse, itertools
from functools import partial
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from eval import evaluate_paired_detail, n_needed
from solver_agent import SolverAgent
from td_train import BEST_VH
from value_net import NetV, Ensemble

OUT = os.path.join(ROOT, 'runs', 'ensemble')
RES = os.path.join(OUT, 'results.json')
S_CKPT = os.path.join(ROOT, '..', 'cant_stop_archive', 'cant_stop_scratch', 'runs', 'scratch30', 'ckpt', 'it0297.pt')
RUNS = ('ens_s2', 'ens_s4', 'ens_s5')  # M1..M3; see TRAINING
# name: (A, B, default pairs, seed); A's score, ref = V_h for the luck adjustment
GATES = dict(E3_vs_H=('E3', 'H', 2300, 60000), M1_vs_H=('M1', 'H', 1200, 60001), M2_vs_H=('M2', 'H', 1200, 60002),
             M3_vs_H=('M3', 'H', 1200, 60003), E3_vs_S_pilot=('E3', 'S', 300, 60004), E3_vs_S=('E3', 'S', None, 60005),
             E4_vs_H=('E4', 'H', 1200, 60006))
RULE = ('"Ensembling helps" iff (1) E3 - mean(M1..M3), each scored vs V_h (adjusted pair score), > 0 with a 95% CI excluding 0 '
        '(SE = sqrt(se_E3^2 + sum se_Mi^2 / 9), from the independent gates); and (2) E3 vs S > 0.5 with a 95% CI excluding 0.5 '
        '(pilot + main pooled; the pilot only sizes the main gate by its SD). Otherwise: "training exhausted at this capacity".')

TRAINING = dict(
    collapsed=['ens_s1', 'ens_s3', 'ens_s4 (first)', 'ens_s5 (first)'],
    diagnosis='flat-V trap: a near-flat random init fits to V ~ 0.511 everywhere at the first update, so stopping (~0.49) never '
              'beats going on, the solver never banks, every turn start is the empty board and every target 0.5112 (zero spread, '
              '69,038 solver states per solve); v_std -> 0 and loss flat ~0.6928 by it 2-4. ens_s2 escaped by luck',
    rate='4 of 6 from-scratch seeds (counting scratch30, which escaped) collapsed without the warmup',
    fix='ens_s4, ens_s5 relaunched with a forced-stop warmup (--stop_eps 0.1 --stop_eps_iters 15: the behaviour policy stops '
        'instead of a chosen go w.p. eps, linear to 0 at it 15; targets stay the greedy solve). ens_s2 has no warmup',
    budgets='ens_s2 266 min from 16:41; ens_s4/s5 213.6 min from 17:33 (all end ~21:07); LR 1e-3 -> 1e-4 in place at '
            '0.67 of each own budget')

def final(run):
    return sorted(glob.glob(os.path.join(ROOT, 'runs', run, 'ckpt', 'it*.pt')))[-1]

def value_fns():
    M = [NetV.load(final(r)).fused() for r in RUNS]
    S = NetV.load(S_CKPT).fused()
    return dict(M1=M[0], M2=M[1], M3=M[2], E3=Ensemble(M), E4=Ensemble(M + [S]), S=S, H=BEST_VH)

def load():
    return json.load(open(RES)) if os.path.exists(RES) else {}

def save(res):
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(RES + '.tmp', 'w'), indent=1)
    os.replace(RES + '.tmp', RES)

def se(ci):
    return (ci[1] - ci[0]) / (2 * 1.96)

def gate(name, n, workers, force=False):
    A, B, n0, seed = GATES[name]
    n = n or n0
    if not n:
        raise SystemExit(f'gate {name}: pass --n (sized from the pilot: step size)')
    if name in load().get('gates', {}) and not force:
        raise SystemExit(f'gate {name} already in {RES}; pass --force to overwrite it')
    v = value_fns()
    t = time.time()
    d = evaluate_paired_detail(partial(SolverAgent, v[A]), partial(SolverAgent, v[B]), n, workers, seed, ref=BEST_VH)
    secs = time.time() - t
    np.savez_compressed(os.path.join(OUT, f'gate_{name}.npz'), result=d['result'], luck_sum=d['luck_sum'], n_events=d['n_events'])
    out = {k: x for k, x in d.items() if k not in ('result', 'luck_sum', 'n_events')}
    res = load()
    res.setdefault('gates', {})[name] = dict(out, A=A, B=B, seed=seed, workers=workers, seconds=round(secs, 1), sec_per_pair=secs / n)
    save(res)
    print(name, json.dumps(res['gates'][name]), flush=True)

def dist():
    # mean |V - V_S| and pairwise mean |dV| of raw values on P_H (900 turn starts of V_h games, branch_diag visited set)
    d = np.load(os.path.join(ROOT, 'runs', 'warm_bias', 'data.npz'))
    sel = d['P_H__sel']
    prog, p = d['P_H__prog'][sel], d['P_H__p'][sel]
    v = value_fns()
    V = {}
    for k, f in v.items():
        V[k] = np.empty(len(p))
        for q in (0, 1):
            V[k][p == q] = f(prog[p == q], q)
    m = lambda x: dict(mean=float(x.mean()), se=float(x.std(ddof=1) / math.sqrt(len(x))))
    res = load()
    res['dist'] = dict(n=len(p), to_S={k: m(np.abs(V[k] - V['S'])) for k in ('M1', 'M2', 'M3', 'E3', 'E4', 'H')},
                       pairwise={f'{a}-{b}': m(np.abs(V[a] - V[b])) for a, b in itertools.combinations(('M1', 'M2', 'M3'), 2)},
                       E3_to_members={k: m(np.abs(V['E3'] - V[k])) for k in ('M1', 'M2', 'M3')},
                       sd_V_S=float(V['S'].std()))
    save(res)
    print(json.dumps(res['dist'], indent=1))

def pool(*ds):
    # pooled stats of gates with the same A, B (per-pair adjusted scores concatenated)
    adj = np.concatenate([(d['result'] - d['luck_sum']).mean(1) for d in ds])
    m, s = float(adj.mean()), float(adj.std(ddof=1))
    h = 1.96 * s / math.sqrt(len(adj))
    return dict(n=len(adj), adj=[m, [m - h, m + h]], sd_adj=s)

def verdict():
    res = load()
    G = res['gates']
    v = {}
    if all(k in G for k in ('E3_vs_H', 'M1_vs_H', 'M2_vs_H', 'M3_vs_H')):
        Ms = [G[f'M{i}_vs_H']['adj'] for i in (1, 2, 3)]
        mM = float(np.mean([x[0] for x in Ms]))
        diff = G['E3_vs_H']['adj'][0] - mM
        s = math.sqrt(se(G['E3_vs_H']['adj'][1]) ** 2 + sum(se(x[1]) ** 2 for x in Ms) / 9)
        v['mean_M_vs_H'] = mM
        v['E3_minus_meanM'] = dict(est=diff, ci=[diff - 1.96 * s, diff + 1.96 * s], se=s, pts=100 * diff,
                                   ci_pts=[100 * (diff - 1.96 * s), 100 * (diff + 1.96 * s)])
        v['c1'] = bool(diff - 1.96 * s > 0)
    ks = [k for k in ('E3_vs_S_pilot', 'E3_vs_S') if k in G]
    if ks:
        v['E3_vs_S'] = pool(*[np.load(os.path.join(OUT, f'gate_{k}.npz')) for k in ks])
        v['E3_vs_S']['gates'] = ks
        v['c2'] = bool(v['E3_vs_S']['adj'][1][0] > 0.5)
    if 'E4_vs_H' in G:
        v['extra_E4_vs_H'] = dict(adj=G['E4_vs_H']['adj'], n=G['E4_vs_H']['n'], note='not part of the rule')
    v['verdict'] = 'ensembling helps' if v.get('c1') and v.get('c2') else 'training exhausted at this capacity'
    res['decision'] = v
    save(res)
    print(json.dumps(v, indent=1))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('step', choices=('rule', 'dist', 'gate', 'verdict', 'size'))
    ap.add_argument('name', nargs='?')
    ap.add_argument('--n', type=int)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--force', action='store_true', help='gate: overwrite an existing result')
    ap.add_argument('--minutes', type=float, default=30, help='size: time left for the E3 vs S main gate')
    a = ap.parse_args()
    if a.step == 'rule':
        res = load()
        res.update(rule=RULE, ref=repr(BEST_VH), training=TRAINING, members={r: os.path.relpath(final(r), ROOT) for r in RUNS},
                   S=os.path.relpath(S_CKPT, ROOT), gates_plan={k: dict(A=g[0], B=g[1], n=g[2], seed=g[3]) for k, g in GATES.items()},
                   ensemble='sigmoid(mean logit)', S_vs_H_ref=dict(source='PR #15 runs/luck main', adj=0.5176, pm=0.0053))
        save(res)
    elif a.step == 'dist':
        dist()
    elif a.step == 'gate':
        gate(a.name, a.n, a.workers, a.force)
    elif a.step == 'size':
        # main E3 vs S pairs that fit in --minutes at the pilot's pace, and the edge they resolve
        g = load()['gates']['E3_vs_S_pilot']
        n = int(a.minutes * 60 / g['sec_per_pair'])
        print(dict(n=n, sd_adj=g['sd_adj'], mde_at_n=2.8 * g['sd_adj'] / math.sqrt(n), n_for_1pt=n_needed(g['sd_adj'], 0.01)))
    else:
        verdict()

if __name__ == '__main__':
    main()
