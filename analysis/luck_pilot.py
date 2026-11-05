# pilot of luck-adjusted paired eval (eval.evaluate_paired_detail ref=BEST_VH): S vs V_h, A/A V_h vs V_h (300 pairs and a
# 600-pair replicate), raw vs adjusted timing. Writes runs/luck/results.json and runs/luck/pairs.npz; plots: plot_luck.py.
# --from_npz recomputes every statistic from pairs.npz (keeping the saved timings) instead of replaying games
import os, sys, json, time, argparse
from functools import partial
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from eval import evaluate_paired, evaluate_paired_detail, paired_stats, n_needed, _ci, _sd
from solver_agent import SolverAgent
from td_train import BEST_VH

OUT = os.path.join(ROOT, 'runs', 'luck')
STAGES = (('main', 50000), ('aa', 50001), ('aa2', 50003))  # name, seed base

def boot(raw, adj, luck, B=4000, seed=0):
    # percentile CIs over pairs: SD raw, SD adj, variance ratio raw/adj, corr(raw pair score, pair mean sum L)
    idx = np.random.default_rng(seed).integers(0, len(raw), (B, len(raw)))
    r, a, l = raw[idx], adj[idx], luck[idx]
    sr, sa = r.std(1, ddof=1), a.std(1, ddof=1)
    c = np.array([np.corrcoef(x, y)[0, 1] for x, y in zip(r, l)])
    pc = lambda x: [float(q) for q in np.percentile(x, [2.5, 97.5])]
    return dict(sd_raw=pc(sr), sd_adj=pc(sa), var_ratio=pc(sr ** 2 / sa ** 2), corr=pc(c))

def stats(name, y, L, ev):
    d = paired_stats(y, L, ev)
    raw, adj = y.mean(1), (y - L).mean(1)
    d['corr_game'] = float(np.corrcoef(y.ravel(), L.ravel())[0, 1])  # result vs sum L, per game
    d['events_sd'] = _sd(ev.ravel())
    if name == 'main':
        d['corr'] = float(np.corrcoef(raw, L.mean(1))[0, 1])
        d['boot'] = boot(raw, adj, L.mean(1))
    else:
        # deterministic self-play mirrors each pair exactly (raw = adj = 0.5, pair-mean luck 0): pair the A-first games of
        # the first half of the seeds with the B-first games of the second half (independent dice) for a non-trivial A/A
        h = len(y) // 2
        rr, ll = np.c_[y[:h, 0], y[h:2 * h, 1]], np.c_[L[:h, 0], L[h:2 * h, 1]]
        raw, adj = rr.mean(1), (rr - ll).mean(1)
        d['unpaired'] = dict(n=h, raw=_ci(raw), adj=_ci(adj), sd_raw=_sd(raw), sd_adj=_sd(adj))
        d['first_mover'] = dict(raw=_ci(y[:, 0]), adj=_ci(y[:, 0] - L[:, 0]))  # P(first mover wins), V_h self-play
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--s_ckpt', default=os.path.join(ROOT, '..', 'cant_stop_archive', 'cant_stop_scratch', 'runs', 'scratch30', 'ckpt', 'it0297.pt'))
    ap.add_argument('--n', type=int, nargs=3, default=[1000, 300, 600], help='pairs: main, aa, aa2')
    ap.add_argument('--n_time', type=int, default=240)
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--from_npz', action='store_true')
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rpath, npath = os.path.join(OUT, 'results.json'), os.path.join(OUT, 'pairs.npz')
    old = json.load(open(rpath)) if a.from_npz else {}
    arrs = dict(np.load(npath)) if a.from_npz else {}
    res = dict(s_ckpt=os.path.relpath(a.s_ckpt, ROOT), ref=repr(BEST_VH), workers=a.workers)
    if not a.from_npz:
        from value_net import NetV
        mS, mV = partial(SolverAgent, NetV.load(a.s_ckpt).fused()), partial(SolverAgent, BEST_VH)
    for (name, seed), n in zip(STAGES, a.n):
        if not a.from_npz:
            t = time.time()
            d = evaluate_paired_detail(mS if name == 'main' else mV, mV, n, a.workers, seed, ref=BEST_VH)
            arrs.update({f'{name}_result': d['result'], f'{name}_luck': d['luck_sum'], f'{name}_events': d['n_events']})
            secs = time.time() - t
        else:
            secs = old.get(name, {}).get('seconds')
        res[name] = dict(stats(name, arrs[f'{name}_result'], arrs[f'{name}_luck'], arrs[f'{name}_events']), seed=seed, seconds=secs)
        print(name, json.dumps(res[name]), flush=True)
    if a.from_npz:
        t = old['sec_per_pair_8w']
    else:
        t = {}
        for ref in (BEST_VH, None):  # same seeds, raw vs adjusted
            t0 = time.time()
            evaluate_paired(mS, mV, a.n_time, a.workers, 50002, ref)
            t['adj' if ref is not None else 'raw'] = (time.time() - t0) / a.n_time
        t['overhead'] = t['adj'] / t['raw'] - 1
    res['sec_per_pair_8w'] = t
    m, b = res['main'], res['main']['boot']
    res['n_needed'] = {f'{mde}': {k: dict(n=n_needed(m['sd_' + k], mde / 100), ci=[n_needed(x, mde / 100) for x in b['sd_' + k]],
                                          hours=n_needed(m['sd_' + k], mde / 100) * t[k] / 3600) for k in ('raw', 'adj')}
                       for mde in (0.5, 1.0)}
    print(json.dumps({k: res[k] for k in ('sec_per_pair_8w', 'n_needed')}, indent=1))
    json.dump(res, open(rpath, 'w'), indent=1, allow_nan=False)
    np.savez(npath, **arrs)

if __name__ == '__main__':
    main()
