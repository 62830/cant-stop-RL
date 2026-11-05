# pilot of luck-adjusted paired eval (eval.evaluate_paired ref=BEST_VH): S vs V_h, A/A V_h vs V_h, raw vs adjusted timing.
# writes runs/luck/results.json and runs/luck/pairs.npz; plots: analysis/plot_luck.py
import os, sys, json, time, argparse
from functools import partial
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from eval import evaluate_paired
from solver_agent import SolverAgent
from td_train import BEST_VH
from value_net import NetV

OUT = os.path.join(ROOT, 'runs', 'luck')
Z = 1.96 + 0.84  # alpha 0.05 two-sided, 80% power

def n_needed(sd, mde):
    return (Z * sd / mde) ** 2

def ci(x):
    m, h = float(x.mean()), 1.96 * float(x.std(ddof=1)) / np.sqrt(len(x))
    return [m, [m - h, m + h]]

def boot(raw, adj, luck, B=4000, seed=0):
    # percentile CIs over pairs: SD raw, SD adj, variance ratio raw/adj, corr(raw pair score, pair mean sum L)
    idx = np.random.default_rng(seed).integers(0, len(raw), (B, len(raw)))
    r, a, l = raw[idx], adj[idx], luck[idx]
    sr, sa = r.std(1, ddof=1), a.std(1, ddof=1)
    c = np.array([np.corrcoef(x, y)[0, 1] for x, y in zip(r, l)])
    pc = lambda x: [float(q) for q in np.percentile(x, [2.5, 97.5])]
    return dict(sd_raw=pc(sr), sd_adj=pc(sa), var_ratio=pc(sr ** 2 / sa ** 2), corr=pc(c))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--s_ckpt', default=os.path.join(ROOT, '..', 'cant_stop_archive', 'cant_stop_scratch', 'runs', 'scratch30', 'ckpt', 'it0297.pt'))
    ap.add_argument('--n', type=int, default=1000)
    ap.add_argument('--n_aa', type=int, default=300)
    ap.add_argument('--n_time', type=int, default=240)
    ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    mS, mV = partial(SolverAgent, NetV.load(a.s_ckpt).fused()), partial(SolverAgent, BEST_VH)
    res, arrs = dict(s_ckpt=os.path.relpath(a.s_ckpt, ROOT), ref=repr(BEST_VH), workers=a.workers), {}
    for name, mb, n, seed in (('main', mS, a.n, 50000), ('aa', mV, a.n_aa, 50001)):
        t = time.time()
        d = evaluate_paired(mb, mV, n, a.workers, seed, ref=BEST_VH, detail=True)
        d['seconds'] = time.time() - t
        res_, L, ev = d.pop('result'), d.pop('luck_sum'), d.pop('n_events')
        raw, adj = res_.mean(1), (res_ - L).mean(1)
        arrs.update({f'{name}_result': res_, f'{name}_luck': L, f'{name}_events': ev})
        d['corr'] = float(np.corrcoef(raw, L.mean(1))[0, 1])
        d['corr_game'] = float(np.corrcoef(res_.ravel(), L.ravel())[0, 1])  # result vs sum L, per game
        d['luck_by_seat'] = [[float(L[:, j].mean()), float(L[:, j].std(ddof=1) / np.sqrt(n))] for j in (0, 1)]  # A first, B first
        d['events_sd'] = float(ev.std(ddof=1))
        if name == 'aa':
            # deterministic self-play mirrors each pair exactly (raw = adj = 0.5); pair A-first games of the first half of
            # the seeds with B-first games of the second half (independent dice) for a non-trivial A/A test of the adjusted mean
            h = n // 2
            rr, ll = np.c_[res_[:h, 0], res_[h:2 * h, 1]], np.c_[L[:h, 0], L[h:2 * h, 1]]
            raw, adj = rr.mean(1), (rr - ll).mean(1)
            d['unpaired'] = dict(raw=ci(raw), adj=ci(adj), sd_raw=float(raw.std(ddof=1)), sd_adj=float(adj.std(ddof=1)))
            d['first_mover'] = dict(raw=ci(res_[:, 0]), adj=ci(res_[:, 0] - L[:, 0]))  # P(first mover wins), V_h self-play
        else:
            d['boot'] = boot(raw, adj, L.mean(1))
        res[name] = d
        print(name, json.dumps(d), flush=True)
    t = {}
    for ref in (BEST_VH, None):  # same seeds, raw vs adjusted
        t0 = time.time()
        evaluate_paired(mS, mV, a.n_time, a.workers, 50002, ref=ref)
        t['adj' if ref is not None else 'raw'] = (time.time() - t0) / a.n_time
    t['overhead'] = t['adj'] / t['raw'] - 1
    res['sec_per_pair_8w'] = t
    m, b = res['main'], res['main']['boot']
    res['n_needed'] = {f'{mde}': {k: dict(n=n_needed(m['sd_' + k], mde / 100), ci=[n_needed(x, mde / 100) for x in b['sd_' + k]],
                                          hours=n_needed(m['sd_' + k], mde / 100) * t[k] / 3600) for k in ('raw', 'adj')}
                       for mde in (0.5, 1.0)}
    print(json.dumps({k: res[k] for k in ('sec_per_pair_8w', 'n_needed')}, indent=1))
    json.dump(res, open(os.path.join(OUT, 'results.json'), 'w'), indent=1)
    np.savez(os.path.join(OUT, 'pairs.npz'), **arrs)

if __name__ == '__main__':
    main()
