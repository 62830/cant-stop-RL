import os, sys, json, time, argparse
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
from multiprocessing import Pool
import numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from branching import DELTA, RHO_MIN, SETS, legal, build_sets, resid_jobs, resid_by_set
from td_train import residual_job

# Step 0 of the branching spec: residual of existing nets on visited / end / close held-out sets (built by branching.py)
NETS = {'td30': os.path.join(ROOT, 'runs/td30/it0082.pt'), 'scratch30': os.path.join(ROOT, 'runs/scratch30/it0297.pt')}
WIDE60 = os.path.join(os.path.dirname(ROOT), 'cant_stop_wide/runs/wide60/it0100.pt')  # PR #10, not merged

def ci(x):
    return [float(np.nanpercentile(x, 2.5)), float(np.nanpercentile(x, 97.5))]

def residuals(pool, path, sets, n, workers):
    # abs residual |V - y| per (turn start, set), NaN where the row has no board; and signed means
    jobs, tag, R = resid_jobs(path, sets, None, 4 * workers)
    D, signed = np.full((n, 3), np.nan), []
    for k, (r, d) in resid_by_set(pool.map(residual_job, jobs), tag, R).items():
        D[r, SETS.index(k)] = np.abs(d)
        signed.append(float(d.mean()))
    return D, signed

def summarize(D, signed, B=5000, seed=0):
    # paired bootstrap over turn starts: visited/end/close of a resampled turn start are drawn together
    bs = np.stack([np.nanmean(D[np.random.default_rng(seed + b).integers(len(D), size=len(D))], 0) for b in range(B)])
    m = np.nanmean(D, 0)
    R = {s: dict(n=int((~np.isnan(D[:, j])).sum()), mean_abs=float(m[j]), ci=ci(bs[:, j]), signed=signed[j])
         for j, s in enumerate(SETS)}
    for j in (1, 2):
        R[SETS[j]].update(ratio=float(m[j] / m[0]), ratio_ci=ci(bs[:, j] / bs[:, 0]))
    return R

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--n', type=int, default=900, help='turn starts (= visited size; end/close are paired with them)')
    ap.add_argument('--games', type=int, default=100, help='V_h games the turn starts come from')
    ap.add_argument('--seed0', type=int, default=10000, help='first game seed; 10000+ is disjoint from every training '
                    'game. --seed0 0 --games 32 --n 200 is exactly td_train.heldout()')
    ap.add_argument('--wide60', default=WIDE60)
    ap.add_argument('--out', default=os.path.join(ROOT, 'runs', 'diag'))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    nets = dict(NETS, wide60=a.wide60)
    t0 = time.time()
    with Pool(a.workers) as pool:
        flat, sets = build_sets(pool, a.seed0, a.games, a.n)
        sizes = {s: len(sets[s][0]) for s in SETS}
        for s, k in sizes.items():
            assert all(legal(b) for b in sets[s][0])
            if k < 0.95 * a.n:
                print(f'WARNING: {s} set has {k} boards (< 95% of {a.n})', flush=True)
        print(f'sets built in {time.time() - t0:.0f}s: {sizes}', flush=True)
        gaps = [x for r in flat for x in r['gaps']]
        res = dict(seed0=a.seed0, games=a.games, delta=DELTA, rho_min=RHO_MIN, sizes=sizes,
                   close_calls=dict(turns=len(flat), per_turn=float(np.mean([r['n_close'] for r in flat])),
                                    frac_turns_ge1=float(np.mean([r['n_close'] > 0 for r in flat])),
                                    expected_per_turn=float(np.mean([r['exp_close'] for r in flat])),
                                    mean_gap=float(np.mean(gaps)), win_mass=float(np.mean([r['win_mass'] for r in flat])),
                                    bust_mass=float(np.mean([r['bust'] for r in flat]))), nets={})
        for name, path in nets.items():
            if not os.path.exists(path):
                print(f'WARNING: skipping {name}, {path} not found', flush=True)
                continue
            t1 = time.time()
            R = summarize(*residuals(pool, path, sets, sizes['visited'], a.workers))
            res['nets'][name] = dict(path=os.path.relpath(path, ROOT), seconds=round(time.time() - t1, 1), **R)
            print(name, json.dumps(R), flush=True)
    res['proceed'] = {k: max(v['end']['ratio'], v['close']['ratio']) >= 1.5 for k, v in res['nets'].items()}
    res['seconds'] = round(time.time() - t0, 1)
    json.dump(res, open(os.path.join(a.out, 'results.json'), 'w'), indent=1)
    print(json.dumps(res['close_calls']), res['proceed'], f"{res['seconds']}s")

if __name__ == '__main__':
    main()
