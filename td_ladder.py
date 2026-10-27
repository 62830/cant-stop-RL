import os, json, glob, time, argparse
from functools import partial
from baselines import V2, V2Tuned
from eval import evaluate
from solver_agent import SolverAgent
from td_train import BEST_VH, log
from value_net import NetV

# ladder over saved snapshots (separate process so training never blocks); v2 is reported only, never used to select

def ckpts(run):
    return sorted(glob.glob(os.path.join('runs', run, 'ckpt', 'it*.pt')))

def it_of(path):
    return int(os.path.basename(path)[2:6])

def agent(path):
    return partial(SolverAgent, NetV.load(path)) if path != 'vh' else partial(SolverAgent, BEST_VH)

def match(mf, it, name, a, b, n, workers, seed):
    t = time.time()
    p, (lo, hi) = evaluate(a, b, n, workers, seed)
    return json.loads(log(mf, kind='ladder', iter=it, opp=name, p=p, lo=lo, hi=hi, n=n, wall_s=round(time.time() - t, 1)))

def watch(a, mf):
    prev, k = None, 0
    while True:
        cs = ckpts(a.run)
        last = cs[-1] if cs else None
        if last and (prev is None or it_of(last) - it_of(prev) >= a.min_gap):
            it, me = it_of(last), agent(last)
            if prev is not None:
                match(mf, it, 'prev', me, agent(prev), a.n, a.workers, it)
            match(mf, it, 'vh', me, agent('vh'), a.n, a.workers, it + 1)
            if k % 2 == 0:
                match(mf, it, 'v2tuned', me, V2Tuned, a.n, a.workers, it + 2)
                match(mf, it, 'v2', me, V2, a.n, a.workers, it + 3)
            prev, k = last, k + 1
        if os.path.exists(os.path.join('runs', a.run, 'DONE')) and prev == last:
            return
        time.sleep(30)

def gate(a, mf):
    lad = [json.loads(l) for l in open(mf)]
    lad = [r for r in lad if r['kind'] == 'ladder' and r['opp'] == 'vh' and r['iter'] > 0]
    it = a.it if a.it is not None else max(lad, key=lambda r: (r['p'], r['iter']))['iter']
    me = agent(os.path.join('runs', a.run, 'ckpt', f'it{it:04d}.pt'))
    res = {'snapshot': it, 'selected_by': 'ladder vs best V_h' if a.it is None else 'manual'}
    for name, b in (('vh', agent('vh')), ('v2tuned', V2Tuned), ('v2', V2)):
        r = match(mf, it, 'gate_' + name, me, b, a.n, a.workers, 777)
        res[name] = {k: r[k] for k in ('p', 'lo', 'hi', 'n', 'wall_s')}
        print(name, res[name], flush=True)
    path = os.path.join('runs', a.run, 'results.json')
    old = json.load(open(path)) if os.path.exists(path) else {}
    json.dump({**old, 'gate': res}, open(path, 'w'), indent=1)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='phase4')
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--min_gap', type=int, default=10)
    ap.add_argument('--gate', action='store_true')
    ap.add_argument('--it', type=int)
    a = ap.parse_args()
    mf = os.path.join('runs', a.run, 'metrics.jsonl')
    (gate if a.gate else watch)(a, mf)
