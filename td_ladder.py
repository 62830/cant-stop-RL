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
    return partial(SolverAgent, NetV.load(path).fused()) if path != 'vh' else partial(SolverAgent, BEST_VH)

def match(mf, it, name, a, b, n, workers, seed):
    t = time.time()
    p, (lo, hi) = evaluate(a, b, n, workers, seed)
    return json.loads(log(mf, kind='ladder', iter=it, opp=name, p=p, lo=lo, hi=hi, n=n, wall_s=round(time.time() - t, 1)))

def trainer_alive(run):
    # td_train writes runs/<run>/train.pid and removes it on a clean exit
    try:
        os.kill(int(open(os.path.join('runs', run, 'train.pid')).read()), 0)
        return True
    except (OSError, ValueError):
        return False

def watch(a, mf):
    # evaluates a snapshot once it is >= min_gap iterations past the previous one; once training is done (DONE) or has
    # died (no live pid for `patience` polls) it evaluates the final snapshot regardless of the gap, then exits
    prev, k, dead = None, 0, 0
    while True:
        done = os.path.exists(os.path.join('runs', a.run, 'DONE'))
        dead = 0 if trainer_alive(a.run) else dead + 1
        finished = done or dead >= a.patience
        cs = ckpts(a.run)
        last = cs[-1] if cs else None
        if last and last != prev and (finished or prev is None or it_of(last) - it_of(prev) >= a.min_gap):
            it, me = it_of(last), agent(last)
            if prev is not None:
                match(mf, it, 'prev', me, agent(prev), a.n, a.workers, it)
            match(mf, it, 'vh', me, agent('vh'), a.n, a.workers, it + 1)
            if k % 2 == 0 or finished:
                match(mf, it, 'v2tuned', me, V2Tuned, a.n, a.workers, it + 2)
                match(mf, it, 'v2', me, V2, a.n, a.workers, it + 3)
            prev, k = last, k + 1
            continue
        if finished:
            if not done:
                print('trainer is gone without DONE; exiting', flush=True)
            return
        time.sleep(30)

def gate(a, mf):
    # gates the last snapshot; with --also_best also the ladder-best one (argmax of noisy ladder points vs V_h)
    cs = ckpts(a.run)
    if not cs:
        raise SystemExit('no checkpoints')
    its = {'last': it_of(cs[-1]) if a.it is None else a.it}
    lad = [json.loads(l) for l in open(mf)] if os.path.exists(mf) else []
    lad = [r for r in lad if r['kind'] == 'ladder' and r['opp'] == 'vh' and r['iter'] > 0]
    if a.also_best and lad:
        its['ladder_best'] = max(lad, key=lambda r: (r['p'], r['iter']))['iter']
    out = {}
    for role, it in its.items():
        if role != 'last' and it == its['last']:
            out[role] = {'snapshot': it, 'same_as': 'last'}
            continue
        me = agent(os.path.join('runs', a.run, 'ckpt', f'it{it:04d}.pt'))
        res = {'snapshot': it}
        for name, b in (('vh', agent('vh')), ('v2tuned', V2Tuned), ('v2', V2)):
            r = match(mf, it, 'gate_' + name, me, b, a.n, a.workers, 777)
            res[name] = {k: r[k] for k in ('p', 'lo', 'hi', 'n', 'wall_s')}
            print(role, it, name, res[name], flush=True)
        out[role] = res
    path = os.path.join('runs', a.run, 'results.json')
    old = json.load(open(path)) if os.path.exists(path) else {}
    json.dump({**old, a.key: out}, open(path, 'w'), indent=1)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='phase4')
    ap.add_argument('--n', type=int, default=200)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--min_gap', type=int, default=10)
    ap.add_argument('--patience', type=int, default=6)
    ap.add_argument('--gate', action='store_true')
    ap.add_argument('--also_best', action='store_true')
    ap.add_argument('--key', default='gate_v2')
    ap.add_argument('--it', type=int)
    a = ap.parse_args()
    mf = os.path.join('runs', a.run, 'metrics.jsonl')
    (gate if a.gate else watch)(a, mf)
