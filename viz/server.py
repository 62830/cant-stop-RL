import os, sys, time, random, argparse, threading
from itertools import product
import numpy as np
from flask import Flask, request, jsonify, send_file, abort
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import fastrules as fr
from fastrules import LEN
from value_net import NetV, Ensemble
from solver_agent import SolverAgent

# stateless JSON API; a state is {prog: [[13], [13]], p, tmp: [13], dice: [4] | null, wait_for_go, winner?}
# probabilities are P(p wins) for the mover p, and *_p0 = P(player 0 wins)
app = Flask(__name__)
agent, info, lock = None, None, threading.Lock()

def init(ckpts):
    # one checkpoint path, or several for an Ensemble (sigmoid of the members' mean logit)
    global agent, info
    ckpts = [ckpts] if isinstance(ckpts, str) else list(ckpts)
    agent = SolverAgent(NetV.load(ckpts[0]).fused() if len(ckpts) == 1 else Ensemble(ckpts))
    info = dict(model='single' if len(ckpts) == 1 else 'ensemble', n=len(ckpts), ckpts=ckpts)
    t = time.time()
    agent.solution(np.zeros((2, 13), np.int8), 0, np.zeros(13, np.int8))  # warm numba + the solver
    print(f'warm-up solve {time.time() - t:.1f}s')

class Bad(Exception):
    pass

def ints(x):
    # nested JSON int lists -> int8 array (no bools, floats or out-of-range values)
    a = np.array(x)
    if a.dtype.kind != 'i' or (np.abs(a) > 100).any():
        raise Bad('expected small integers')
    return a.astype(np.int8)

def check_dice(d):
    if d is not None and not (isinstance(d, list) and len(d) == 4 and all(type(x) is int and 1 <= x <= 6 for x in d)):
        raise Bad('dice: 4 integers in 1..6 (or null)')
    return d

def parse(s):
    # JSON state -> (prog, p, tmp, dice, wait_for_go); Bad(reason) if impossible
    if not isinstance(s, dict):
        raise Bad('state must be an object')
    try:
        prog, tmp = ints(s['prog']), ints(s.get('tmp') or [0] * 13)
        p, dice, wfg = s['p'], check_dice(s.get('dice')), bool(s.get('wait_for_go'))
    except (KeyError, TypeError, ValueError) as e:
        raise Bad(f'malformed state: {e}')
    if prog.shape != (2, 13) or tmp.shape != (13,) or type(p) is not int or p not in (0, 1):
        raise Bad('shape: prog [2][13], tmp [13], p in {0, 1}')
    if prog[:, :2].any() or tmp[:2].any():
        raise Bad('columns 0 and 1 do not exist (must be 0)')
    if (prog < 0).any() or (prog > LEN).any():
        raise Bad('progress outside 0..LEN')
    for c in range(2, 13):
        for q in (0, 1):
            if prog[q, c] == LEN[c] and prog[1 - q, c]:
                raise Bad(f'column {c} is claimed by P{q} but P{1 - q} has progress there')
    w, sc = s.get('winner'), [fr.score(prog, q) for q in (0, 1)]
    if w is None and max(sc) >= 3:
        raise Bad('a player already has 3 columns')
    if w is not None and (w not in (0, 1) or type(w) is not int or sc[w] < 3 or sc[1 - w] >= 3):
        raise Bad('winner must be the only player with 3 columns')
    if fr.n_markers(tmp) > 3:
        raise Bad('more than 3 temp markers')
    for c in range(2, 13):
        if tmp[c] and fr.blocked(prog, c):
            raise Bad(f'temp marker on blocked column {c}')
        if tmp[c] and not prog[p, c] < tmp[c] <= LEN[c]:
            raise Bad(f'temp marker on column {c} must be above P{p} progress and <= {LEN[c]}')
    if dice is not None and wfg:
        raise Bad('dice are rolled only after go')
    if wfg and not tmp.any():
        raise Bad('go/stop phase needs temp markers')
    if not wfg and dice is None and tmp.any():
        raise Bad('temp markers need dice to pick from, or the go/stop phase')
    return prog, p, tmp, dice, wfg

def state(prog, p, tmp=None, dice=None, wfg=False, **kw):
    return dict(prog=prog.tolist(), p=int(p), tmp=(np.zeros(13, int) if tmp is None else tmp).tolist(), dice=dice,
                wait_for_go=wfg, **kw)

num = lambda x: None if x is None or not np.isfinite(x) else float(x)
p0 = lambda v, p: None if v is None else (v if p == 0 else 1 - v)

def picks(prog, p, tmp, dice):
    # [((a, b), new tmp)] in fastrules order
    mv = np.empty((fr.MAX_PICKS, 2), np.int8)
    n = fr.roll_moves(fr.legal_moves(prog, p, tmp), fr.roll_index(dice), mv)
    return [((int(a), int(b)), fr.apply_move(prog, p, tmp, a, b)) for a, b in mv[:n]]

def won_go(prog, p, tmp, bv):
    # (go, pbust) when banking tmp already wins: going on keeps the win unless the roll busts to bv
    pb = float(sum(fr.ROLL_PROB[r] for r in range(fr.N_ROLLS) if fr.bust(prog, p, tmp, r)))
    return 1 - pb * (1 - bv), pb

def evaluate(prog, p, tmp, dice, wfg):
    v_net = float(agent.value_fn(prog[None], p)[0])
    out = dict(p=p, v_net=v_net)
    if fr.stop_wins(prog, p, tmp):
        sol, bv = None, 1 - float(agent.value_fn(prog[None], 1 - p)[0])
        go, pb = won_go(prog, p, tmp, bv)
        out.update(v_solved=1.0, stop=1.0, go=go, pbust=pb, bust_val=bv)
    else:
        sol = agent.solution(prog, p, tmp)
        i = sol.row(tmp)
        out.update(v_solved=num(sol.best[i]), stop=num(sol.stop[i]), go=num(sol.go[i]), pbust=num(sol.pbust[i]),
                   bust_val=num(sol.bust_val))
    if wfg:
        out['decision'] = 'go' if out['go'] is not None and out['go'] > out['stop'] else 'stop'
    if dice is not None:
        win = lambda t: fr.stop_wins(prog, p, t)
        ps = [dict(move=m, tmp=t.tolist(), value=1.0 if sol is None or win(t) else float(sol.value(t)))
              for m, t in picks(prog, p, tmp, dice)]
        ps.sort(key=lambda d: -d['value'])  # stable: ties keep fastrules order
        out.update(picks=ps, bust=not ps, best=0 if ps else None)
        out['v_roll'] = ps[0]['value'] if ps else out['bust_val']
    for k in ('v_net', 'v_solved', 'stop', 'go', 'bust_val', 'v_roll'):
        if k in out:
            out[k + '_p0'] = p0(out[k], p)
    for d in out.get('picks', []):
        d['value_p0'] = p0(d['value'], p)
    return out

def roll(dice=None):
    return list(dice) if dice else [random.randint(1, 6) for _ in range(4)]

def new_turn(prog, p, dice, events):
    # p's turn starts with a roll; a roll with no legal pick busts straight to the other player
    while True:
        dice = roll(dice)
        if picks(prog, p, np.zeros(13, np.int8), dice):
            return state(prog, p, dice=dice)
        events.append(dict(type='bust', p=p, dice=dice))
        p, dice = 1 - p, None

def step(prog, p, tmp, dice, wfg, action, next_dice=None):
    events = []
    if wfg:
        if action == 'go':
            d = roll(next_dice)
            if picks(prog, p, tmp, d):
                return state(prog, p, tmp, d), events
            events.append(dict(type='bust', p=p, dice=d))
            return new_turn(prog, 1 - p, None, events), events
        if action != 'stop':
            raise Bad('expected go or stop')
        new = fr.apply_stop(prog, p, tmp)
        if fr.score(new, p) >= 3:
            events.append(dict(type='win', p=p))
            return state(new, p, winner=p), events
        events.append(dict(type='banked', p=p, tmp=tmp.tolist()))
        return new_turn(new, 1 - p, next_dice, events), events
    if dice is None:
        return new_turn(prog, p, next_dice, events), events
    ps = picks(prog, p, tmp, dice)
    if not ps:  # an edited bust roll: any action ends the turn
        events.append(dict(type='bust', p=p, dice=dice))
        return new_turn(prog, 1 - p, next_dice, events), events
    m = action.get('pick') if isinstance(action, dict) else None
    if not (isinstance(m, list) and len(m) == 2 and all(type(x) is int for x in m)):
        raise Bad('expected {pick: [a, b]}')
    m = tuple(m)
    for mv, t in ps:
        if mv == m:
            return state(prog, p, t, wfg=True), events
    raise Bad(f'illegal pick {list(m)} for dice {dice}')

def best_action(prog, p, tmp, dice, wfg):
    e = evaluate(prog, p, tmp, dice, wfg)
    return e['decision'] if wfg else dict(pick=e['picks'][0]['move'])

def api(f):
    def h():
        try:
            with lock:
                return jsonify(f())
        except Bad as e:
            return jsonify(error=str(e)), 400
    h.__name__ = f.__name__
    return h

def live(s):
    st = parse(s)
    if s.get('winner') in (0, 1):
        raise Bad('game is over')
    return st

@app.post('/eval')
@api
def eval_():
    return evaluate(*live(body_state()))

@app.post('/validate')
@api
def validate():
    parse(body_state())
    return dict(ok=True)

@app.post('/act')
@api
def act():
    st = live(body_state())
    if st[3] is None and not st[4]:
        return dict(action='roll')
    if st[3] is not None and not picks(*st[:4]):
        return dict(action='null')  # bust
    return dict(action=best_action(*st))

@app.post('/step')
@api
def step_():
    b = request.get_json(force=True)
    s, events = step(*live(body_state()), b.get('action'), check_dice(b.get('next_dice')))
    return dict(state=s, events=events)

def boundary(prog, p, cols):
    # stop/go/pbust for every tmp with markers on exactly cols, read from one turn-start solve; win cells (unexpanded
    # leaves there) get evaluate's won-stop values, whose bust odds depend only on which of the 3 columns are topped
    if not (isinstance(cols, list) and len(cols) == 3 and all(type(c) is int and 2 <= c <= 12 for c in cols)) \
            or len(set(cols)) < 3:
        raise Bad('cols: 3 distinct integers in 2..12')
    for c in cols:
        if fr.blocked(prog, c):
            raise Bad(f'column {c} is blocked')
    lo, hi = [int(prog[p, c]) + 1 for c in cols], [int(LEN[c]) for c in cols]  # lo <= hi: a full column is blocked
    sol = agent.solution(prog, p, np.zeros(13, np.int8))
    hs = np.meshgrid(*[np.arange(l, h + 1) for l, h in zip(lo, hi)], indexing='ij')
    o, mul, slot = np.argsort(cols), 1, fr.SIT_OFF[tuple(sorted(cols))]  # fr.slot, vectorised
    for a in o:
        slot, mul = slot + mul * (hs[a] - 1), mul * int(LEN[cols[a]])
    rows = sol.index.rows[slot]
    g = {k: np.where(rows >= 0, getattr(sol, k)[rows], np.nan) for k in ('stop', 'go', 'pbust')}
    win = sum(h == x for h, x in zip(hs, hi)) + fr.score(prog, p) >= 3  # fr.stop_wins; may be unreachable (no row)
    g['stop'][win] = 1
    bv, t = 1 - float(agent.value_fn(prog[None], 1 - p)[0]), np.zeros(13, np.int8)
    for top in product((0, 1), repeat=3):
        m = win & np.logical_and.reduce([(hs[a] == hi[a]) == top[a] for a in range(3)])
        if m.any():
            t[cols] = [h if q else l for l, h, q in zip(lo, hi, top)]
            g['go'][m], g['pbust'][m] = won_go(prog, p, t, bv)
    nul = lambda x: np.where(np.isnan(x), None, x).tolist()
    return dict(cols=cols, lo=lo, hi=hi, win=win.tolist(), v_net=float(agent.value_fn(prog[None], p)[0]),
                **{k: nul(v) for k, v in g.items()})

@app.post('/boundary')
@api
def boundary_():
    prog, p = live(body_state())[:2]
    return boundary(prog, p, request.get_json(force=True).get('cols'))

def body_state():
    b = request.get_json(force=True)
    if not isinstance(b, dict):
        raise Bad('body must be an object')
    return b.get('state', b)

@app.get('/info')
def info_():
    return jsonify(info)

@app.get('/')
def index():
    return send_file(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'index.html'))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', nargs='+', default=['runs/td30/it0082.pt'], help='one net, or several for an ensemble')
    ap.add_argument('--port', type=int, default=5000)
    a = ap.parse_args()
    init(a.ckpt)
    app.run(port=a.port, threaded=True)
