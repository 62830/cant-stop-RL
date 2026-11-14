import os, sys, copy, random
import numpy as np
import pytest
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'viz'))
import fastrules as fr
from game import State, Action
from solver_agent import SolverAgent, tmp_of
import server as S

@pytest.fixture(scope='module')
def c():
    if S.agent is None:
        S.init(os.path.join(ROOT, 'runs/td30/it0082.pt'))
    return S.app.test_client()

def js(s):
    # game.State -> API state
    prog, p, tmp = fr.from_state(s)
    return S.state(prog, p, tmp, None if s.wait_for_go else list(s.dice), s.wait_for_go)

def ok(r):
    assert r.status_code == 200, r.json
    return r.json

def test_eval_initial(c):
    st = S.state(np.zeros((2, 13), np.int8), 0)
    e0, e1 = ok(c.post('/eval', json=st)), ok(c.post('/eval', json=dict(st, p=1)))
    assert 0 < e0['v_net'] < 1 and 0 < e0['v_solved'] < 1 and e0['stop'] is None
    assert e0['v_net_p0'] == e0['v_net'] and abs(e1['v_net_p0'] - (1 - e1['v_net'])) < 1e-12
    assert abs(e1['v_solved_p0'] - (1 - e1['v_solved'])) < 1e-12
    e = ok(c.post('/eval', json=dict(st, dice=[1, 2, 3, 4])))
    vs = [d['value'] for d in e['picks']]
    assert vs == sorted(vs, reverse=True) and e['v_roll'] == vs[0] <= e['v_solved'] + 1e-9

def _states(n, seed=0):
    # mid-game states (picks and go/stop) from a random-ish playout
    random.seed(seed)
    out, s = [], State()
    while len(out) < n:
        if s.is_terminal():
            s = State()
        legal = s.get_legal_actions()
        if not legal[0].is_null() and len(legal) > 1 and random.random() < 0.3:
            out.append(copy.deepcopy(s))
        a = Action('go') if s.wait_for_go and random.random() < 0.8 else random.choice(legal)
        s, _ = s.update(a)
    return out

def test_act_matches_agent(c):
    ref = SolverAgent(S.agent.value_fn)
    for s in _states(20):
        a = ref.act(s)
        got = ok(c.post('/act', json=js(s)))['action']
        if s.wait_for_go:
            assert got == str(a)
        else:
            prog, p, tmp = fr.from_state(s)
            want = tmp_of(*s.unsafe_pick(a))
            new = fr.apply_move(prog, p, tmp, *got['pick'])
            sol = ref.solution(prog, p, tmp)
            v = lambda t: 1.0 if fr.stop_wins(prog, p, t) else sol.value(t)
            assert abs(v(new) - v(want)) < 1e-12

def test_autoplay(c):
    random.seed(1)
    s = ok(c.post('/step', json=dict(state=S.state(np.zeros((2, 13), np.int8), 0), action=None)))['state']
    for _ in range(2000):
        assert ok(c.post('/validate', json=s))['ok']
        if 'winner' in s:
            break
        a = ok(c.post('/act', json=s))['action']
        s = ok(c.post('/step', json=dict(state=s, action=a)))['state']
    assert s.get('winner') in (0, 1) and fr.score(np.array(s['prog'], np.int8), s['winner']) >= 3

def test_validate_rejects(c):
    z = np.zeros((2, 13), np.int8)
    base = S.state(z, 0)
    bad = [dict(base, prog=[[0] * 12 + [4], [0] * 13]),          # beyond LEN
           dict(base, prog=[[0] * 12 + [3], [0] * 12 + [1]]),     # claimed + opponent progress
           dict(base, prog=[[0, 0, 3, 5, 7] + [0] * 8, [0] * 13]),  # 3 claimed
           dict(base, tmp=[0, 0, 1, 1, 1, 1] + [0] * 7),           # 4 markers
           dict(base, prog=[[0, 0, 2] + [0] * 10, [0] * 13], tmp=[0, 0, 2] + [0] * 10),  # tmp not above prog
           dict(base, p=1, prog=[[0, 0, 3] + [0] * 10, [0] * 13], tmp=[0, 0, 1] + [0] * 10),  # blocked
           dict(base, dice=[0, 1, 2, 3]),
           dict(base, dice=[1, 2, 3]),
           dict(base, dice=[1, 2, 3, True]),
           dict(base, wait_for_go=True),
           dict(base, tmp=[1] + [0] * 12, dice=[1, 2, 3, 4]),                  # column 0 doesn't exist
           dict(base, prog=[[0, 2] + [0] * 11, [0] * 13]),
           dict(base, tmp=[0, 0, 1] + [0] * 10),                                 # mid-turn without dice or go/stop
           dict(base, prog=[[0] * 12 + [1000], [0] * 13]),                       # int8 overflow
           dict(base, prog=[[0] * 12 + [1.5], [0] * 13]),
           dict(base, p=True),
           dict(base, winner=0),                                                 # winner without 3 columns
           dict(base, prog=[[0, 0, 3, 5, 7] + [0] * 8, [0] * 13], winner=1)]
    for s in bad:
        r = c.post('/validate', json=s)
        assert r.status_code == 400 and r.json['error'], s
        assert c.post('/eval', json=s).status_code == 400

def test_bust(c):
    # markers on 2, 3, 4 and a roll of all 6s (only 12 reachable) busts
    prog = np.zeros((2, 13), np.int8)
    st = S.state(prog, 0, tmp_of([2, 3, 4], [1, 1, 1]), [6, 6, 6, 6])
    e = ok(c.post('/eval', json=st))
    assert e['bust'] and e['picks'] == [] and e['v_roll'] == e['bust_val']
    assert ok(c.post('/act', json=st))['action'] == 'null'
    r = ok(c.post('/step', json=dict(state=st, action='null')))
    assert r['events'][0]['type'] == 'bust' and r['state']['p'] == 1

def test_bad_requests(c):
    st = S.state(np.zeros((2, 13), np.int8), 0, dice=[1, 2, 3, 4])
    for b in ([1, 2], 'x', None):
        assert c.post('/eval', json=b).status_code == 400
    for d in ([7, 1, 1, 1], [1, 1, 1], ['1', 1, 1, 1], [True, 1, 1, 1], []):
        assert c.post('/step', json=dict(state=dict(st, dice=None), action=None, dice=d)).status_code == 400
    for a in (None, 'go', {'pick': 5}, {'pick': ['4', 6]}):
        assert c.post('/step', json=dict(state=st, action=a)).status_code == 400
    r = ok(c.post('/step', json=dict(state=dict(st, dice=None), action=None, dice=[6, 6, 6, 6])))
    assert r['state']['dice'] == [6, 6, 6, 6]

def test_won_stop(c):
    # P0 holds 2 and 3; a marker at the top of 12 wins on stop, go risks a bust
    prog = np.zeros((2, 13), np.int8)
    prog[0, 2], prog[0, 3] = 3, 5
    e = ok(c.post('/eval', json=S.state(prog, 0, tmp_of([12], [3]), wfg=True)))
    assert e['stop'] == 1 and 0 < e['pbust'] < 1 and e['go'] < 1 and e['decision'] == 'stop'

def _boundary(c, prog, p, cols, n=6):
    # grid shapes, and n random cells plus the all-top corner against /eval of that go/stop-phase state
    b = ok(c.post('/boundary', json=dict(state=S.state(prog, p), cols=cols)))
    shape = tuple(h - l + 1 for l, h in zip(b['lo'], b['hi']))
    assert b['cols'] == cols and b['lo'] == [prog[p, x] + 1 for x in cols]
    assert all(np.array(b[k], float).shape == shape for k in ('stop', 'go', 'pbust', 'win'))
    for ijk in [tuple(random.randrange(n) for n in shape) for _ in range(n)] + [tuple(n - 1 for n in shape)]:
        e = ok(c.post('/eval', json=S.state(prog, p, tmp_of(cols, [l + d for l, d in zip(b['lo'], ijk)]), wfg=True)))
        i, j, k = ijk
        assert b['win'][i][j][k] == (e['stop'] == 1)
        for x in ('stop', 'go', 'pbust'):
            assert abs(e[x] - b[x][i][j][k]) < 1e-9, (x, ijk)
    return b

def test_boundary(c):
    random.seed(2)
    prog = np.zeros((2, 13), np.int8)
    prog[1, 2:6] = 1, 2, 3, 4  # lo > 1 on 2 and 5
    prog[1, 12], prog[1, 11] = 3, 5  # P1 holds 11 and 12: any marker topping a column wins on stop
    b = _boundary(c, prog, 1, [5, 2, 7])
    assert b['win'][-1][-1][-1] and not b['win'][0][0][0]
    prog[1, 7] = 6  # same columns, new board
    b2 = _boundary(c, prog, 1, [5, 2, 7])
    assert b2['lo'][2] == 7 and len(b2['stop'][0][0]) == len(b['stop'][0][0]) - 6
    st = S.state(prog, 1)
    for cs in ([5, 5, 7], [5, 2, 11], [12, 2, 3], [1, 2, 3], [2, 3, 13], [2, 3], ['2', 3, 4]):  # 11, 12 blocked
        assert c.post('/boundary', json=dict(state=st, cols=cs)).status_code == 400
    prog[1, 2] = 3  # P1 has won
    assert c.post('/boundary', json=dict(state=S.state(prog, 1, winner=1), cols=[5, 6, 7])).status_code == 400

def _vnet(c, boards):
    return np.array([ok(c.post('/eval', json=S.state(b, p)))['v_net'] for b, p in boards])

def _boards():
    return [(np.zeros((2, 13), np.int8), 0)] + [fr.from_state(s)[:2] for s in _states(4, seed=3)]

def test_single_ckpt(c):
    from value_net import FusedNet, NetV
    assert S.info == dict(model='single', n=1, ckpts=[os.path.join(ROOT, 'runs/td30/it0082.pt')])
    assert isinstance(S.agent.value_fn, FusedNet) and c.get('/info').json == S.info
    ref = NetV.load(S.info['ckpts'][0])
    assert np.allclose(_vnet(c, _boards()), [ref(b[None], p)[0] for b, p in _boards()], atol=1e-6)

def test_ensemble_ckpts(c):
    from value_net import Ensemble, NetV
    ens = [os.path.join(ROOT, f) for f in ('runs/td30/it0082.pt', 'runs/scratch30/it0297.pt', 'runs/wide60/it0100.pt')]
    old = S.agent, S.info
    try:
        S.init(ens)
        assert c.get('/info').json == dict(model='ensemble', n=3, ckpts=ens)
        got, e, nets = _vnet(c, _boards()), Ensemble(ens), [NetV.load(f) for f in ens]
        assert (got == [e(b[None], p)[0] for b, p in _boards()]).all()
        z = [np.mean([n.logits(b[None], p).double().numpy() for n in nets]) for b, p in _boards()]
        assert np.allclose(got, 1 / (1 + np.exp(-np.array(z))), atol=1e-6, rtol=0)
        s = _states(1, seed=4)[0]
        assert ok(c.post('/act', json=js(s)))['action']
    finally:
        S.agent, S.info = old
