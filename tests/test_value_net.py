import os, sys, json, pickle, subprocess
import pytest
import numpy as np
import torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import fastrules as fr
from value_net import NetV, make_net, features, N_IN
from td_train import pov
from turn_solver import solve_turn

def boards(n, seed=0):
    rng = np.random.default_rng(seed)
    progs = np.zeros((n, 2, 13), np.int8)
    for c in range(2, 13):
        progs[:, :, c] = rng.integers(0, fr.LEN[c] + 1, (n, 2))
        both = progs[:, 0, c] == fr.LEN[c]
        progs[both, 1, c] = np.minimum(progs[both, 1, c], fr.LEN[c] - 1)
    return progs

def test_features_shape_and_symmetry():
    progs = boards(500)
    f = features(progs, 0)
    assert f.shape == (500, N_IN) and f.dtype == bool
    assert (features(progs[:, ::-1], 1) == f).all()
    assert (features(pov(progs, np.ones(500, int)), 0) == features(progs, 1)).all()
    # thermometer counts heights; claimed bits flag full columns
    assert (f[:, :72].sum(1) == np.minimum(progs[:, 0, 2:], fr.LEN[2:] - 1).sum(1)).all()
    assert (f[:, 144:155] == (progs[:, 0, 2:] == fr.LEN[2:])).all()
    assert not features(np.zeros((1, 2, 13), np.int8), 0).any()

def test_adapter_matches_forward():
    torch.manual_seed(0)
    net = make_net((32, 16))
    v = NetV(net)
    progs = boards(300, 1)
    for m in (0, 1):
        with torch.no_grad():
            ref = torch.sigmoid(net(torch.tensor(features(progs, m), dtype=torch.float32)))[:, 0].double().numpy()
        out = v(progs, m)
        assert out.dtype == np.float64 and out.shape == (300,)
        np.testing.assert_allclose(out, ref, rtol=1e-6)
    np.testing.assert_allclose(pickle.loads(pickle.dumps(v))(progs, 0), v(progs, 0))
    sol = solve_turn(progs[0], 0, v, depth=2)
    assert 0 <= sol.value(np.zeros(13, np.int8)) <= 1

def test_tiny_training_iteration_and_resume(tmp_path):
    cmd = [sys.executable, os.path.join(ROOT, 'td_train.py'), '--run', 'test', '--workers', '2', '--warm_games', '3', '--games', '2',
           '--iters', '1', '--n_resid', '4', '--calib_every', '1', '--hidden', '16,8']
    run = lambda *extra: subprocess.run(cmd + list(extra), cwd=tmp_path, capture_output=True, env={**os.environ, 'PYTHONPATH': ROOT})
    d = tmp_path / 'runs' / 'test'
    assert run().returncode == 0
    recs = [json.loads(l) for l in open(d / 'metrics.jsonl')]
    assert {'config', 'warm', 'update', 'iter', 'calib'} <= {r['kind'] for r in recs}
    it = next(r for r in recs if r['kind'] == 'iter')
    assert it['iter'] == 1 and it['residual'] >= 0 and it['pairs'] > 0
    assert len(next(r for r in recs if r['kind'] == 'update')['loss']) == it['steps']
    assert (d / 'DONE').exists() and not (d / 'train.pid').exists()
    assert NetV.load(str(d / 'ckpt' / 'it0001.pt'))(boards(5), 0).shape == (5,)
    assert run().returncode != 0  # existing run dir without --resume
    assert run('--resume').returncode == 0
    recs = [json.loads(l) for l in open(d / 'metrics.jsonl')]
    assert [r['iter'] for r in recs if r['kind'] == 'resume'] == [1]
    assert [r['iter'] for r in recs if r['kind'] == 'iter'] == [1, 2]
    assert (d / 'ckpt' / 'it0002.pt').exists() and not list((d / 'ckpt').glob('*.tmp'))

def test_tiny_training_no_warm(tmp_path):
    cmd = [sys.executable, os.path.join(ROOT, 'td_train.py'), '--run', 'test', '--workers', '2', '--warm_games', '3', '--games', '2',
           '--iters', '1', '--n_resid', '4', '--calib_every', '1', '--hidden', '16,8', '--no_warm']
    assert subprocess.run(cmd, cwd=tmp_path, capture_output=True, env={**os.environ, 'PYTHONPATH': ROOT}).returncode == 0
    d = tmp_path / 'runs' / 'test'
    recs = [json.loads(l) for l in open(d / 'metrics.jsonl')]
    assert next(r for r in recs if r['kind'] == 'warm')['init'] == 'random'
    assert [r['iter'] for r in recs if r['kind'] == 'iter'] == [1]
    assert (d / 'ckpt' / 'it0000.pt').exists() and (d / 'ckpt' / 'it0001.pt').exists()

def test_ensemble_mean_logit():
    from value_net import Ensemble
    from turn_solver import _stop_boards
    torch.manual_seed(0)
    nets = [NetV(make_net((128, 64))) for _ in range(3)]
    e = pickle.loads(pickle.dumps(Ensemble([n.fused() for n in nets])))
    with pytest.raises(TypeError, match='FusedNets'):
        Ensemble([NetV(make_net((16,)))])
    progs = boards(50, 2)
    sig = lambda z: 1 / (1 + np.exp(-z))
    for m in (0, 1):
        ref = sig(np.mean([n.logits(progs, m).double().numpy() for n in nets], 0))
        np.testing.assert_allclose(e(progs, m), ref, atol=1e-6)
    for prog in progs[:10]:
        p = 0
        keys, _, _ = fr.enumerate_turn(prog, p)
        bs, win = _stop_boards(prog, p, keys)
        ask = ~win & (keys != 0)
        q = np.concatenate([bs[ask], prog[None]])
        ref = sig(np.mean([n.logits(q, 1 - p).double().numpy() for n in nets], 0))
        np.testing.assert_allclose(e.banked(prog, p, keys[ask]), ref, atol=1e-5)

def test_stop_eps_warmup(tmp_path, monkeypatch):
    import importlib.util, td_train
    from game import State
    old = tmp_path / 'td_train_head.py'
    try:
        src = subprocess.run(['git', 'show', '3fbcb7a:td_train.py'], cwd=ROOT, capture_output=True, text=True)
    except OSError:
        src = None
    if src is None or src.returncode or not src.stdout:
        pytest.skip('git or pinned commit 3fbcb7a (td_train before stop_eps) unavailable')
    old.write_text(src.stdout)
    spec = importlib.util.spec_from_file_location('td_train_head', old)
    head = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(head)
    same = lambda x, y: all(np.array_equal(a[0], b[0]) and a[1:] == b[1:] for a, b in zip(x, y)) and len(x) == len(y)
    for seed in (1, 2):
        assert same(td_train.play((td_train.BEST_VH, seed, None, 0.0))[0], head.play((td_train.BEST_VH, seed))[0])
    acts, update = [], State.update
    monkeypatch.setattr(State, 'update', lambda s, a: (acts.append((s.wait_for_go, a.value)), update(s, a))[1])
    td_train.play((td_train.BEST_VH, 3, None, 1.0))
    gs = [v for w, v in acts if w]
    assert gs and all(v == 2 for v in gs)  # every go/stop decision stops

def test_resume_keeps_dropped_lr(tmp_path):
    cmd = [sys.executable, os.path.join(ROOT, 'td_train.py'), '--run', 'test', '--workers', '2', '--warm_games', '3', '--games', '2',
           '--iters', '1', '--n_resid', '4', '--hidden', '16,8', '--no_warm', '--n_set', '0', '--lr_drop_frac', '0', '--lr_low', '2e-4']
    run = lambda *extra: subprocess.run(cmd + list(extra), cwd=tmp_path, capture_output=True, env={**os.environ, 'PYTHONPATH': ROOT})
    assert run().returncode == 0 and run('--resume').returncode == 0
    recs = [json.loads(l) for l in open(tmp_path / 'runs' / 'test' / 'metrics.jsonl')]
    assert [r['lr'] for r in recs if r['kind'] == 'lr_drop'] == [2e-4]  # not dropped again: lr is already low
    assert [r['lr'] for r in recs if r['kind'] == 'resume'] == [2e-4]
