import os, sys, json, pickle, subprocess
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

def test_tiny_training_iteration(tmp_path):
    cmd = [sys.executable, os.path.join(ROOT, 'td_train.py'), '--run', 'test', '--workers', '2', '--warm_games', '3', '--games', '2',
           '--iters', '1', '--n_resid', '4', '--calib_every', '1', '--hidden', '16,8']
    subprocess.run(cmd, cwd=tmp_path, check=True, capture_output=True, env={**os.environ, 'PYTHONPATH': ROOT})
    recs = [json.loads(l) for l in open(tmp_path / 'runs' / 'test' / 'metrics.jsonl')]
    kinds = [r['kind'] for r in recs]
    assert {'config', 'warm', 'update', 'iter', 'calib'} <= set(kinds)
    it = next(r for r in recs if r['kind'] == 'iter')
    assert it['residual'] >= 0 and it['pairs'] > 0
    assert (tmp_path / 'runs' / 'test' / 'ckpt' / 'it0001.pt').exists()
    assert NetV.load(str(tmp_path / 'runs' / 'test' / 'ckpt' / 'it0001.pt'))(boards(5), 0).shape == (5,)
