import os
import numpy as np
import torch
from torch import nn
from numba import njit
from fastrules import LEN, STRIDE

# features from m's POV: my thermometer, opp thermometer (cells 1..LEN-1 per column, 72 each), claimed by me, by opp
COL = np.concatenate([np.full(LEN[c] - 1, c) for c in range(2, 13)])
THR = np.concatenate([np.arange(1, LEN[c]) for c in range(2, 13)]).astype(np.int8)
N_IN = 2 * len(COL) + 22

def features(progs, m):
    progs = np.asarray(progs)
    me, op = progs[:, m], progs[:, 1 - m]
    return np.concatenate([me[:, COL] >= THR, op[:, COL] >= THR, me[:, 2:] == LEN[2:], op[:, 2:] == LEN[2:]], 1)

def make_net(hidden=(128, 64)):
    dims, layers = (N_IN,) + tuple(hidden), []
    for a, b in zip(dims, dims[1:]):
        layers += [nn.Linear(a, b), nn.ReLU()]
    return nn.Sequential(*layers, nn.Linear(dims[-1], 1))

class NetV:
    # value_fn adapter: P(m wins) at the start of m's turn; torch threads are set only when unpickled/loaded
    # (workers: 1 thread each), never as a side effect of construction
    def __init__(self, net, threads=1):
        self.net, self.threads = net.eval(), threads
        self.hidden = tuple(l.out_features for l in net if isinstance(l, nn.Linear))[:-1]

    @torch.inference_mode()
    def logits(self, progs, m):
        return self.net(torch.from_numpy(features(progs, m)).float())[:, 0]

    def __call__(self, progs, m):
        return torch.sigmoid(self.logits(progs, m)).double().numpy()

    def __getstate__(self):
        return {'hidden': self.hidden, 'sd': self.net.state_dict(), 'threads': self.threads}

    def __setstate__(self, d):
        net = make_net(d['hidden'])
        net.load_state_dict(d['sd'])
        self.__init__(net, d['threads'])
        torch.set_num_threads(self.threads)

    def fused(self):
        return FusedNet(self)

    def save(self, path):
        torch.save(self.__getstate__(), path + '.tmp')
        os.replace(path + '.tmp', path)

    @classmethod
    def load(cls, path, threads=1):
        v = cls.__new__(cls)
        v.__setstate__(dict(torch.load(path), threads=threads))
        return v

class FusedNet:
    # NetV (two hidden layers) plus a numba kernel for turn_solver.stop_values. A stop board differs from the turn start
    # only in the mover's <= 3 tmp columns (thermometer cells + claim bit, and a claim resets the opponent's column), so
    # h1 = W1 x(stop board) + b1 = h1(turn start) + per-(column, height) deltas; no feature matrix is built.
    # Boards are evaluated as NetV does (P(to_move wins)); __call__ is the torch NetV itself
    def __init__(self, v):
        self.v = v
        l1, l2, l3 = [l.weight.detach().double().numpy() for l in v.net if isinstance(l, nn.Linear)]
        b1, b2, b3 = [l.bias.detach().double().numpy() for l in v.net if isinstance(l, nn.Linear)]
        # cum[blk, c, h] = W1 columns of the features on when column c of the blk (0 me, 1 opp) player is at height h
        cum = np.zeros((2, 13, 14, len(b1)))
        cells = np.cumsum(np.r_[0, LEN[2:12] - 1])
        for blk in (0, 1):
            for i, c in enumerate(range(2, 13)):
                for h in range(1, LEN[c] + 1):
                    on = [blk * len(COL) + cells[i] + t - 1 for t in range(1, min(h, LEN[c] - 1) + 1)]
                    on += [2 * len(COL) + 11 * blk + i] if h == LEN[c] else []
                    cum[blk, c, h] = l1[:, on].sum(1)
        f32 = lambda a: np.ascontiguousarray(a, np.float32)
        self.w = (cum, b1, f32(l2.T), f32(b2), f32(l3[0]), float(b3[0]))

    def __call__(self, progs, m):
        return self.v(progs, m)

    def stop_values(self, prog, p, keys):
        return _fused_stop(prog, p, keys, *self.w)

    def __getstate__(self):
        return self.v.__getstate__()

    def __setstate__(self, d):
        v = NetV.__new__(NetV)
        v.__setstate__(d)
        self.__init__(v)

BLOCK = 32  # boards per layer-2 GEMM; 32 x 128 x 64 stays under OpenBLAS's multithreading threshold (single-threaded workers)

@njit(cache=True, fastmath=True)
def _flush(R, n, rows, w2t, b2, w3, b3, out):
    # out[rows[:n]] = 1 - sigmoid(net) for the relu'd first-layer activations R[:n]
    Z = R[:n] @ w2t
    for b in range(n):
        z = b3
        for j in range(len(b2)):
            a = Z[b, j] + b2[j]
            if a > 0:
                z += a * w3[j]
        out[rows[b]] = 1 - 1 / (1 + np.exp(-z))

@njit(cache=True)
def _fused_stop(prog, p, keys, cum, b1, w2t, b2, w3, b3):
    # turn_solver.stop_values for the net (queried for the opponent m = 1 - p, so p's columns are the "opp" block)
    m, n1 = 1 - p, len(b1)
    base = b1.copy()
    owned = 0
    for c in range(2, 13):
        base += cum[0, c, prog[m, c]] + cum[1, c, prog[p, c]]
        owned += prog[p, c] == LEN[c]
    delta = np.zeros((13, 14, n1), np.float32)
    for c in range(2, 13):
        for h in range(prog[p, c] + 1, LEN[c] + 1):
            d = cum[1, c, h] - cum[1, c, prog[p, c]]
            if h == LEN[c]:
                d = d - cum[0, c, prog[m, c]]
            delta[c, h] = d.astype(np.float32)
    base32 = base.astype(np.float32)
    R, rows = np.empty((BLOCK, n1), np.float32), np.empty(BLOCK, np.int64)
    cs, hs = np.zeros(3, np.int64), np.zeros(3, np.int64)
    stop = np.empty(len(keys) + 1)
    nb = 0
    for i in range(len(keys) + 1):
        k = keys[i] if i < len(keys) else 0
        if k == 0 and i < len(keys):
            stop[i] = -np.inf
            continue
        w, nc = owned, 0
        for c in range(12, 1, -1):
            h = k // STRIDE[c]
            if h > 0:
                k -= h * STRIDE[c]
                cs[nc], hs[nc] = c, h
                nc += 1
                w += h == LEN[c]
        if w >= 3:
            stop[i] = 1.0
            continue
        for c in range(nc, 3):
            cs[c], hs[c] = 0, 0
        d0, d1, d2 = delta[cs[0], hs[0]], delta[cs[1], hs[1]], delta[cs[2], hs[2]]
        for j in range(n1):
            R[nb, j] = max(base32[j] + d0[j] + d1[j] + d2[j], 0)
        rows[nb] = i
        nb += 1
        if nb == BLOCK or i == len(keys):
            _flush(R, nb, rows, w2t, b2, w3, b3, stop)
            nb = 0
    return stop[:-1], stop[-1]
