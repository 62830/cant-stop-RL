import numpy as np
import torch
from torch import nn
from fastrules import LEN

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
    # value_fn adapter: P(m wins) at the start of m's turn; single-threaded inference (one per worker)
    def __init__(self, net, threads=1):
        self.net, self.threads = net.eval(), threads
        self.hidden = tuple(l.out_features for l in net if isinstance(l, nn.Linear))[:-1]
        torch.set_num_threads(threads)

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

    def save(self, path):
        torch.save(self.__getstate__(), path)

    @classmethod
    def load(cls, path, threads=1):
        v = cls.__new__(cls)
        v.__setstate__(dict(torch.load(path), threads=threads))
        return v
