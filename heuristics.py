import numpy as np
from fastrules import LEN

# V_h(board, m) = sigmoid(theta . f(board, m) + bias): P(m wins) at the start of m's turn, features from m's POV
FRAC = LEN[2:].astype(np.float64)
KINDS = ('top3', 'sum', 'claim_top3')

def _top3(f):
    return -np.partition(-f, 2, axis=-1)[..., :3].sum(-1)

def features(progs, m, kind):
    progs = np.asarray(progs)
    f = progs[:, :, 2:] / FRAC
    me, op = f[:, m], f[:, 1 - m]
    if kind == 'top3':
        return (_top3(me) - _top3(op))[:, None]
    if kind == 'sum':
        return (me.sum(1) - op.sum(1))[:, None]
    if kind == 'claim_top3':
        return np.stack([_top3(me) - _top3(op), (me == 1).sum(1) - (op == 1).sum(1)], 1).astype(np.float64)
    raise ValueError(kind)

class VH:
    def __init__(self, kind='top3', theta=(4.0,), bias=0.0):
        self.kind, self.theta, self.bias = kind, np.asarray(theta, np.float64), float(bias)

    def __call__(self, progs, m):
        return 1 / (1 + np.exp(-(features(progs, m, self.kind) @ self.theta + self.bias)))

    def __repr__(self):
        return f'VH({self.kind!r}, theta={self.theta.round(4).tolist()}, bias={self.bias:.4f})'

def fit(X, y, l2=1e-6, iters=50):
    # logistic regression with intercept by Newton/IRLS; returns (theta, bias)
    A = np.c_[X, np.ones(len(X))]
    w = np.zeros(A.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-A @ w))
        g = A.T @ (p - y) + l2 * w
        H = (A * (p * (1 - p))[:, None]).T @ A + l2 * np.eye(len(w))
        step = np.linalg.solve(H, g)
        w -= step
        if np.abs(step).max() < 1e-10:
            break
    return w[:-1], w[-1]

def log_loss(p, y):
    p = np.clip(p, 1e-12, 1 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
