from __future__ import annotations
import math
import numpy as np
from pam2zero.activity import core
from pam2zero.activity.settings import BASE_KEYS, LEFT, PARAMETERS, RIGHT, nn, torch

def initial_parameters(variant, seed, bias=0.0, dtype=np.float32):
    if variant not in PARAMETERS:
        raise ValueError('Only baseline and shape_fixed are supported')
    p = core.make_parameters('path_pair', seed, bias, dtype)
    if variant != 'baseline':
        p['shape_weight'] = np.zeros((23, 5), dtype=dtype)
        p['shape_pair'] = np.zeros(5, dtype=dtype)
    return p

def reference_forward(p, x, shape, variant, return_parts=False):
    base = core.NumpyChain('path_pair', 0, dtype=p['guide'].dtype)
    base.set_arrays({k: p[k] for k in BASE_KEYS})
    prediction = base.predict(x)
    single = np.zeros(len(x), dtype=prediction.dtype)
    pair = single.copy()
    if variant != 'baseline':
        s = shape.copy()
        single = (s * p['shape_weight'][None]).sum((1, 2))
        prefix = np.r_[0.0, np.cumsum(core.EDGE_LOG_RANGE * np.tanh(p['edge']))]
        coeff = np.exp(prefix[RIGHT] - prefix[LEFT])
        coeff *= (1 / (1 + np.exp(-p['distance'])))[RIGHT - LEFT - 1]
        pair = (s[
            :,
            LEFT] * s[:,
            RIGHT] * p['shape_pair'][None,
            None] * coeff[None,
            :,
            None]).sum((1,
            2
        ))
    parts = dict(base=prediction, shape_single=single, shape_pair=pair)
    total = prediction + single + pair
    return (total, parts) if return_parts else total

class ShapeChain(nn.Module if nn is not None else object):

    def __init__(self, variant, seed, bias=0.0):
        super().__init__()
        self.variant = variant
        self.p = nn.ParameterDict({
            k: nn.Parameter(torch.from_numpy(v)) for k,
            v in initial_parameters(variant, seed, bias).items()
        })
        self.register_buffer('gp', torch.arange(20))
        self.register_buffer('pp', torch.arange(3))
        self.register_buffer('positions', torch.arange(23))
        self.register_buffer('left', torch.tensor(LEFT, dtype=torch.long))
        self.register_buffer('right', torch.tensor(RIGHT, dtype=torch.long))
        self.register_buffer('dist', torch.tensor(RIGHT - LEFT - 1, dtype=torch.long))

    def forward(self, x, shape, return_parts=False):
        p = self.p
        g = p['guide'] - p['guide'].mean(1, keepdim=True)
        pam = p['pam'] - p['pam'].mean(1, keepdim=True)
        base = g[
            self.gp[None],
            x[:, :20]].sum(1) + pam[self.pp[None],
            x[:, 20:]].sum(1) + p['bias'][0
        ]
        raw = p['interaction']
        J = raw - raw.mean(0, keepdim=True) - raw.mean(1, keepdim=True) + raw.mean()
        prefix = torch.cat([
            p['edge'].new_zeros(1),
            torch.cumsum(core.EDGE_LOG_RANGE * torch.tanh(p['edge']), 0)
        ])
        coeff = torch.exp(prefix[self.right] - prefix[self.left])
        coeff = coeff * torch.sigmoid(p['distance'])[self.dist]
        base = base + (J[x[:, self.left], x[:, self.right]] * coeff[None]).sum(1)
        single = base.new_zeros(len(x))
        pair = base.new_zeros(len(x))
        if self.variant != 'baseline':
            s = shape
            single = (s * p['shape_weight'][None]).sum((1, 2))
            pair = (s[
                :,
                self.left] * s[:,
                self.right] * p['shape_pair'][None,
                None] * coeff[None,
                :,
                None]).sum((1,
                2
            ))
        total = base + single + pair
        return (
            total,
            dict(base=base, shape_single=single, shape_pair=pair)
        ) if return_parts else total

class Optimizer:

    def __init__(self, model):
        self.p = dict(model.p.items())
        self.t = 0
        self.m = {k: torch.zeros_like(v) for k, v in self.p.items()}
        self.v = {k: torch.zeros_like(v) for k, v in self.p.items()}

    def step(self):
        self.t += 1
        alpha = core.LEARNING_RATE * math.sqrt(1 - 0.999 ** self.t) / (1 - 0.9 ** self.t)
        with torch.no_grad():
            for k, p in self.p.items():
                if p.grad is None:
                    continue
                p.mul_(1 - core.WEIGHT_DECAY)
                self.m[k].mul_(0.9).add_(p.grad, alpha=0.1)
                self.v[k].mul_(0.999).addcmul_(p.grad, p.grad, value=0.001)
                p.addcdiv_(self.m[k], self.v[k].sqrt() + core.ADAM_EPSILON, value=-alpha)

def arrays(model):
    return {k: v.detach().cpu().numpy().copy() for k, v in model.p.items()}

def restore(model, p):
    with torch.no_grad():
        for k, v in model.p.items():
            v.copy_(torch.as_tensor(p[k], dtype=v.dtype, device=v.device))

def predict(model, x, shape, indices=None, parts=False):
    if indices is None:
        indices = np.arange(len(x))
    predictions = []
    components = {}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(indices), 4096):
            ids = torch.as_tensor(indices[start:start + 4096], device=x.device, dtype=torch.long)
            p, c = model(x.index_select(0, ids), shape.index_select(0, ids), True)
            predictions.append(p.cpu().numpy().astype(np.float64))
            if parts:
                for k, v in c.items():
                    components.setdefault(k, []).append(v.cpu().numpy().astype(np.float64))
    result = np.concatenate(predictions)
    return (result, {k: np.concatenate(v) for k, v in components.items()}) if parts else result
