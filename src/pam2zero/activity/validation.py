from __future__ import annotations
import numpy as np
from pam2zero.activity.settings import PARAMETERS, torch

def self_test():
    from pam2zero.activity.model import ShapeChain, initial_parameters, reference_forward, restore
    from pam2zero.activity.data import fit_scaler
    rng = np.random.RandomState(91)
    x = rng.randint(0, 4, (13, 23))
    s = rng.normal(size=(13, 23, 5))
    for variant, count in PARAMETERS.items():
        p = initial_parameters(variant, 42, 0.4, np.float64)
        assert sum((v.size for v in p.values())) == count
        base = reference_forward(
            initial_parameters('baseline', 42, 0.4, np.float64),
            x,
            s,
            'baseline'
        )
        assert np.allclose(reference_forward(p, x, s, variant), base)
        if variant != 'baseline':
            p['shape_weight'][:] = rng.normal(0, 0.02, (23, 5))
            p['shape_pair'][:] = rng.normal(0, 0.01, 5)
        pred, parts = reference_forward(p, x, s, variant, True)
        assert np.allclose(pred, sum(parts.values()))
        assert np.allclose(
            pred,
            np.r_[
                reference_forward(p, x[:5], s[:5], variant),
                reference_forward(p, x[5:], s[5:], variant)
            ]
        )
    sc1 = fit_scaler(s, np.arange(8))
    changed = s.copy()
    changed[8:] += 10000
    sc2 = fit_scaler(changed, np.arange(8))
    assert all((np.array_equal(sc1[k], sc2[k]) for k in sc1))
    if torch is None:
        print('PASS: NumPy reference, counts, zero initialization, component sum, batch isolation, train-only scaler. Torch tests skipped: unavailable.')
        return
    for variant in PARAMETERS:
        m = ShapeChain(variant, 42, 0.4).double()
        values = initial_parameters(variant, 42, 0.4, np.float64)
        if variant != 'baseline':
            values['shape_weight'][:] = rng.normal(0, 0.02, (23, 5))
            values['shape_pair'][:] = rng.normal(0, 0.01, 5)
        restore(m, values)
        xx = torch.tensor(x, dtype=torch.long)
        ss = torch.tensor(s, dtype=torch.float64)
        yy = torch.tensor(rng.normal(0.4, 0.1, len(x)), dtype=torch.float64)
        assert np.allclose(
            m(xx, ss).detach().numpy(),
            reference_forward(values, x, s, variant),
            atol=1e-10
        )
        mse = ((m(xx, ss) - yy) ** 2).mean()
        objective = mse
        objective.backward()
        for name, param in m.p.items():
            assert param.grad is not None and bool(torch.isfinite(param.grad).all())
            index = param.numel() // 2
            analytic = float(param.grad.reshape(-1)[index])
            old = float(param.detach().reshape(-1)[index])
            h = 1e-06
            losses = []
            with torch.no_grad():
                for sign in [1, -1]:
                    param.reshape(-1)[index] = old + sign * h
                    value = ((m(xx, ss) - yy) ** 2).mean()
                    losses.append(float(value))
                param.reshape(-1)[index] = old
            assert abs(analytic - (losses[0] - losses[1]) / (2 * h)) < 1e-06, (variant, name)
    print('PASS: Torch/NumPy forward, finite-difference gradients, counts, train-only scaling.')
