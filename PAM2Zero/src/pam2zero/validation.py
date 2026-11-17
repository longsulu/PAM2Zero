from __future__ import annotations
from pam2zero import config as cfg
import torch


@torch.no_grad()
def final_self_test():
    from pam2zero.models.factory import make_model
    from pam2zero.evaluation.inference import score_own_float64
    saved = cfg.FINAL_INITIAL_STATES
    scale = cfg.FINAL_SCALE
    if not saved:
        p = torch.zeros(24, 36)
        p[1:21] = torch.linspace(-1, 1, 720).reshape(20, 36)
        cfg.FINAL_INITIAL_STATES = {42: dict(prior_template=p, initial_d=p[1:21].clone())}
        cfg.FINAL_SCALE = 0.5
    m = make_model(42, 'full').double().eval()
    x = torch.ones(5, 23, dtype=torch.long, device=cfg.DEVICE)
    x[1, 4] = 2
    x[2, 8] = 8
    x[3, 4] = 2
    x[3, 8] = 8
    assert sum((p.numel() for p in m.parameters())) == 833
    assert torch.count_nonzero(m.initial_pam) == 0
    own = score_own_float64(m, x.cpu().numpy())
    assert abs(own[3] - own[1] - own[2] + own[0]) < 1e-10
    a = m(x)
    x[:, 20] = 7
    torch.testing.assert_close(a, m(x), rtol=0, atol=0)
    torch.testing.assert_close(m(x), torch.cat([m(xx[None]) for xx in x]), rtol=1e-10, atol=1e-10)
    for arm, expected in [
        ('own', 795),
        ('frozen_D', 113),
        ('no_pam', 761),
        ('unit_edges', 814),
        ('constant_distance', 814),
        ('pam_one', 833),
        ('pure_D', 792)
    ]:
        q = make_model(42, arm)
        assert sum((p.numel() for p in q.parameters() if p.requires_grad)) == expected, (
            arm,
            expected
        )
    cfg.FINAL_INITIAL_STATES = saved
    cfg.FINAL_SCALE = scale
    print('PASS:833 parameters; ZERO PAM initialization; PAM-first ignored; OWN additive; batch isolation; ablation parameter counts.')
