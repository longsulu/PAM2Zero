from __future__ import annotations
from pam2zero import config as cfg
import torch

def align_prior(d):
    if cfg.D_SOURCE_ORDER == 'official_lookup':
        return torch.flip(d, dims=[0]).contiguous()
    if cfg.D_SOURCE_ORDER == 'input_forward':
        return d.clone()
    raise ValueError('D_SOURCE_ORDER must be official_lookup/input_forward')

def _make_component(seed, arm):
    from pam2zero.models.competition import DCompetition20PAM2
    from pam2zero.models.baselines import FreeAdditive
    from pam2zero.utils.io import seed_everything
    seed_everything(seed)
    if arm == 'simple_additive':
        return FreeAdditive().to(cfg.DEVICE)
    m = DCompetition20PAM2(load_prior_D(seed), input_scale=reference_scale()).to(cfg.DEVICE)
    m.check_inputs = False
    if arm in ('own', 'own_budget'):
        m.edge_raw.requires_grad_(False)
        m.raw_distance.requires_grad_(False)
    elif arm == 'frozen_D':
        m.matrix_d.requires_grad_(False)
    elif arm == 'random_D':
        a = m.initial_d[:, [c - 1 for c in cfg.ORDINARY_CODES]]
        r = torch.randn(
            (20, 36),
            generator=torch.Generator().manual_seed(seed + 314159)) * float(a.std(unbiased=False)) + float(a.mean()
        )
        with torch.no_grad():
            m.matrix_d.copy_(r.to(cfg.DEVICE))
            m.initial_d.copy_(r.to(cfg.DEVICE))
            m.prior_template[1:21].copy_(r.to(cfg.DEVICE))
    elif arm == 'no_pam':
        m.use_pam = False
        m.matrix_pam.requires_grad_(False)
    elif arm == 'pam_zero':
        with torch.no_grad():
            m.matrix_pam.zero_()
            m.initial_pam.zero_()
            m.prior_template[22:24].zero_()
    elif arm == 'unit_edges':
        m.edge_raw.requires_grad_(False)
    elif arm == 'constant_distance':
        m.raw_distance.requires_grad_(False)
    elif arm != 'full':
        raise ValueError(arm)
    return m

def load_prior_D(seed=42):
    st = cfg.FINAL_INITIAL_STATES.get(seed, cfg.FINAL_INITIAL_STATES.get(42))
    if st is None:
        raise RuntimeError('Source initialization not loaded')
    d = st['prior_template'].detach().cpu().clone()
    d[1:21] = st['initial_d'].detach().cpu()
    d[21:24] = 0.0
    return d

def reference_prior():
    return load_prior_D(42)

def reference_scale():
    return torch.tensor(float(cfg.FINAL_SCALE))

def make_model(seed, arm):
    if arm == 'factor_D':
        from pam2zero.models.factorized import FactorizedDCompetition
        from pam2zero.utils.io import seed_everything
        seed_everything(seed)
        return FactorizedDCompetition(seed).to(cfg.DEVICE)
    from pam2zero.models.baselines import PureDCalculation
    if arm == 'pure_D':
        return PureDCalculation(seed).to(cfg.DEVICE)
    if arm == 'pam_one':
        m = _make_component(seed, 'full')
        with torch.no_grad():
            m.matrix_pam.zero_()
            m.matrix_pam[:, [c - 1 for c in cfg.MATCH_CODES]] = 1.0
            m.initial_pam.copy_(m.matrix_pam)
            m.prior_template[22:24].copy_(m.matrix_pam)
        return m
    return _make_component(seed, arm)
