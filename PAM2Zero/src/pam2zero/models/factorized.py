import torch
from torch import nn
from pam2zero import config as cfg
from pam2zero.models.competition import DCompetition20PAM2
from pam2zero.models.factory import load_prior_D, reference_scale


class FactorizedDCompetition(DCompetition20PAM2):

    def __init__(self, seed):
        super().__init__(load_prior_D(seed), input_scale=reference_scale())
        d = self._parameters.pop('matrix_d').detach().clone()
        cols = torch.tensor([c - 1 for c in cfg.ORDINARY_CODES], device=d.device)
        mu = d[:, cols].mean()
        position = d[:, cols].mean(1) - mu
        kind = d.mean(0) - mu
        self.d_mu = nn.Parameter(mu.clone())
        self.d_position = nn.Parameter(position)
        self.d_type = nn.Parameter(kind)
        self.register_buffer('factor_active_columns', cols, persistent=False)
        self.check_inputs = False

    def __getattr__(self, name):
        if name == 'matrix_d' and 'd_mu' in self.__dict__.get('_parameters', {}):
            params = self.__dict__['_parameters']
            a = params['d_position']
            b = params['d_type']
            a = a - a.mean()
            b = b - b[self.factor_active_columns].mean()
            return params['d_mu'] + a[:, None] + b[None, :]
        return super().__getattr__(name)
