from __future__ import annotations
from pam2zero import config as cfg
import torch.nn.functional as F
from torch import nn
import torch


class FreeAdditive(nn.Module):

    def __init__(self):
        super().__init__()
        self.matrix_d = nn.Parameter(torch.zeros(20, 36))
        self.matrix_pam = nn.Parameter(torch.zeros(2, 36))
        self.bias = nn.Parameter(torch.zeros(()))

    def forward(self, codes):
        z = self.matrix_d[
            torch.arange(20, device=codes.device)[None],
            codes[:, :20].long() - 1].sum(1
        )
        z = z + self.matrix_pam[
            torch.arange(2, device=codes.device)[None],
            codes[:, 21:23].long() - 1].sum(1
        ) + self.bias
        return torch.stack([-z / 2, z / 2], 1)


class PureDCalculation(FreeAdditive):

    def __init__(self, seed):
        from pam2zero.models.factory import load_prior_D, reference_scale
        nn.Module.__init__(self)
        prior = load_prior_D(seed)
        self.matrix_d = nn.Parameter(prior[1:21].clone())
        self.matrix_pam = nn.Parameter(torch.zeros(2, 36))
        self.register_buffer('input_scale', reference_scale().clone())

    def forward(self, codes):
        x = codes[:, :20].long()
        match = (x == 1) | (x == 7) | (x == 13) | (x == 19)
        value = F.softplus(self.matrix_d[
            torch.arange(20, device=x.device)[None],
            x - 1
        ] / self.input_scale) + cfg.FORCE_FLOOR
        z = (value * torch.where(match, 1.0, -1.0)).sum(1)
        z += self.matrix_pam[
            torch.arange(2, device=x.device)[None],
            codes[:, 21:23].long() - 1].sum(1
        ) / self.input_scale
        return torch.stack([-z / 2, z / 2], dim=1)
