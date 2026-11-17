from __future__ import annotations
from pam2zero import config as cfg
import torch.nn.functional as F
import math
from torch import nn
import torch


def _inverse_softplus(value):
    return math.log(math.expm1(value))


class DCompetition20PAM2(nn.Module):

    def __init__(self, prior, use_pam=cfg.USE_PAM, input_scale=None):
        super().__init__()
        d = torch.as_tensor(prior, dtype=torch.float32).clone()
        if tuple(d.shape) != (24, 36):
            raise ValueError('PAM model requires original (24,36) D')
        self.use_pam = bool(use_pam)
        if not bool(torch.isfinite(d).all()):
            raise ValueError('non-finite D')
        if not 0 < cfg.CONTEST_INIT < cfg.CONTEST_MAX:
            raise ValueError('require 0<CONTEST_INIT<CONTEST_MAX')
        d20 = d[1:21].clone()
        self.matrix_d = nn.Parameter(d20)
        self.matrix_pam = nn.Parameter(d[22:24].clone(), requires_grad=self.use_pam)
        self.register_buffer('initial_pam', d[22:24].clone())
        self.register_buffer('pam_positions', torch.arange(2), persistent=False)
        self.register_buffer('initial_d', d20.clone())
        self.register_buffer('prior_template', d)
        values = d20[:, [c - 1 for c in cfg.MATCH_CODES + cfg.MISMATCH_CODES]].abs().flatten()
        values = values[values > 0]
        scale = values.median().clamp_min(0.0001) if len(values) else torch.tensor(1.0)
        if input_scale is not None:
            scale = torch.as_tensor(input_scale, dtype=torch.float32).clone()
        self.register_buffer('input_scale', scale)
        self.register_buffer('positions', torch.arange(20), persistent=False)
        match = torch.zeros(37, dtype=torch.bool)
        match[list(cfg.MATCH_CODES)] = True
        valid = torch.zeros(37, dtype=torch.bool)
        valid[list(cfg.MATCH_CODES + cfg.MISMATCH_CODES)] = True
        self.register_buffer('match_lut', match, persistent=False)
        self.register_buffer('valid_lut', valid, persistent=False)
        self.raw_team_gain = nn.Parameter(torch.full((2,), _inverse_softplus(1.0 - 0.0001)))
        self.raw_temperature = nn.Parameter(torch.tensor(_inverse_softplus(0.95)))
        self.raw_distance = nn.Parameter(torch.full(
            (19,),
            math.log(cfg.CONTEST_INIT / (cfg.CONTEST_MAX - cfg.CONTEST_INIT))
        ))
        self.edge_raw = nn.Parameter(torch.zeros(19))
        self.check_inputs = True
        self.active_codes = cfg.MATCH_CODES + cfg.MISMATCH_CODES

    def team_gains(self):
        return F.softplus(self.raw_team_gain) + 0.0001

    def temperature(self):
        return F.softplus(self.raw_temperature) + 0.05

    def distance_strengths(self):
        return cfg.CONTEST_MAX * torch.sigmoid(self.raw_distance)

    def edge_gains(self):
        return torch.exp(cfg.EDGE_LOG_RANGE * torch.tanh(self.edge_raw))

    def contributions(self, codes, return_pairs=False):
        if codes.ndim != 2 or codes.shape[1] != 23:
            raise ValueError('expected normalized (B,23): guide20 + PAM3')
        full = codes.long()
        if self.check_inputs:
            if torch.any((full < 1) | (full > 36)):
                raise ValueError('pair codes must be 1..36')
            if not bool(self.valid_lut[full[:, 20:]].all()):
                raise ValueError('PAM has unsupported gap/bulge')
        x = full[:, :20]
        if self.check_inputs:
            if torch.any((x < 1) | (x > 36)):
                raise ValueError('pair codes must be 1..36')
            if not bool(self.valid_lut[x].all()):
                raise ValueError('Guide positions contain unsupported gap/bulge codes')
        is_match = self.match_lut[x]
        d_raw = self.matrix_d[self.positions[None], x - 1]
        gains = self.team_gains()
        node_force = (F.softplus(d_raw / self.input_scale) + cfg.FORCE_FLOOR) * torch.where(
            is_match,
            gains[0],
            gains[1]
        )
        own_match = (node_force * is_match).sum(1)
        own_mismatch = (node_force * ~is_match).sum(1)
        bonus_match = node_force.new_zeros(len(x))
        bonus_mismatch = node_force.new_zeros(len(x))
        edge = self.edge_gains()
        strength = self.distance_strengths()
        # Products follow adjacent edges, without recurrent message passing.
        path = node_force.new_ones(20)
        records = []
        for distance in range(1, 20):
            path = path[:-1] * edge[distance - 1:]
            left_match = is_match[:, :-distance]
            cross = left_match.ne(is_match[:, distance:])
            left_force = node_force[:, :-distance]
            right_force = node_force[:, distance:]
            pot = strength[distance - 1] * path[None] * cross
            left_award = pot * (left_force / (left_force + right_force))
            right_award = pot - left_award
            match_award = torch.where(left_match, left_award, right_award)
            mismatch_award = torch.where(left_match, right_award, left_award)
            bonus_match = bonus_match + match_award.sum(1)
            bonus_mismatch = bonus_mismatch + mismatch_award.sum(1)
            if return_pairs:
                records.append((distance, cross, pot, left_award, right_award))
        match_total = own_match + bonus_match
        mismatch_total = own_mismatch + bonus_mismatch
        core_margin = match_total - mismatch_total
        pam_values = self.matrix_pam[self.pam_positions[None], full[:, 21:23] - 1]
        pam_score = pam_values.sum(1) / self.input_scale if self.use_pam else core_margin.new_zeros(len(x))
        margin = core_margin + pam_score
        return dict(
            core_margin=core_margin,
            pam_values=pam_values,
            pam_score=pam_score,
            raw_d=d_raw,
            node_force=node_force,
            is_match=is_match,
            own_match=own_match,
            own_mismatch=own_mismatch,
            bonus_match=bonus_match,
            bonus_mismatch=bonus_mismatch,
            match_total=match_total,
            mismatch_total=mismatch_total,
            margin=margin,
            score=margin / self.temperature(),
            pairs=records
        )

    def forward_legacy(self, codes):
        c = self.contributions(codes)
        return torch.stack(
            [c['mismatch_total'] - 0.5 * c['pam_score'], c['match_total'] + 0.5 * c['pam_score']],
            dim=1) / self.temperature(
        )

    def forward(self, codes):
        if cfg.SOURCE_NUMERICS != 'typed_vector':
            return self.forward_legacy(codes)
        z = typed_source_scores(self, codes)['full']
        return torch.stack([-z / 2, z / 2], 1)


def typed_source_scores(model, codes):
    x = codes[:, :20].long()
    match = model.match_lut[x]
    raw = model.matrix_d[model.positions[None], x - 1]
    gains = model.team_gains()
    f = (F.softplus(raw / model.input_scale) + cfg.FORCE_FLOOR) * torch.where(
        match,
        gains[0],
        gains[1]
    )
    single = (f * torch.where(match, 1.0, -1.0)).sum(1)
    pam = model.matrix_pam[
        model.pam_positions[None],
        codes[
            :,
            21:23].long() - 1].sum(1) / model.input_scale if model.use_pam else single.new_zeros(len(x
        )
    )
    pi, pj = torch.triu_indices(20, 20, offset=1, device=x.device)
    pd = pj - pi - 1
    lf = f[:, pi]
    rf = f[:, pj]
    mi = match[:, pi]
    mj = match[:, pj]
    prefix = torch.cat([
        model.edge_raw.new_zeros(1),
        (cfg.EDGE_LOG_RANGE * torch.tanh(model.edge_raw)).cumsum(0)
    ])
    path = (prefix[pj] - prefix[pi]).exp()
    pot = model.distance_strengths()[pd] * path
    contest_pair = pot[None] * (mi != mj) * torch.where(mi, 1.0, -1.0) * (lf - rf) / (lf + rf)
    competition = contest_pair.sum(1)
    own = (single + pam) / model.temperature()
    return dict(own=own, full=own + competition / model.temperature())
