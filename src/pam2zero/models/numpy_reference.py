import numpy as np
from pam2zero import config as cfg


def numpy_reference(state, codes):
    sp = lambda v: np.logaddexp(0, np.asarray(v))
    x = np.asarray(codes, dtype=int)
    guide = x[:, :20]
    match = np.isin(guide, cfg.MATCH_CODES)
    gain = sp(state['raw_team_gain']) + 0.0001
    f = (sp(state['matrix_d'][
        np.arange(20)[None],
        guide - 1] / state['input_scale'
    ]) + cfg.FORCE_FLOOR) * np.where(
        match,
        gain[0],
        gain[1]
    )
    a = (f * match).sum(1)
    b = (f * ~match).sum(1)
    ba = np.zeros(len(x))
    bb = np.zeros(len(x))
    edge = np.exp(cfg.EDGE_LOG_RANGE * np.tanh(state['edge_raw']))
    dist = cfg.CONTEST_MAX / (1 + np.exp(-state['raw_distance']))
    for i in range(20):
        for j in range(i + 1, 20):
            pot = dist[j - i - 1] * np.prod(edge[i:j]) * (match[:, i] != match[:, j])
            award = pot * f[:, i] / (f[:, i] + f[:, j])
            other = pot - award
            ba += np.where(match[:, i], award, other)
            bb += np.where(match[:, i], other, award)
    npam = state['matrix_pam'].shape[0]
    pam = state['matrix_pam'][
        np.arange(npam)[None],
        x[:, 23 - npam:] - 1].sum(1) / state['input_scale'
    ]
    temp = sp(state['raw_temperature']) + 0.05
    return dict(
        full=(a + ba + 0.5 * pam) / temp - (b + bb - 0.5 * pam) / temp,
        own=(a + 0.5 * pam) / temp - (b - 0.5 * pam) / temp
    )
