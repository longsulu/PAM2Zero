from __future__ import annotations
from pam2zero import config as cfg
import numpy as np
import pandas as pd
import torch


@torch.no_grad()
def export_parameters(model, name, seed):
    from pam2zero.utils.io import dump_json
    folder = cfg.OUT_DIR / 'parameters' / name / ('seed_%s' % seed)
    folder.mkdir(parents=True, exist_ok=True)
    decode = {v: k for k, v in cfg.ENCODE_DICT_36.items()}
    rows = []
    pam_positions = [22, 23] if model.matrix_pam.shape[0] == 2 else [21, 22, 23]
    for region, positions, a, b in [
        ('guide', list(range(1, 21)), model.matrix_d, model.initial_d),
        ('PAM', pam_positions, model.matrix_pam, model.initial_pam)
    ]:
        aa = a.cpu().numpy()
        bb = b.cpu().numpy()
        for row, pos in enumerate(positions):
            for code in cfg.ORDINARY_CODES:
                rows.append(dict(
                    region=region,
                    original_position=pos,
                    code=code,
                    pair=decode[code],
                    initial_raw_D=float(bb[row, code - 1]),
                    learned_raw_D=float(aa[row, code - 1]),
                    delta=float(aa[row, code - 1] - bb[row, code - 1])
                ))
    pd.DataFrame(rows).to_csv(folder / 'D_and_PAM_changes.csv', index=False)
    pd.DataFrame(dict(edge=np.arange(1, 20), gain=model.edge_gains().cpu().numpy())).to_csv(
        folder / 'edge_gains.csv',
        index=False
    )
    pd.DataFrame(dict(
        distance=np.arange(1, 20),
        strength=model.distance_strengths().cpu().numpy()
    )).to_csv(
        folder / 'distance_strengths.csv',
        index=False
    )
    dump_json(
        folder / 'scalars.json',
        dict(
            match_gain=float(model.team_gains()[0]),
            mismatch_gain=float(model.team_gains()[1]),
            temperature=float(model.temperature()),
            raw_D_is_not_physical_force=True
        )
    )


@torch.no_grad()
def _export_interpretability(tr, support):
    from pam2zero.data.encoding import encode_pair
    from pam2zero.utils.io import json_write
    from pam2zero.evaluation.inference import read_model, score_branch
    out = cfg.OUT_DIR / 'interpretation'
    out.mkdir(exist_ok=True)
    decode = {v: k for k, v in cfg.ENCODE_DICT_36.items()}
    letters = np.array([decode[i][0] for i in range(1, 37)])
    guide_strings = np.array(
        [''.join(row) for row in letters[tr[:, :20].astype(int) - 1]],
        dtype=object
    )
    guides = sorted(set(guide_strings))[:cfg.INTERPRET_GUIDES]
    json_write(
        out / 'probe_design.json',
        dict(
            guides=guides,
            selection='lexicographically first unique CHANGEseq guides',
            PAM='TGG',
            positions='1..20 forward input; position20 proximal PAM',
            finite_difference='z(double)-z(single_i)-z(single_j)+z(WT)',
            not_measured_force=True,
            background_count=len(guides)
        )
    )
    singles = []
    doubles = []
    effective = []
    pamrows = []
    edge = []
    dist = []
    param = []
    for seed in cfg.SEEDS:
        cp = cfg.OUT_DIR / 'training/full' / ('seed_%s' % seed) / 'best_full_TTISS.pt'
        m, meta = read_model(cp)
        export_parameters(m, 'Main833', seed)
        d = m.matrix_d.detach().cpu().numpy()
        initial = m.initial_d.cpu().numpy()
        gain = m.team_gains().cpu().numpy()
        temp = float(m.temperature())
        scale = float(m.input_scale)
        actual = (np.logaddexp(0, d / scale) + cfg.FORCE_FLOOR) * np.where(
            np.isin(np.arange(1, 37), cfg.MATCH_CODES)[None],
            gain[0],
            -gain[1]
        ) / temp
        for p in range(20):
            for c in cfg.ORDINARY_CODES:
                effective.append(dict(
                    seed=seed,
                    position=p + 1,
                    pair=decode[c],
                    code=c,
                    support=int(support[p, c - 1]),
                    initial_D=float(initial[p, c - 1]),
                    learned_D=float(d[p, c - 1]),
                    own_logit_contribution=float(actual[p, c - 1])
                ))
        for p in range(2):
            for c in cfg.ORDINARY_CODES:
                pamrows.append(dict(
                    seed=seed,
                    pam_position=p + 2,
                    code=c,
                    pair=decode[c],
                    support=int(support[21 + p, c - 1]),
                    contribution=float(m.matrix_pam[p, c - 1]) / scale / temp
                ))
        for i, v in enumerate(m.edge_gains().cpu().numpy()):
            edge.append(dict(seed=seed, edge=i + 1, gain=float(v)))
        for i, v in enumerate(m.distance_strengths().cpu().numpy()):
            dist.append(dict(seed=seed, distance=i + 1, strength=float(v)))
        for region, table in [('guide', m.matrix_d), ('PAM2', m.matrix_pam)]:
            param.append(dict(
                seed=seed,
                region=region,
                allocated=table.numel(),
                observed_entries=int((support[:20] if region == 'guide' else support[21:23]).astype(bool).sum())
            ))
        for guide in guides:
            wt = np.array(encode_pair(guide + 'TGG', guide + 'TGG'), dtype=np.uint8)
            codes = [wt.copy()]
            sm = []
            dm = []
            for p in range(20):
                for alt in 'ACGT':
                    if alt == guide[p]:
                        continue
                    x = wt.copy()
                    x[p] = cfg.ENCODE_DICT_36[guide[p] + alt]
                    sm.append((p, alt, len(codes)))
                    codes.append(x)
            chosen = {p: next((v for v in sm if v[0] == p)) for p in range(20)}
            for i in range(20):
                for j in range(i + 1, 20):
                    x = wt.copy()
                    x[i] = codes[chosen[i][2]][i]
                    x[j] = codes[chosen[j][2]][j]
                    dm.append((i, j, len(codes)))
                    codes.append(x)
            x = np.array(codes, dtype=np.uint8)
            ss = {b: score_branch(m, x, b) for b in ['own', 'full']}
            for b, s in ss.items():
                for p, alt, idx in sm:
                    singles.append(dict(
                        seed=seed,
                        guide=guide,
                        branch=b,
                        position=p + 1,
                        reference=guide[p],
                        alternate=alt,
                        delta_logit=float(s[idx] - s[0])
                    ))
                for i, j, idx in dm:
                    doubles.append(dict(
                        seed=seed,
                        guide=guide,
                        branch=b,
                        position_i=i + 1,
                        position_j=j + 1,
                        distance=j - i,
                        synergy=float(s[idx] - s[chosen[i][2]] - s[chosen[j][2]] + s[0])
                    ))
        del m
    for name, data in [
        ('effective_node_contributions', effective),
        ('PAM2_contributions', pamrows),
        ('edge_gains', edge),
        ('distance_strengths', dist),
        ('single_counterfactuals', singles),
        ('pair_finite_differences', doubles),
        ('observed_parameter_support', param)
    ]:
        pd.DataFrame(data).to_csv(out / (name + '.csv'), index=False)
    dd = pd.DataFrame(doubles)
    dd.groupby([
        'branch',
        'distance']).synergy.agg(['mean',
        'std',
        lambda s: s.abs().mean()
    ]).to_csv(out / 'synergy_by_distance.csv')
    q = dd[dd.branch.eq('full')].pivot(
        index=['guide', 'position_i', 'position_j'],
        columns='seed',
        values='synergy'
    )
    pd.DataFrame(dict(
        abs_mean=q.abs().mean(1),
        all_seed_positive=(q > 0.0001).all(1),
        all_seed_negative=(q < -0.0001).all(1)
    )).to_csv(out / 'synergy_seed_sign_consistency.csv')


def interpretation_suite(tr, support):
    from pam2zero.evaluation.inference import read_model, score_own_float64
    _export_interpretability(tr, support)
    out = cfg.OUT_DIR / 'interpretation'
    rows = []
    for seed in cfg.SEEDS:
        m, cp = read_model(cfg.OUT_DIR / 'training/full' / ('seed_%s' % seed) / 'best_full_TTISS.pt')
        x = tr[:min(len(tr), 256)].copy()
        ref = (x[:, :20].astype(int) - 1) // 6 * 6 + 1
        base = x.copy()
        base[:, :20] = ref
        for i, j in [(0, 1), (3, 12), (0, 19), (10, 19)]:
            a = base.copy()
            b = base.copy()
            ab = base.copy()
            ca = (ref[:, i] - 1) // 6 * 6 + 2
            cb = (ref[:, j] - 1) // 6 * 6 + 2
            a[:, i] = ca
            b[:, j] = cb
            ab[:, i] = ca
            ab[:, j] = cb
            v = [score_own_float64(m, q) for q in [base, a, b, ab]]
            diff = v[3] - v[1] - v[2] + v[0]
            rows.append(dict(
                seed=seed,
                i=i + 1,
                j=j + 1,
                max_abs_own_interaction=float(abs(diff).max())
            ))
            if abs(diff).max() > 1e-09:
                raise AssertionError('OWN additivity check failed')
    pd.DataFrame(rows).to_csv(out / 'OWN_exact_additivity_audit.csv', index=False)
