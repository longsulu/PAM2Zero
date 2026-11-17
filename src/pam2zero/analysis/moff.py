from __future__ import annotations
from pam2zero import config as cfg
from collections import Counter
import argparse
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import re
from scipy.stats import spearmanr
import torch


def read_native_moff(path, args, out):
    from pam2zero.utils.io import file_hash, json_write
    if path.suffix.lower() != '.xlsx':
        return None
    if any((getattr(
        args,
        k,
        None) for k in ('guide_column',
        'target_column',
        'ratio_column',
        'condition_column'
    ))):
        return None
    xls = pd.ExcelFile(path)
    sheet = 'Tandem_Libs_T1-3'
    if sheet not in xls.sheet_names or (args.sheet and args.sheet != sheet):
        return None
    raw = pd.read_excel(xls, sheet_name=sheet, header=[0, 1])
    expected = [
        'sgRNA sequence',
        'On-target sequence',
        'Off-target sequence',
        'Mismatch type',
        'Read counts of five editing types',
        'Read counts of five editing types',
        'Read counts of five editing types',
        'Read counts of five editing types',
        'Read counts of five editing types',
        'MLE estimated reads for four cleavage states',
        'MLE estimated reads for four cleavage states',
        'MLE estimated reads for four cleavage states',
        'MLE estimated reads for four cleavage states',
        'Inde rate at off-target',
        'Indel rate at on-target',
        'off-on-ratio',
        'Library'
    ]
    if list(raw.columns.get_level_values(0)) != expected:
        raise ValueError('MOFF native sheet layout changed. Expected verified17-column dual-header table; please send original file.')
    if raw.columns[11][1] != 'Right cleavage' or raw.columns[12][1] != 'Both cleavage':
        raise ValueError('MOFF cleavage read columns changed')
    rows = []
    audit = []
    for idx, values in enumerate(raw.itertuples(index=False, name=None)):
        guide = normalize_sequence(values[0])
        reference = normalize_sequence(values[1])
        target = normalize_sequence(values[2])
        reason = 'kept'
        leader = False
        if len(reference) != 23 or len(target) != 23 or any((x not in 'ACGT' for x in reference + target)):
            reason = 'noncanonical_reference_or_target'
        else:
            core = reference[:20]
            if guide == core:
                pass
            elif len(guide) == 21 and guide[0] == 'G' and (guide[1:] == core):
                leader = True
            else:
                reason = 'guide_does_not_match_reference20_or_verified_leading_G'
        try:
            reads = float(values[11]) + float(values[12])
            ratio = float(values[15])
        except (ValueError, TypeError):
            reads = float('nan')
            ratio = float('nan')
        if not np.isfinite(reads) or reads < 100:
            reason = 'on_target_edited_reads_below100'
        elif not np.isfinite(ratio) or ratio < 0:
            reason = 'missing_or_invalid_reported_ratio'
        annotation = str(values[3])
        label = re.match('M([123]):', annotation)
        if reason == 'kept' and reference[20:] != target[20:]:
            reason = 'PAM_mutation_excluded_for_guide_interaction'
        if reason == 'kept':
            observed = [20 - j for j in range(20) if reference[j] != target[j]]
            annotated = list(map(int, re.findall('p(\\d+)r', annotation)))
            if label is None or len(observed) != int(label.group(1)) or sorted(observed) != sorted(annotated):
                reason = 'mismatch_count_or_position_annotation_disagrees'
        audit.append(dict(
            source_row=idx + 3,
            reason=reason,
            extra_leading_G_removed=leader,
            on_target_edited_reads=reads,
            guide_original=guide,
            reference=reference,
            target=target,
            reported_ratio=ratio,
            annotation=annotation
        ))
        if reason == 'kept':
            rows.append(dict(
                guide=reference,
                target=target,
                off_on_ratio=ratio,
                condition=str(values[16]),
                source_row=idx + 3
            ))
    pd.DataFrame(audit).to_csv(out / 'native_input_audit.csv', index=False, encoding='utf-8-sig')
    json_write(
        out / 'column_mapping.json',
        dict(
            adapter='MOFF_SuppData1_verified_v2',
            sheet=sheet,
            source_SHA256=file_hash(path),
            input_rows=len(raw),
            native_kept_rows=len(rows),
            source_url='https://pmc-oa-opendata.s3.amazonaws.com/PMC8789861.1/41467_2022_28028_MOESM4_ESM.xlsx',
                
                
            reason_counts=dict(Counter((r['reason'] for r in audit))),
            guide='On-target sequence: verified guide20 plus real reference PAM',
            target='Off-target sequence',
            ratio='off-on-ratio (reported, NOT recalculated)',
            condition='Library',
            quality_filter='MLE Right cleavage + Both cleavage >=100',
            leader='Only strip extra leading G when remaining20 exactly equals reference20',
            note='This workbook is audited as supplied; its row count is not forced to match counts in manuscript text.'
        )
    )
    if not rows:
        raise ValueError('No native MOFF rows pass checks. See native_input_audit.csv.')
    print(
        'MOFF native table:',
        len(raw),
        'rows;',
        len(rows),
        'with qualified reported off/on ratios; no assumed PAM.',
        flush=True
    )
    return pd.DataFrame(rows)


def normalize_sequence(v):
    if pd.isna(v):
        return ''
    return re.sub('\\s+', '', str(v)).upper().replace('U', 'T')


def clean_measurements(df, pam, out):
    from pam2zero.utils.io import json_write
    rows = []
    excluded = []
    counts = Counter()
    for row in df.to_dict('records'):
        g = normalize_sequence(row['guide'])
        t = normalize_sequence(row['target'])
        reason = None
        condition = str(row['condition']).strip()
        if pd.isna(row['condition']) or not condition:
            reason = 'missing_condition'
        try:
            r = float(row['off_on_ratio'])
        except (ValueError, TypeError):
            r = float('nan')
        if not np.isfinite(r) or r < 0:
            reason = 'invalid_or_missing_ratio'
        if len(g) == 20 and pam:
            g += pam
            counts['guide_PAM_explicitly_assumed'] += 1
        if len(t) == 20 and pam:
            t += pam
            counts['target_PAM_explicitly_assumed'] += 1
        if len(g) != 23 or len(t) != 23:
            reason = 'needs_guide23_and_target23'
        elif any((c not in 'ACGT' for c in g + t)):
            reason = 'ambiguous_or_gap_bases'
        elif g[20:] != t[20:]:
            reason = 'PAM_differs_from_reference'
        if reason:
            excluded.append(dict(row, reason=reason))
            continue
        dif = tuple(((i, t[i]) for i in range(20) if g[i] != t[i]))
        rows.append(dict(
            guide=g,
            target=t,
            off_on_ratio=r,
            condition=condition,
            source_row=row['source_row'],
            n_mismatch=len(dif)
        ))
    if not rows:
        pd.DataFrame(excluded).to_csv(out / 'excluded_rows.csv', index=False, encoding='utf-8-sig')
        raise ValueError('没有合格序列。查看 excluded_rows.csv；需要明确23nt序列，20nt须显式指定已核实的 --pam。')
    frame = pd.DataFrame(rows)
    keys = ['condition', 'guide', 'target']
    conflict = frame.groupby(keys, dropna=False)['off_on_ratio'].transform('nunique') > 1
    for row in frame[conflict].to_dict('records'):
        excluded.append(dict(row, reason='discordant_duplicate_measurements'))
    counts['discordant_duplicate_rows'] = int(conflict.sum())
    frame = frame[~conflict].copy()
    dup = frame.duplicated(keys)
    counts['identical_duplicate_rows_removed'] = int(dup.sum())
    frame = frame[~dup].copy()
    counts['input_rows'] = len(df)
    counts['kept_unique_rows'] = len(frame)
    counts.update({'excluded_' + k: v for k, v in Counter((x['reason'] for x in excluded)).items()})
    pd.DataFrame(excluded).to_csv(out / 'excluded_rows.csv', index=False, encoding='utf-8-sig')
    frame.to_csv(out / 'clean_measurements.csv', index=False, encoding='utf-8-sig')
    json_write(out / 'cleaning_audit.json', dict(counts))
    return frame


def assemble_pairs(frame, out=None):
    lookup = {(r.condition, r.guide, r.target): r for r in frame.itertuples()}
    rows = []
    missing = []
    for r in frame.itertuples():
        if r.n_mismatch != 2:
            continue
        pos = [i for i in range(20) if r.guide[i] != r.target[i]]
        i, j = pos
        a = list(r.guide)
        a[i] = r.target[i]
        a = ''.join(a)
        b = list(r.guide)
        b[j] = r.target[j]
        b = ''.join(b)
        ra = lookup.get((r.condition, r.guide, a))
        rb = lookup.get((r.condition, r.guide, b))
        if ra is None or rb is None:
            missing.append(dict(
                guide=r.guide,
                target=r.target,
                condition=r.condition,
                missing_A=ra is None,
                missing_B=rb is None
            ))
            continue
        rows.append(dict(
            guide=r.guide,
            condition=r.condition,
            ref=r.guide,
            A=a,
            B=b,
            AB=r.target,
            r_A=ra.off_on_ratio,
            r_B=rb.off_on_ratio,
            r_AB=r.off_on_ratio,
            pos_i=i + 1,
            pos_j=j + 1,
            distance=j - i,
            pam_relative_i=20 - i,
            pam_relative_j=20 - j,
            type_A=r.guide[i] + r.target[i],
            type_B=r.guide[j] + r.target[j],
            row_A=ra.source_row,
            row_B=rb.source_row,
            row_AB=r.source_row
        ))
    if out is not None:
        pd.DataFrame(missing).to_csv(
            out / 'missing_single_mutants.csv',
            index=False,
            encoding='utf-8-sig'
        )
    if not rows:
        raise ValueError('没有完整 A/B/AB 成套测量。需 MOFF Supplementary Data 1 的定量数据；不能用模型补全实测单错配。')
    pairs = pd.DataFrame(rows)
    pairs['observed_raw_residual'] = pairs.r_AB - pairs.r_A * pairs.r_B
    positive = (pairs[['r_A', 'r_B', 'r_AB']] > 0).all(axis=1)
    pairs['observed_log_CE'] = np.nan
    pairs.loc[
        positive,
        'observed_log_CE'] = np.log(pairs.loc[
            positive,
            'r_AB']) - np.log(pairs.loc[positive,
            'r_A']) - np.log(pairs.loc[positive,
            'r_B'
        ]
    )
    pairs['all_three_ge_002'] = (pairs[['r_A', 'r_B', 'r_AB']] >= 0.02).all(axis=1)
    pairs['observed_log_CE_eps001'] = np.log(pairs.r_AB + 0.01) - np.log(pairs.r_A + 0.01) - np.log(pairs.r_B + 0.01)
    return pairs


def correlations(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if len(x) < 4:
        return dict(n=len(x), rho=np.nan, pearson=np.nan, status='too_few_pairs')
    if np.ptp(x) < 1e-10 or np.ptp(y) < 1e-10:
        return dict(
            n=len(x),
            rho=np.nan,
            pearson=np.nan,
            status='constant_prediction_or_observation'
        )
    return dict(
        n=len(x),
        rho=float(spearmanr(x, y)[0]),
        pearson=float(pearsonr(x, y)[0]),
        status='ok'
    )


def cluster_ci(df, x, y, repeats):
    base = correlations(df[x], df[y])
    base.update(
        n_guides=int(df.guide.nunique()),
        rho_CI_low=np.nan,
        rho_CI_high=np.nan,
        bootstrap_valid=0
    )
    if base['status'] != 'ok' or base['n_guides'] < 3 or repeats <= 0:
        return base
    groups = [v.to_numpy() for v in df.reset_index(drop=True).groupby('guide').groups.values()]
    xx = df[x].to_numpy()
    yy = df[y].to_numpy()
    rng = np.random.RandomState(2026)
    vals = []
    for _ in range(repeats):
        ix = np.concatenate([groups[i] for i in rng.randint(len(groups), size=len(groups))])
        r = correlations(xx[ix], yy[ix])['rho']
        if np.isfinite(r):
            vals.append(r)
    if vals:
        lo, hi = np.quantile(vals, [0.025, 0.975])
        base.update(rho_CI_low=float(lo), rho_CI_high=float(hi), bootstrap_valid=len(vals))
    return base


def final_moff_audit():
    from pam2zero.utils.io import file_hash, json_write
    from pam2zero.evaluation.inference import read_model, score_own_float64
    from pam2zero.models.competition import typed_source_scores
    files = sorted((cfg.ROOT / 'data').rglob('41467_2022_28028_MOESM4_ESM*.xlsx'))
    if len(files) != 1:
        raise FileNotFoundError('Need one original MOFF Supplementary Data1 workbook under data; found ' + str(len(files)))
    out = cfg.OUT_DIR / 'MOFF_interaction_audit'
    out.mkdir(exist_ok=True)
    args = argparse.Namespace(
        sheet=None,
        guide_column=None,
        target_column=None,
        ratio_column=None,
        condition_column=None
    )
    frame = clean_measurements(read_native_moff(files[0], args, out), None, out)
    pairs = assemble_pairs(frame, out)
    allrows = []
    for seed in cfg.SEEDS:
        m, _ = read_model(cfg.OUT_DIR / 'training/full' / ('seed_%s' % seed) / 'best_full_TTISS.pt')
        m = m.double()
        raw = []
        for i in range(0, len(pairs), 1024):
            block = pairs.iloc[i:i + 1024]
            x = np.array(
                [[cfg.ENCODE_DICT_36[a + b] for a, b in zip(r.guide, getattr(r, c))] for r in block.itertuples() for c in ['ref', 'A', 'B', 'AB']],
                    
                    
                    
                dtype=np.uint8
            )
            with torch.no_grad():
                v = (typed_source_scores(
                    m,
                    torch.as_tensor(
                        x,
                        dtype=torch.long,
                        device=cfg.DEVICE))['full'] if cfg.SOURCE_NUMERICS == 'typed_vector' else m.contributions(torch.as_tensor(
                            x,
                            
                        dtype=torch.long,
                        device=cfg.DEVICE
                    ))['score'
                ]).cpu().numpy().reshape(
                    -1,
                    4
                )
            own = score_own_float64(m, x).reshape(-1, 4)
            q = block.copy()
            q['seed'] = seed
            q['full_interaction'] = v[:, 3] - v[:, 1] - v[:, 2] + v[:, 0]
            e = own[:, 3] - own[:, 1] - own[:, 2] + own[:, 0]
            if abs(e).max() > 1e-09:
                raise AssertionError('MOFF OWN additive invariant')
            q['own_interaction'] = 0.0
            q['own_activity'] = own[:, 3]
            q['full_activity'] = v[:, 3]
            raw.append(q)
        allrows.append(pd.concat(raw, ignore_index=True))
    results = pd.concat(allrows, ignore_index=True)
    results.to_csv(out / 'per_seed_paired_predictions.csv', index=False)
    keys = [
        'guide',
        'condition',
        'AB',
        'distance',
        'all_three_ge_002',
        'observed_log_CE',
        'observed_raw_residual',
        'r_AB'
    ]
    mean = results.groupby(
        keys,
        dropna=False)[[
            'full_interaction',
            'own_interaction',
            'own_activity',
            'full_activity'
        ]].mean().reset_index(
    )
    mean.to_csv(out / 'seed_averaged_paired_predictions.csv', index=False)
    rows = []
    for view in ['own', 'full']:
        for scope, x, y, mask in [
            ('primary_log_CE', view + '_interaction', 'observed_log_CE', mean.all_three_ge_002),
            ('raw_residual', view + '_interaction', 'observed_raw_residual', np.ones(len(mean), bool)),
                
                
                
            ('activity_only_NOT_interaction', view + '_activity', 'r_AB', np.ones(len(mean), bool))
        ]:
            rows.append(dict(
                view=view,
                scope=scope,
                **cluster_ci(mean.loc[mask], x, y, cfg.BOOTSTRAP_REPS)
            ))
    pd.DataFrame(rows).to_csv(out / 'seed_averaged_summary.csv', index=False)
    json_write(
        out / 'protocol.json',
        dict(
            source=str(files[0]),
            sha256=file_hash(files[0]),
            neural_training=False,
            seeds=cfg.SEEDS,
            bootstrap_unit='guide after seed-averaging',
            independence='baseline prior/training overlap with MOFF not certified',
            physical_force_measurement=False
        )
    )
