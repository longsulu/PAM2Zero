from __future__ import annotations
from pam2zero import config as cfg
import numpy as np
import pandas as pd


def bootstrap_delta(delta, rng, reps):
    delta = np.asarray(delta, dtype=float)
    n = len(delta)
    est = []
    for start in range(0, reps, 500):
        est.extend(delta[rng.integers(0, n, size=(min(500, reps - start), n))].mean(1))
    lo, hi = np.quantile(est, [0.025, 0.975])
    if n <= 16:
        bits = np.arange(2 ** n)[:, None] >> np.arange(n) & 1
        null = ((2 * bits - 1) * delta).mean(1)
        p = float((np.abs(null) >= abs(delta.mean()) - 1e-15).mean())
    else:
        hits = 0
        for start in range(0, reps, 500):
            count = min(500, reps - start)
            null = ((2 * rng.integers(0, 2, (count, n)) - 1) * delta).mean(1)
            hits += int((np.abs(null) >= abs(delta.mean()) - 1e-15).sum())
        p = (hits + 1) / (reps + 1)
    return (float(lo), float(hi), float(p))


def statistical_summary(df, per):
    metrics_cols = [
        'Macro_AUPRC',
        'Macro_AP',
        'Macro_AUROC',
        'MCC_locked',
        'ACC_locked',
        'Recall_locked',
        'Precision_locked',
        'F1_locked',
        'Specificity_locked'] + [c for c in df if c.startswith((
            'Recall_at_',
            'Precision_at_',
            'Candidates_for_recall'
        ))
    ]
    agg = df.groupby(
        ['dataset', 'cohort', 'strategy'],
        sort=False).agg(n_runs=('seed', 'nunique'),
        N=('N', 'first'),
        Positive=('Positive', 'first'),
        Valid_sgRNA=('Valid_sgRNA', 'first'),
        **{k + '_' + f: (k, f) for k in metrics_cols for f in ['mean', 'std']}).reset_index(
    )
    agg.to_csv(cfg.OUT_DIR / 'summary_mean_std.csv', index=False)
    seven = df[df.dataset.isin(cfg.TEST_NAMES)].groupby([
        'strategy',
        'seed'
    ])[metrics_cols].mean().reset_index()
    seven.to_csv(cfg.OUT_DIR / 'seven_mean_per_seed.csv', index=False)
    seven.groupby('strategy')[metrics_cols].agg([
        'mean',
        'std',
        'min',
        'max'
    ]).to_csv(cfg.OUT_DIR / 'seven_stability.csv')
    comparisons = [
        ('training_competition_independent', 'Main_fullSelect_own', 'Ablation_ownTrain_ownSelect_own'),
            
            
            
        ('training_competition_budget_epoch', 'Main_fullSelect_own', 'Ablation_ownBudget_sameFullEpoch_own'),
            
            
            
        ('training_competition_budget_ownselect', 'Main_fullSelect_own', 'Ablation_ownBudget_ownSelect_own'),
            
            
            
        ('same_own_selection_training_effect', 'Control_fullTrain_ownSelect_own', 'Ablation_ownTrain_ownSelect_own'),
            
            
            
        ('readonly_remove_contest', 'Main_fullSelect_own', 'Control_fullSelect_full'),
        ('ours_vs_official', 'Main_fullSelect_own', 'Official_MD'),
        ('existing_vs_official', 'ExistingPAM01_frozen_own', 'Official_MD'),
        ('rerun_vs_existing', 'Main_fullSelect_own', 'ExistingPAM01_frozen_own')
    ]
    for arm in cfg.SIMPLE_ARMS:
        comparisons.append(('main_vs_' + arm, 'Main_fullSelect_own', cfg.SIMPLE_LABELS[arm]))
    for arm in cfg.COMPONENT_ARMS:
        comparisons += [
            ('component_' + arm + '_own', 'Main_fullSelect_own', 'Component_' + arm + '_fullSelect_own'),
                
                
                
            (
                'component_' + arm + '_full',
                'Control_fullSelect_full',
                'Component_' + arm + '_fullSelect_full'
            )
        ]
    rng = np.random.default_rng(cfg.BOOTSTRAP_SEED)
    stats = []
    seedrows = []
    deltarows = []
    for label, a, b in comparisons:
        for (ds, cohort), block in per.groupby(['dataset', 'cohort'], sort=False):
            aa = block[block.strategy.eq(a)]
            bb = block[block.strategy.eq(b)]
            if aa.empty or bb.empty:
                continue
            sa = set(aa.seed.astype(str))
            sb = set(bb.seed.astype(str))
            official = b == 'Official_MD'
            common = sa if official else sa & sb
            if not common:
                continue
            aa = aa[aa.seed.astype(str).isin(common)]
            if not official:
                bb = bb[bb.seed.astype(str).isin(common)]
            for metric in [
                'AUPRC',
                'AP',
                'AUROC',
                'MCC_locked',
                'F1_locked',
                'ACC_locked'] + [c for c in block if c.startswith((
                    'Recall_at_',
                    'Precision_at_',
                    'Candidates_for_recall'
                ))
            ]:
                am = aa.groupby('sgRNA')[metric].mean()
                bm = bb.groupby('sgRNA')[metric].mean()
                joint = pd.concat([am.rename('A'), bm.rename('B')], axis=1, join='inner').dropna()
                d = joint.A - joint.B
                if len(joint) != len(am) or len(joint) != len(bm):
                    raise ValueError('配对有效guide集合不同')
                lo, hi, p = bootstrap_delta(d.values, rng, cfg.BOOTSTRAP_REPS)
                stats.append(dict(
                    comparison=label,
                    dataset=ds,
                    cohort=cohort,
                    metric=metric,
                    A=a,
                    B=b,
                    n_guides=len(d),
                    n_paired_seeds=len(common),
                    higher_is_better=not metric.startswith('Candidates_for_'),
                    delta=float(d.mean()),
                    CI95_low=lo,
                    CI95_high=hi,
                    paired_signflip_p=p,
                    CI_unit='guide, seed-averaged; conditional on trained seeds'
                ))
                for guide, v in d.items():
                    deltarows.append(dict(
                        comparison=label,
                        dataset=ds,
                        cohort=cohort,
                        metric=metric,
                        sgRNA=guide,
                        delta=float(v)
                    ))
            for seed in sorted(common):
                av = aa[aa.seed.astype(str).eq(seed)].AUPRC.mean()
                bv = bb.AUPRC.mean() if official else bb[bb.seed.astype(str).eq(seed)].AUPRC.mean()
                seedrows.append(dict(
                    comparison=label,
                    dataset=ds,
                    cohort=cohort,
                    seed=seed,
                    delta_AUPRC=av - bv
                ))
    stats = pd.DataFrame(stats)
    if not stats.empty:
        stats['Holm_p_within_comparison_metric'] = np.nan
        for _, idx in stats.groupby(['comparison', 'metric']).groups.items():
            idx = list(idx)
            o = np.argsort(stats.loc[idx, 'paired_signflip_p'].values)
            p = stats.loc[idx, 'paired_signflip_p'].values[o]
            adjusted = np.minimum(1.0, np.maximum.accumulate(p * (len(p) - np.arange(len(p)))))
            for j, v in zip(o, adjusted):
                stats.at[idx[j], 'Holm_p_within_comparison_metric'] = v
    stats.to_csv(cfg.OUT_DIR / 'paired_guide_CI.csv', index=False)
    pd.DataFrame(seedrows).to_csv(cfg.OUT_DIR / 'paired_seed_deltas.csv', index=False)
    deltas = pd.DataFrame(deltarows)
    deltas.to_csv(cfg.OUT_DIR / 'paired_guide_deltas.csv', index=False)
    combined = []
    if not deltas.empty:
        sub = deltas[deltas.dataset.isin(cfg.TEST_NAMES) & deltas.metric.eq('AUPRC')]
        for label, block in sub.groupby('comparison'):
            if set(block.dataset) != set(cfg.TEST_NAMES):
                continue
            mat = block.pivot(
                index='sgRNA',
                columns='dataset',
                values='delta').reindex(columns=cfg.TEST_NAMES
            )
            present = mat.notna().to_numpy().astype(float)
            values = mat.fillna(0.0).to_numpy()
            n = len(mat)
            est = []
            attempts = 0
            while len(est) < cfg.BOOTSTRAP_REPS:
                count = min(500, cfg.BOOTSTRAP_REPS - len(est))
                w = rng.multinomial(n, np.full(n, 1 / n), size=count)
                denom = w @ present
                valid = (denom > 0).all(1)
                attempts += count
                if valid.any():
                    est.extend((w[valid] @ values / denom[valid]).mean(1))
                if attempts > cfg.BOOTSTRAP_REPS * 10:
                    raise RuntimeError('跨数据集guide重采样有效次数不足')
            lo, hi = np.quantile(est, [0.025, 0.975])
            combined.append(dict(
                comparison=label,
                Mean7_delta=float(mat.mean().mean()),
                CI95_low=lo,
                CI95_high=hi,
                unique_guides=n,
                unit='global guide clusters; equal dataset means; conditional on seeds'
            ))
    pd.DataFrame(combined).to_csv(cfg.OUT_DIR / 'Mean7_cluster_guide_CI.csv', index=False)
    print('\n五种子与官方对比：\n', agg.to_string(index=False), flush=True)
    print('\nMean7配对区间：\n', pd.DataFrame(combined).to_string(index=False), flush=True)


def additional_paired(per):
    rng = np.random.default_rng(cfg.BOOTSTRAP_SEED)
    rows = []
    comparisons = [
        (cfg.PRIMARY, 'Readonly_no_pam_own'),
        ('Control_fullSelect_full', 'Readonly_unit_edges_full'),
        ('Control_fullSelect_full', 'Readonly_constant_distance_full'),
        (cfg.PRIMARY, 'Official_M'),
        (cfg.PRIMARY, 'Official_D')
    ]
    for a, b in comparisons:
        for (dataset, cohort), p in per.groupby(['dataset', 'cohort']):
            aa = p[p.strategy.eq(a)].groupby('sgRNA')[['AUPRC', 'AP']].mean()
            bb = p[p.strategy.eq(b)].groupby('sgRNA')[['AUPRC', 'AP']].mean()
            if aa.empty or bb.empty:
                continue
            if set(aa.index) != set(bb.index):
                raise ValueError('配对guide集合不同')
            for metric in ['AUPRC', 'AP']:
                v = (aa[metric] - bb[metric]).to_numpy()
                lo, hi, pv = bootstrap_delta(v, rng, cfg.BOOTSTRAP_REPS)
                rows.append(dict(
                    A=a,
                    B=b,
                    dataset=dataset,
                    cohort=cohort,
                    metric=metric,
                    delta=v.mean(),
                    CI_low=lo,
                    CI_high=hi,
                    p=pv,
                    n_guides=len(v)
                ))
    d = pd.DataFrame(rows)
    if len(d):
        d['Holm_p'] = np.nan
        for _, idx in d.groupby(['A', 'B', 'metric']).groups.items():
            ix = np.array(list(idx))
            o = np.argsort(d.loc[ix, 'p'].to_numpy())
            p = d.loc[ix[o], 'p'].to_numpy()
            v = np.minimum(1, np.maximum.accumulate(p * (len(p) - np.arange(len(p)))))
            d.loc[ix[o], 'Holm_p'] = v
    d.to_csv(cfg.OUT_DIR / 'additional_readonly_official_pairs.csv', index=False)
