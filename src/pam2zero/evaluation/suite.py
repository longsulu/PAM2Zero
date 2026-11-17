from __future__ import annotations
from pam2zero import config as cfg
from pathlib import Path
import hashlib
import math
import numpy as np
import pandas as pd
import torch


@torch.no_grad()
def load_spec(spec):
    from pam2zero.evaluation.inference import read_model
    m, cp = read_model(spec['path'])
    condition = spec.get('intervention')
    if condition == 'no_pam':
        m.use_pam = False
    elif condition == 'unit_edges':
        m.edge_raw.zero_()
    elif condition == 'constant_distance':
        m.raw_distance.fill_(math.log(cfg.CONTEST_INIT / (cfg.CONTEST_MAX - cfg.CONTEST_INIT)))
    return (m, cp)


def annotation_keys(codes):
    return np.array(
        [hashlib.sha256(bytes(r)).hexdigest() for r in np.asarray(codes, dtype=np.uint8)],
        dtype=object
    )


def write_error_scores(dataset, codes, y, g, scores, thresholds, kept=None):
    if not cfg.SAVE_SAMPLE_PREDICTIONS:
        return
    folder = cfg.OUT_DIR / 'sample_predictions'
    folder.mkdir(exist_ok=True)
    keys = annotation_keys(codes)
    mm = (~np.isin(codes[:, :20], cfg.MATCH_CODES)).sum(1)
    pam = np.where(np.isin(codes[:, 21:23], cfg.MATCH_CODES), '1', '0')
    patterns = np.char.add(pam[:, 0], pam[:, 1])
    for seed in cfg.SEEDS:
        key = (cfg.PRIMARY, str(seed))
        v = scores[key]
        df = pd.DataFrame(dict(
            sample_id=np.arange(len(y)),
            input_key=keys,
            sgRNA=g,
            label=y,
            mismatch_count=mm,
            PAM_last2_pattern=patterns,
            logit=v,
            TTISS_threshold=thresholds[key],
            predicted_positive=v >= thresholds[key]
        ))
        if kept is not None:
            for col in kept:
                if col.startswith('source_') or col in ('sgRNA', 'DNA', 'csv_row'):
                    df['metadata_' + col] = kept[col].to_numpy()
        df.to_csv(
            folder / (dataset + '_seed_' + str(seed) + '.csv.gz'),
            index=False,
            compression='gzip'
        )


def stratified_report(dataset, codes, y, g, scores, rows):
    from pam2zero.evaluation.metrics import compute_metrics_paper
    bits = np.isin(codes[:, 20:23], cfg.MATCH_CODES)
    mm = (~np.isin(codes[:, :20], cfg.MATCH_CODES)).sum(1)
    masks = {
        **{'PAM_' + ''.join(map(str, p)): np.all(bits == np.array(p, bool)[None], axis=1) for p in [(0, 1, 1), (1, 1, 1)]},
            
            
            
        'MM_1_2': mm <= 2,
        'MM_3_4': (mm >= 3) & (mm <= 4),
        'MM_5plus': mm >= 5
    }
    for name, mask in masks.items():
        if not mask.any():
            continue
        for (strategy, seed), v in scores.items():
            if strategy not in (cfg.PRIMARY, 'Control_fullSelect_full', 'Official_MD'):
                continue
            try:
                m, _ = compute_metrics_paper(y[mask], v[mask], g[mask])
            except RuntimeError:
                continue
            rows.append(dict(
                dataset=dataset,
                stratum=name,
                strategy=strategy,
                seed=seed,
                N=int(mask.sum()),
                Positive=int(y[mask].sum()),
                **m
            ))


def evaluation_suite(specs, val, vy, vg):
    from pam2zero.evaluation.inference import cached_score, dataset_token, predict_official
    from pam2zero.utils.io import json_write
    from pam2zero.data.loaders import load_test_csv_original, overlap_cohorts, prepare_dataset, reference_guides
    from pam2zero.experiments.paper_suite import log_problem, optional
    from pam2zero.evaluation.metrics import score_and_report, threshold_mcc
    official = None
    paths = [cfg.ROOT / 'models/M_models/M_model_0.pth', cfg.ROOT / 'models/D_models/D_model_0.pth']
    if all((p.is_file() for p in paths)):
        official = optional(
            '官方TorchScript加载',
            lambda: tuple((torch.jit.load(str(p), map_location=cfg.DEVICE).eval() for p in paths))
        )
    else:
        log_problem(
            '官方TorchScript加载',
            FileNotFoundError('缺少models/M_models/M_model_0.pth或models/D_models/D_model_0.pth')
        )
    officials = [dict(
        strategy=n,
        seed='released0') for n in ['Official_M',
        'Official_D',
        'Official_MD'
    ]] if official else []
    allspecs = specs + officials
    thresholds = {}
    trows = []
    token = dataset_token(val)
    for spec in specs:
        v = cached_score(spec, val, token)
        key = (spec['strategy'], str(spec['seed']))
        thresholds[key] = threshold_mcc(vy, v)
    if official:
        v = predict_official(*official, val, cfg.OFFICIAL_BATCH_SIZE)
        for n, k in [
            ('Official_M', 'CrisprPr_M'),
            ('Official_D', 'CrisprPr_D'),
            ('Official_MD', 'CrisprPr_ensemble')
        ]:
            thresholds[n, 'released0'] = threshold_mcc(vy, v[k])
    for (strategy, seed), threshold in thresholds.items():
        trows.append(dict(
            strategy=strategy,
            seed=seed,
            threshold=threshold,
            source='TTISS pooled MCC only; >=; no test fitting'
        ))
    pd.DataFrame(trows).to_csv(cfg.OUT_DIR / 'TTISS_locked_thresholds.csv', index=False)
    rows = []
    pers = []
    strata = []

    def evaluate(name, codes, y, g, masks, kept=None):
        from pam2zero.evaluation.inference import cached_score, dataset_token, predict_official
        from pam2zero.evaluation.metrics import score_and_report
        token = dataset_token(codes)
        ss = {(s['strategy'], str(s['seed'])): cached_score(s, codes, token) for s in specs}
        for seed in cfg.SEEDS:
            for condition in ('unit_edges', 'constant_distance'):
                k = ('Readonly_' + condition + '_own', str(seed))
                if k in ss:
                    np.testing.assert_array_equal(ss[k], ss[cfg.PRIMARY, str(seed)])
        if official:
            oo = predict_official(*official, codes, cfg.OFFICIAL_BATCH_SIZE)
            for n, k in [
                ('Official_M', 'CrisprPr_M'),
                ('Official_D', 'CrisprPr_D'),
                ('Official_MD', 'CrisprPr_ensemble')
            ]:
                ss[n, 'released0'] = oo[k]
        for cohort, mask in masks.items():
            score_and_report(name, cohort, codes, y, g, allspecs, thresholds, rows, pers, ss, mask)
        stratified_report(name, codes, y, g, ss, strata)
        write_error_scores(name, codes, y, g, ss, thresholds, kept)
        pd.DataFrame(rows).to_csv(cfg.OUT_DIR / 'all_run_metrics.csv', index=False)
        pd.concat(
            pers,
            ignore_index=True).to_csv(cfg.OUT_DIR / 'per_guide_metrics.csv',
            index=False
        )
        print('[评估完成]', name, flush=True)
    for name in cfg.TEST_NAMES:
        x, y, g = load_test_csv_original(cfg.TEST_DIR / (name + '.csv'))
        valid = sum((len(np.unique(y[g == a])) == 2 for a in np.unique(g)))
        if valid != cfg.EXPECTED_VALID_SGRNA[name]:
            raise ValueError(name + '有效guide数与原协议不符')
        evaluate(name, x, y, g, {'original_protocol': np.ones(len(y), bool)})
    if cfg.RUN_EXTERNAL:
        refs = optional('TrueOT相似guide参照加载', lambda: reference_guides(cfg.ROOT))
        refs = refs[0] if refs else {}
        candidates = sorted((p for p in cfg.TEST_DIR.rglob('*') if p.suffix.lower() in (
            '.csv',
            '.xlsx',
            '.xlsm') and (not p.name.startswith('~$')) and ('trueot' in p.name.lower() or 'table_s2' in p.name.lower()
        )))
        seen = {}
        discovery = []
        for p in candidates:

            def external_one():
                from pam2zero.data.loaders import overlap_cohorts, prepare_dataset
                sub = cfg.OUT_DIR / 'external_audit' / p.stem
                x, y, g, kept = prepare_dataset(p, sub)
                sig = hashlib.sha256('\n'.join(sorted((
                    k + '|' + str(int(v)) for k,
                    v in zip(kept.input_key, y)
                ))).encode()).hexdigest()
                if sig in seen:
                    discovery.append(dict(file=str(p), duplicate_of=seen[sig]))
                    return
                seen[sig] = str(p)
                discovery.append(dict(file=str(p), status='evaluated'))
                masks = overlap_cohorts(g, refs, sub)
                kept.to_csv(sub / 'kept_rows.csv', index=False)
                evaluate('TrueOT_' + p.stem, x, y, g, masks, kept)
            optional('外部样本表 ' + p.name, external_one)
        json_write(cfg.OUT_DIR / 'external_discovery.json', discovery)
        if not candidates:
            log_problem('TrueOT', FileNotFoundError('测试集目录未找到TrueOT CSV/XLSX'))
    pd.DataFrame(strata).to_csv(cfg.OUT_DIR / 'stratified_metrics.csv', index=False)
    return (pd.DataFrame(rows), pd.concat(pers, ignore_index=True))


def error_analysis():
    from pam2zero.evaluation.statistics import bootstrap_delta
    p = Path(cfg.ERROR_ANNOTATIONS) if cfg.ERROR_ANNOTATIONS else cfg.ROOT / 'data/error_annotations.csv'
    if not p.is_file():
        raise FileNotFoundError('未提供data/error_annotations.csv；逐样本预测已导出，按INPUT_GUIDE.txt回连后重跑--eval-only。')
    ann = pd.read_csv(p)
    required = {'dataset', 'sample_id', 'input_key', 'accessibility', 'cell_type'}
    if not required.issubset(ann):
        raise ValueError('表观注释缺列：' + str(required - set(ann)))
    if ann.duplicated(['dataset', 'sample_id']).any():
        raise ValueError('同dataset/sample_id有多条细胞注释，必须先明确实验对应细胞，不能跨细胞混合')
    ann['accessibility'] = pd.to_numeric(ann.accessibility, errors='coerce')
    if ann.cell_type.isna().any():
        raise ValueError('细胞类型缺失，不能当作对应实验的开放数据')
    result = []
    matched = []
    out = cfg.OUT_DIR / 'error_analysis'
    out.mkdir(exist_ok=True)
    for dataset, a in ann.groupby('dataset'):
        parts = []
        for seed in cfg.SEEDS:
            path = cfg.OUT_DIR / 'sample_predictions' / (str(dataset) + '_seed_' + str(seed) + '.csv.gz')
            if not path.is_file():
                raise FileNotFoundError(str(path))
            d = pd.read_csv(path)
            parts.append(d.assign(seed=seed))
        pred = pd.concat(parts, ignore_index=True)
        checks = pred.groupby('sample_id').agg(
            keys=('input_key', 'nunique'),
            labels=('label', 'nunique'),
            guides=('sgRNA', 'nunique')
        )
        if (checks != 1).any().any():
            raise ValueError('不同seed的样本身份不一致')
        d = pred.groupby(
            'sample_id',
            as_index=False).agg(input_key=('input_key', 'first'),
            label=('label', 'first'),
            sgRNA=('sgRNA', 'first'),
            mismatch_count=('mismatch_count', 'first'),
            PAM_last2_pattern=('PAM_last2_pattern', 'first'),
            logit=('logit', 'mean'),
            positive_vote=('predicted_positive', 'mean')
        )
        d = d.merge(a, on=['sample_id', 'input_key'], how='inner', validate='one_to_one')
        if len(d) != len(a):
            raise ValueError('注释sample_id/input_key不能完整匹配 ' + str(dataset))
        d['error_group'] = np.where(d.label.eq(1), 'positive_label', 'negative_label')
        d['score_bin'] = d.groupby('sgRNA').logit.transform(lambda x: pd.qcut(
            x.rank(method='average', pct=True),
            q=5,
            labels=False,
            duplicates='drop'
        ))
        high = d[d.positive_vote >= 0.5].dropna(subset=['accessibility']).copy()
        for (
            guide,
            cell,
            binid,
            mm,
            pam), block in high.groupby([
                'sgRNA',
                'cell_type',
                'score_bin',
                'mismatch_count',
                'PAM_last2_pattern'
            ]
        ):
            pos = block.loc[block.label.eq(1), 'accessibility']
            neg = block.loc[block.label.eq(0), 'accessibility']
            if not len(pos) or not len(neg):
                continue
            matched.append(dict(
                dataset=dataset,
                sgRNA=guide,
                cell_type=cell,
                score_bin=binid,
                mismatch_count=mm,
                PAM_last2_pattern=pam,
                TP_n=len(pos),
                FP_n=len(neg),
                TP_mean=pos.mean(),
                FP_mean=neg.mean(),
                FP_minus_TP=neg.mean() - pos.mean()
            ))
        d.to_csv(
            out / (str(dataset) + '_annotated_predictions.csv.gz'),
            index=False,
            compression='gzip'
        )
    m = pd.DataFrame(matched)
    m.to_csv(out / 'score_stratum_FP_vs_TP.csv', index=False)
    if len(m):
        rng = np.random.default_rng(cfg.BOOTSTRAP_SEED)
        for dataset, b in m.groupby('dataset'):
            v = b.groupby('sgRNA').FP_minus_TP.mean().to_numpy()
            lo, hi, pv = bootstrap_delta(v, rng, cfg.BOOTSTRAP_REPS)
            result.append(dict(
                dataset=dataset,
                n_guides=len(v),
                mean_FP_minus_TP=float(v.mean()),
                CI_low=lo,
                CI_high=hi,
                signflip_p=pv,
                note='score-stratified high-score comparison; observational, not proof of false negatives; same-cell annotations required'
            ))
    pd.DataFrame(result).to_csv(out / 'guide_paired_accessibility_summary.csv', index=False)


def specimen_specs():
    specs = []

    def add(name, seed, arm, selection, branch, intervention=None, filename=None):
        path = cfg.OUT_DIR / 'training' / arm / ('seed_%s' % seed) / (filename or 'best_' + selection + '_TTISS.pt')
        if not path.is_file():
            raise FileNotFoundError(path)
        specs.append(dict(
            strategy=name,
            seed=seed,
            path=path,
            branch=branch,
            intervention=intervention
        ))
    for seed in cfg.SEEDS:
        add(cfg.PRIMARY, seed, 'full', 'full', 'own')
        add('Control_fullSelect_full', seed, 'full', 'full', 'full')
        add('Ablation_ownTrain_ownSelect_own', seed, 'own', 'own', 'own')
        for arm in cfg.COMPONENT_ARMS:
            for b in ['own', 'full']:
                add('Component_' + arm + '_fullSelect_' + b, seed, arm, 'full', b)
        add(cfg.SIMPLE_LABELS['simple_additive'], seed, 'simple_additive', 'own', 'own')
        add(cfg.SIMPLE_LABELS['pure_D'], seed, 'pure_D', 'own', 'own')
        if cfg.FINAL_BUDGET_AVAILABLE:
            add('Ablation_ownBudget_ownSelect_own', seed, 'own_budget', 'own', 'own')
            add(
                'Ablation_ownBudget_sameFullEpoch_own',
                seed,
                'own_budget',
                'own',
                'own',
                filename='same_full_selected_epoch.pt'
            )
        for condition in ['no_pam', 'unit_edges', 'constant_distance']:
            for b in ['own', 'full']:
                add('Readonly_' + condition + '_' + b, seed, 'full', 'full', b, condition)
    return specs
