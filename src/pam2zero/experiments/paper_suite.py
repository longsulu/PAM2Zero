from __future__ import annotations
from pam2zero import config as cfg
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
import time
import torch
import traceback


def log_problem(stage, error):
    from pam2zero.utils.io import json_write
    cfg.STAGES[stage] = dict(status='skipped_or_failed', reason=str(error))
    json_write(cfg.OUT_DIR / 'stage_status.json', cfg.STAGES)
    print('[可选阶段未完成]', stage, str(error), flush=True)


def optional(stage, fn):
    from pam2zero.utils.io import json_write
    try:
        result = fn()
        cfg.STAGES[stage] = dict(status='completed')
        json_write(cfg.OUT_DIR / 'stage_status.json', cfg.STAGES)
        return result
    except Exception as error:
        log_problem(stage, error)
        with open(cfg.OUT_DIR / 'optional_errors.txt', 'a', encoding='utf-8') as f:
            f.write('\n' + stage + '\n' + traceback.format_exc())
        return None


def final_source_paths(root, explicit):
    candidates = [explicit] if explicit else [
        root / 'models/PAM2Zero_TypedPairEdges_v1/baseline',
        root / 'models/PAM01_Paper_Unified833_v1/training/pam_zero',
        root / 'models/PAM01_Paper_Unified833/training/pam_zero',
        root / 'models/PAM2Zero_5Seeds_v1',
        root / 'models/PAM01_Paper_Unified833_v1/training/full'
    ]
    found = []
    for source in candidates:
        if source is None:
            continue
        paths = {}
        for seed in cfg.FINAL_SEEDS:
            options = [
                source / ('seed_%s' % seed) / 'best.pt',
                source / ('seed_%s' % seed) / 'best_full_TTISS.pt',
                source / 'training/pam_zero' / ('seed_%s' % seed) / 'best_full_TTISS.pt',
                source / 'training/full' / ('seed_%s' % seed) / 'best_full_TTISS.pt',
                source / 'baseline' / ('seed_%s' % seed) / 'best.pt'
            ]
            valid = [p for p in options if p.is_file()]
            if not valid:
                break
            paths[seed] = valid[0]
        if len(paths) == 5:
            found.append((source, paths))
    if not found:
        raise FileNotFoundError('Need EXISTING five-seed833 ZERO-PAM models. Set FINAL_SOURCE or --source-models DIRECTORY. No main-model retraining fallback. Checked: ' + str(candidates))
    return found[0]


def source_config(path, cp):
    config = cp.get('config', {})
    if isinstance(config, dict) and 'resolved_config' in config:
        config = config['resolved_config']
    if not isinstance(config, dict):
        config = {}
    for parent in list(path.parents)[:6]:
        p = parent / 'protocol.json'
        if p.is_file():
            obj = json.loads(p.read_text(encoding='utf-8-sig'))
            q = obj.get('resolved_config', obj.get('config', {}))
            if isinstance(q, dict):
                config = {**q, **config}
    return config


def final_import(seed, path, val, vy, vg):
    from pam2zero.evaluation.metrics import compute_metrics_paper
    from pam2zero.utils.io import file_hash, json_write, safe_load, save_atomic
    from pam2zero.models.factory import make_model
    from pam2zero.evaluation.inference import predict_both
    cp = safe_load(path)
    state = cp['state_dict']
    m = make_model(seed, 'full')
    m.load_state_dict(state, strict=True)
    m.check_inputs = False
    m.eval()
    v = predict_both(m, val)
    vm = {b: compute_metrics_paper(vy, s, vg)[0] for b, s in v.items()}
    saved = cp.get('validation', {})
    if isinstance(saved, dict):
        recorded = saved.get(
            'full',
            {}).get('Macro_AUPRC') if isinstance(saved.get('full'),
            dict) else saved.get('Macro_AUPRC'
        )
        if recorded is not None and abs(float(recorded) - vm['full']['Macro_AUPRC']) > 2e-05:
            json_write(
                cfg.OUT_DIR / 'source_reproduction_failure.json',
                dict(
                    path=str(path),
                    architecture=cp.get('architecture'),
                    numerics=cfg.SOURCE_NUMERICS,
                    recorded=float(recorded),
                    recomputed=vm['full']['Macro_AUPRC'],
                    delta=vm['full']['Macro_AUPRC'] - float(recorded),
                    source_config=source_config(path, cp)
                )
            )
            raise ValueError('Source TTISS FULL cannot reproduce: recorded=%s recomputed=%s delta=%s; see source_reproduction_failure.json; ' % (
                recorded,
                vm['full']['Macro_AUPRC'],
                vm['full']['Macro_AUPRC'] - float(recorded)) + str(path
            ))
    epoch = int(cp.get('epoch', cp.get('best_epoch', -1)))
    if epoch < 0:
        raise ValueError('Source checkpoint missing selected epoch: ' + str(path))
    run = cfg.OUT_DIR / 'training/full' / ('seed_%s' % seed)
    run.mkdir(parents=True, exist_ok=True)
    imported = dict(
        architecture=cfg.ARCHITECTURE,
        fingerprint=cfg.PROTOCOL_FINGERPRINT,
        arm='full',
        seed=seed,
        epoch=epoch,
        selection='full',
        validation=vm,
        state_dict=state,
        source_checkpoint=str(path),
        source_sha256=file_hash(path)
    )
    save_atomic(imported, run / 'best_full_TTISS.pt')
    if cfg.FINAL_BUDGET_AVAILABLE:
        history = cfg.FINAL_SOURCE_HISTORIES[seed]
        save_atomic(
            dict(fingerprint=cfg.PROTOCOL_FINGERPRINT, finished=True, epoch=len(history), history=history),
                
                
                
            run / 'last_state.pt'
        )
    return dict(
        seed=seed,
        path=str(path),
        epoch=epoch,
        source_sha256=file_hash(path),
        **{'TTISS_' + b: vm[b]['Macro_AUPRC'] for b in vm}
    )


def final_main():
    cfg.initialize_device()
    from pam2zero.utils.provenance import code_hashes
    from pam2zero.evaluation.statistics import additional_paired, statistical_summary
    from pam2zero.data.loaders import compute_group_weights, dataset_identity_audit, load_train_ttiss_balanced
    from pam2zero.visualization.figures import ensure_plotting, final_paper_extras, paper_plots
    from pam2zero.evaluation.suite import error_analysis, evaluation_suite, specimen_specs
    from pam2zero.utils.io import file_hash, json_write, safe_load
    from pam2zero.analysis.efficiency import final_efficiency
    from pam2zero.analysis.moff import final_moff_audit
    from pam2zero.validation import final_self_test
    from pam2zero.analysis.interpretability import interpretation_suite
    from pam2zero.models.factory import make_model
    from pam2zero.probability.workflow import probability_inputs, probability_run
    from pam2zero.experiments.training import train_arm
    p = argparse.ArgumentParser(description='Final PAM2nt ZERO-PAM paper experiments; reuse existing five seeds, retrain ablations only')
    p.add_argument('--root', type=Path, default=cfg.ROOT)
    p.add_argument('--source-models', type=Path, default=cfg.FINAL_SOURCE)
    p.add_argument('--out', type=Path)
    p.add_argument('--self-test', action='store_true')
    p.add_argument('--eval-only', action='store_true')
    p.add_argument('--epochs', type=int)
    p.add_argument('--patience', type=int)
    p.add_argument('--cpu', action='store_true')
    p.add_argument('--save-sample-predictions', action='store_true')
    p.add_argument('--error-annotations', type=Path)
    p.add_argument('--probability-zip', type=Path)
    p.add_argument('--on-target-csv', type=Path)
    args = p.parse_args()
    cfg.SAVE_SAMPLE_PREDICTIONS = args.save_sample_predictions
    cfg.ERROR_ANNOTATIONS = args.error_annotations
    if args.probability_zip:
        cfg.RUN_PROBABILITY = True
        cfg.PROBABILITY_RESULT_ZIP = args.probability_zip
        cfg.PROBABILITY_ON_TARGET_CSV = args.on_target_csv
    threadpool_limits(1)
    torch.set_num_threads(1)
    if args.cpu:
        vars(cfg)['DEVICE'] = torch.device('cpu')
    if args.self_test:
        final_self_test()
        return
    cfg.ROOT = args.root
    cfg.SEEDS = cfg.FINAL_SEEDS.copy()
    cfg.PRIMARY = 'Main_fullSelect_own'
    cfg.OUT_DIR = args.out or cfg.ROOT / 'models/FinalPAM2Zero_PaperSuite_v1'
    cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)
    source, paths = final_source_paths(cfg.ROOT, args.source_models)
    source_hashes = {str(v): file_hash(v) for v in paths.values()}
    configs = []
    source_audit = []
    for seed, path in paths.items():
        cp = safe_load(path)
        st = cp.get('state_dict', {})
        if tuple(st.get('matrix_d', torch.empty(0)).shape) != (
            20,
            36) or tuple(st.get('matrix_pam', torch.empty(0)).shape) != (2,
            36
        ):
            raise ValueError('Not833 PAM2: ' + str(path))
        if not torch.equal(st['initial_pam'], torch.zeros_like(st['initial_pam'])):
            raise ValueError('NOT ZERO-initialized PAM source: ' + str(path))
        if cp.get('seed', seed) != seed:
            raise ValueError('seed mismatch')
        cfg.FINAL_INITIAL_STATES[seed] = st
        configs.append(source_config(path, cp))
    for seed, path in paths.items():
        candidates = [path.parent / 'last_state.pt', path.parent / 'last.pt']
        for last in candidates:
            if not last.is_file():
                continue
            obj = safe_load(last)
            history = obj.get('history', [])
            selected = int(safe_load(path).get('epoch', -1))
            if history and all(('lr_used' in row and len(row['lr_used']) == 2 for row in history)) and (0 < selected <= len(history)):
                cfg.FINAL_SOURCE_HISTORIES[seed] = history
                break
    cfg.FINAL_BUDGET_AVAILABLE = len(cfg.FINAL_SOURCE_HISTORIES) == 5
    vars(cfg)['SOURCE_NUMERICS'] = safe_load(paths[42]).get(
        'numerics',
        'typed_vector' if safe_load(paths[42]).get('architecture') == 'PAM2ZeroTypedPairs-v1' else 'legacy_logits'
    )
    cfg.FINAL_SCALE = float(cfg.FINAL_INITIAL_STATES[42]['input_scale'])
    for seed, st in cfg.FINAL_INITIAL_STATES.items():
        if float(st['input_scale']) != cfg.FINAL_SCALE or not torch.equal(
            st['initial_d'],
            cfg.FINAL_INITIAL_STATES[42]['initial_d']
        ):
            raise ValueError('Source initial D/scale differ across seeds')
    cfg.FINAL_CONFIG = configs[0]
    for k in cfg.CONFIG_KEYS:
        vals = [q[k] for q in configs if k in q]
        if vals and any((v != vals[0] for v in vals)):
            raise ValueError('Seed protocols differ: ' + k)
        if vals:
            vars(cfg)[k] = vals[0]
    needed = [
        'LR_D',
        'LR_OTHER',
        'FOCAL_ALPHA',
        'FOCAL_GAMMA',
        'GROUP_POWER',
        'GROUP_FLOOR',
        'GROUP_CAP',
        'WEIGHT_DECAY'
    ]
    absent = [k for k in needed if k not in cfg.FINAL_CONFIG]
    if absent:
        raise ValueError('Source protocol/config missing ' + str(absent) + '. Keep original protocol.json above checkpoints; cannot invent fair ablation settings.')
    if args.epochs is not None:
        vars(cfg)['EPOCHS'] = args.epochs
    if args.patience is not None:
        vars(cfg)['PATIENCE'] = args.patience
    cfg.TRAIN_DATA_PATH = cfg.ROOT / 'data/train_data/CHANGEseq_encoded.pkl'
    cfg.VAL_DATA_PATH = cfg.ROOT / 'data/val_data/ttiss_encoded.pkl'
    cfg.TEST_DIR = cfg.ROOT / 'data/test_data'
    cfg.ARMS = [
        'own',
        'frozen_D',
        'random_D',
        'no_pam',
        'pam_one',
        'unit_edges',
        'constant_distance',
        'simple_additive',
        'pure_D'
    ]
    cfg.COMPONENT_ARMS = [
        'frozen_D',
        'random_D',
        'no_pam',
        'pam_one',
        'unit_edges',
        'constant_distance'
    ]
    if cfg.FINAL_BUDGET_AVAILABLE:
        cfg.ARMS.append('own_budget')
    files = [Path(str(b).replace('.pkl', '_' + part + '.pkl')) for b in [
        cfg.TRAIN_DATA_PATH,
        cfg.VAL_DATA_PATH] for part in ['pos_list',
        'labels',
        'sg_list'
    ]]
    hashes = {str(q): file_hash(q) for q in files}
    settings = dict(
        version='FinalPAM2ZeroPaper-fixed-v2',
        source_sha256=source_hashes,
        training_sha256=hashes,
        config={k: vars(cfg)[k] for k in cfg.CONFIG_KEYS},
        arms=cfg.ARMS,
        seeds=cfg.SEEDS,
        primary=cfg.PRIMARY,
        script_sha256=file_hash(__file__),
        code_sha256=code_hashes()
    )
    cfg.PROTOCOL_FINGERPRINT = hashlib.sha256(json.dumps(
        settings,
        sort_keys=True,
        default=str).encode(
    )).hexdigest()
    (cfg.OUT_DIR / 'INPUT_GUIDE.txt').write_text(
        'Final PAM2 ZERO: source five-seed checkpoints mandatory. Use --source-models DIRECTORY if auto-discovery fails. Prior .npy is not needed: initial D/scale copied from checkpoint buffers. MAIN never retrained. MOFF_init/869 are not used.\n' + cfg.INPUT_TEXT,
            
            
            
        encoding='utf8'
    )
    protocol = cfg.OUT_DIR / 'protocol.json'
    if protocol.exists() and json.loads(protocol.read_text())['fingerprint'] != cfg.PROTOCOL_FINGERPRINT:
        old_protocol = json.loads(protocol.read_text(encoding='utf8'))
        trained = [q for q in (cfg.OUT_DIR / 'training').glob('*/*/*.pt') if q.parent.parent.name != 'full']
        if trained:
            raise ValueError('Existing trained ablations use a different script/protocol; preserved all weights. Use --out models/FinalPAM2Zero_PaperSuite_Fixed_v2 to keep this run separate.')
        backup = cfg.OUT_DIR / ('protocol_before_failed_import_' + time.strftime('%Y%m%d_%H%M%S') + '.json')
        backup.write_text(json.dumps(old_protocol, ensure_ascii=False, indent=2), encoding='utf8')
        print(
            'Recovered training-before-start protocol conflict. Previous protocol backed up:',
            backup,
            flush=True
        )
    json_write(protocol, dict(fingerprint=cfg.PROTOCOL_FINGERPRINT, **settings))
    (cfg.OUT_DIR / 'METHOD_评估协议.txt').write_text(cfg.FINAL_METHOD, encoding='utf8')
    final_self_test()
    ensure_plotting()
    data = load_train_ttiss_balanced()
    tr, y, gid, groups, val, vy, vg = data
    weights = compute_group_weights(
        gid,
        groups,
        cfg.GROUP_POWER,
        cfg.GROUP_FLOOR,
        cfg.GROUP_CAP,
        cfg.OUT_DIR
    )
    for seed, path in paths.items():
        source_audit.append(final_import(seed, path, val, vy, vg))
    pd.DataFrame(source_audit).to_csv(cfg.OUT_DIR / 'main_checkpoint_reproduction.csv', index=False)
    cfg.STAGES['主模型复用'] = dict(
        status='completed',
        seeds=cfg.SEEDS,
        source=str(source),
        retrained=False
    )
    cfg.STAGES['匹配训练预算消融'] = dict(
        status='completed' if cfg.FINAL_BUDGET_AVAILABLE else 'unavailable',
        reason='Exact source epoch/LR schedule available for all five seeds' if cfg.FINAL_BUDGET_AVAILABLE else 'Source last_state.pt/last.pt lacks complete two-group lr_used histories; independent no-competition retraining is included.'
    )
    if not args.eval_only:
        for seed in cfg.SEEDS:
            for arm in cfg.ARMS:
                train_arm(arm, seed, data, weights, cfg.PROTOCOL_FINGERPRINT)
    cfg.STAGES['组件消融'] = dict(status='completed', arms=cfg.ARMS, seeds=cfg.SEEDS)
    specs = specimen_specs()
    df, per = evaluation_suite(specs, val, vy, vg)
    df['Global_AP'] = df['Global_AUPRC']
    df.to_csv(cfg.OUT_DIR / 'all_run_metrics.csv', index=False)
    statistical_summary(df, per)
    additional_paired(per)
    counts = []
    for arm in ['full'] + cfg.ARMS:
        m = make_model(42, arm)
        counts.append(dict(
            arm=arm,
            allocated=sum((q.numel() for q in m.parameters())),
            trainable=sum((q.numel() for q in m.parameters() if q.requires_grad))
        ))
    pd.DataFrame(counts).to_csv(cfg.OUT_DIR / 'parameter_counts.csv', index=False)
    support = np.stack([np.bincount(tr[:, i], minlength=37)[1:37] for i in range(23)])
    interpretation_suite(tr, support)
    cfg.STAGES['解释性分析'] = dict(status='completed')
    final_efficiency(tr)
    optional('输入身份审计', lambda: dataset_identity_audit(tr, groups, val, vg))
    optional('表观错误分析', error_analysis)
    optional('MOFF实验互作验证', final_moff_audit)
    if cfg.RUN_PROBABILITY:
        optional('重复检出概率模型', lambda: probability_run(probability_inputs(), groups, vg))
    paper_plots()
    final_paper_extras()
    cfg.STAGES['论文图表'] = dict(status='completed')
    for q, h in {**source_hashes, **hashes}.items():
        if file_hash(q) != h:
            raise RuntimeError('Source file changed: ' + q)
    json_write(cfg.OUT_DIR / 'stage_status.json', cfg.STAGES)
    json_write(
        cfg.OUT_DIR / 'completion.json',
        dict(
            core_completed=True,
            seeds=cfg.SEEDS,
            main_retrained=False,
            source_hashes_unchanged=True,
            incomplete_modules={k: v for k, v in cfg.STAGES.items() if v.get('status') != 'completed'},
                
                
            output=str(cfg.OUT_DIR)
        )
    )
    print(
        'DONE: main five-seed833 reused; component ablations and interpretation saved. Check stage_status.json for missing optional data. OUTPUT:',
            
            
            
        cfg.OUT_DIR
    )
