from __future__ import annotations
import argparse, json, hashlib, zipfile
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.special import expit
from threadpoolctl import threadpool_limits
from pam2zero import config as cfg
from pam2zero.probability.models import pm_binary_label, pm_binary_matrix, pm_bounds_and_start, pm_ctmc_activity, pm_evaluate_models, pm_fit_model, pm_history_nll, pm_make_pools, pm_pool_sensitivity, pm_probabilities, pm_source_consistency
SNAPSHOT = json.loads((Path(__file__).resolve().parents[1] / 'resources' / 'seed42_own_coefficients.json').read_text())
REFERENCE_AUDIT = {
    'CCR5': {'guide': 'CCR5', 'guide20': 'GTGTTCATCTTTGGTTTTGT', 'train_min_Hamming': 11, 'TTISS_min_Hamming': 9, 'unseen_teacher_within2': True},
        
        
        
    'FANCF': {'guide': 'FANCF', 'guide20': 'GGAATCCCTTCTGCAGCACC', 'train_min_Hamming': 6, 'TTISS_min_Hamming': 8, 'unseen_teacher_within2': True},
        
        
        
    'HEK1': {'guide': 'HEK1', 'guide20': 'GGGAAAGACCCAGCATCCGT', 'train_min_Hamming': 9, 'TTISS_min_Hamming': 10, 'unseen_teacher_within2': True},
        
        
        
    'HEK2': {'guide': 'HEK2', 'guide20': 'GAACACAAAGCATAGACTGC', 'train_min_Hamming': 9, 'TTISS_min_Hamming': 10, 'unseen_teacher_within2': True},
        
        
        
    'HEK3': {'guide': 'HEK3', 'guide20': 'GGCCCAGACTGAGCACGTGA', 'train_min_Hamming': 8, 'TTISS_min_Hamming': 9, 'unseen_teacher_within2': True},
        
        
        
    'IL2RG': {'guide': 'IL2RG', 'guide20': 'TGGTAATGATGGCTTCAACA', 'train_min_Hamming': 10, 'TTISS_min_Hamming': 11, 'unseen_teacher_within2': True},
        
        
        
    'RNF2': {'guide': 'RNF2', 'guide20': 'GTCATCTTAGTCATTACCTG', 'train_min_Hamming': 9, 'TTISS_min_Hamming': 11, 'unseen_teacher_within2': True},
        
        
        
    'VEGFA1': {
        'guide': 'VEGFA1',
        'guide20': 'GGGTGGGGGGAGTTTGCTCC',
        'train_min_Hamming': 9,
        'TTISS_min_Hamming': 8,
        'unseen_teacher_within2': True
    }
}
GUIDE_COEF = np.full((20, 36), np.nan)
PAM_COEF = np.full((2, 36), np.nan)
for i, c, v in SNAPSHOT['guide']:
    GUIDE_COEF[int(i) - 1, int(c) - 1] = v
for i, c, v in SNAPSHOT['pam']:
    PAM_COEF[int(i) - 2, int(c) - 1] = v

def rule_B(x):
    x = str(x)
    return x[:-3] if len(x) > 20 else x

def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def encode_pair(sg, off):
    pairs = []
    for s, o in zip(sg, off):
        if s == 'N':
            s = o
        elif o == 'N':
            o = s
        if s + o not in cfg.ENCODE_DICT_36:
            raise ValueError('invalid base pair ' + s + o)
        pairs.append(cfg.ENCODE_DICT_36[s + o])
    return tuple(pairs)

def ensure_plotting():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as e:
        if cfg.REQUIRE_FIGURES:
            raise RuntimeError('完整图表需要matplotlib，请先在CRISPR_GNN安装：python -m pip install matplotlib') from e
        return None
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 10,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'pdf.fonttype': 42,
        'svg.fonttype': 'none'
    })
    return plt

def save_figure(fig, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ['.png', '.pdf', '.svg']:
        fig.savefig(path.with_suffix(suffix), dpi=300, bbox_inches='tight')
    import matplotlib.pyplot as plt
    plt.close(fig)

def probability_inputs():
    if not cfg.RUN_PROBABILITY:
        return None
    if cfg.PROBABILITY_RESULT_ZIP:
        p = Path(cfg.PROBABILITY_RESULT_ZIP)
        candidates = [p if p.is_absolute() else cfg.ROOT / p]
    else:
        candidates = sorted(
            (cfg.ROOT / 'models/PAM01_ReplicateValidation_Local_CoordV2').glob('*/results.zip'),
            key=lambda p: p.stat().st_mtime,
            reverse=True
        )
    required = [
        'LOCK_BEFORE_EXTERNAL_SCORING.json',
        'audit.json',
        'prepared_sequences.csv',
        'nominated_sites_predictions.csv',
        'hybrid_capture_predictions.csv',
        'rhAmpSeq_secondary_predictions.csv'
    ]
    chosen = None
    for path in candidates:
        with zipfile.ZipFile(path) as z:
            if set(required).issubset(z.namelist()):
                lock = json.loads(z.read(required[0]))
                audit = json.loads(z.read('audit.json'))
                if lock.get('coordinate_normalization_version') == 2:
                    chosen = path
                    break
    if chosen is None:
        raise FileNotFoundError('概率模块需要之前已经跑完的坐标v2结果：models/PAM01_ReplicateValidation_Local_CoordV2/*/results.zip；可在顶部PROBABILITY_RESULT_ZIP指定。不会下载或重做坐标。')
    on_candidates = []
    if cfg.PROBABILITY_ON_TARGET_CSV:
        pp = Path(cfg.PROBABILITY_ON_TARGET_CSV)
        on_candidates.append(pp if pp.is_absolute() else cfg.ROOT / pp)
    on_candidates.extend((Path(p) for p in audit.get(
        'source_hashes',
        {}) if Path(p
    ).name == 'CSV3_with_sequences.csv'))
    on_candidates.extend(sorted(
        (cfg.ROOT / 'data/Chaudhari2020_sequences').glob('*/CSV3_with_sequences.csv'),
        reverse=True
    ))
    on = next((p for p in on_candidates if p.is_file()), None)
    if on is None:
        raise FileNotFoundError('需要已提取的CSV3_with_sequences.csv，包括on-target；请设置PROBABILITY_ON_TARGET_CSV。不会联网提序列。')
    with zipfile.ZipFile(chosen) as z:
        schemas = {name: list(pd.read_csv(z.open(name), nrows=0).columns) for name in required[2:]}
    if not all((c in schemas['nominated_sites_predictions.csv'] for c in [
        'key',
        'guide',
        'guide23',
        'DNA',
        'mismatches'] + ['GUIDE-seq_detected_%d' % i for i in [1, 2, 3]
    ])):
        raise ValueError('重复检出表缺列，不能将普通二分类表当三重复历史')
    expectations = {
        'prepared_sequences.csv': ['key', 'guide', 'guide23', 'DNA'],
        'hybrid_capture_predictions.csv': ['key', 'guide', 'guide23', 'DNA', 'mismatches', 'hybrid_label'],
            
            
            
        'rhAmpSeq_secondary_predictions.csv': [
            'key',
            'guide',
            'guide23',
            'DNA',
            'mismatches',
            'rhAmpSeq_avg_delta_gt_0_2pct'
        ]
    }
    for name, cols in expectations.items():
        if not set(cols).issubset(schemas[name]):
            raise ValueError('概率文件缺列：' + name + ' ' + str(set(cols) - set(schemas[name])))
    oncols = set(pd.read_csv(on, nrows=0).columns)
    if not {'guide23', 'DNA', 'chromosome', 'start', 'end'}.issubset(oncols):
        raise ValueError('on-target文件缺少已提取序列/坐标列')
    return dict(
        source_zip=chosen,
        on_target_csv=on,
        source_sha256=file_hash(chosen),
        on_target_sha256=file_hash(on),
        old_lock=lock,
        schemas=schemas
    )

def probability_self_test():
    pool = dict(
        z=np.array([-2.0, -0.5, 1.0, 2.0], dtype=np.float64),
        h=np.array([2, 1, 0, 0]),
        keys=np.array(['a', 'b', 'c']),
        n_off=3,
        n_on=1
    )
    for kind in cfg.MODEL_NAMES:
        _, theta = pm_bounds_and_start(kind)
        q, latent, rank = pm_probabilities(kind, pool, theta)
        assert q.shape == (
            4,
            3) and np.isfinite(q).all() and ((q >= 0) & (q <= 1)).all() and np.isfinite(rank).all(
        )
        yy = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=float)
        assert np.isfinite(pm_history_nll(kind, pool, theta, np.arange(3), yy))
    _, diag = pm_ctmc_activity(pool, 0.5, 1.0, 0.0, True)
    assert abs(diag['resource_mass'] - 1) < 1e-10
    order = np.array([2, 0, 1, 3])
    permuted = dict(pool, z=pool['z'][order], h=pool['h'][order])
    a = pm_ctmc_activity(pool, 0.5, 1.0, 0.0)
    b = pm_ctmc_activity(permuted, 0.5, 1.0, 0.0)
    assert np.allclose(a[order], b, rtol=1e-09, atol=1e-12)
    extra = dict(pool, z=np.r_[pool['z'], 5.0], h=np.r_[pool['h'], 0])
    _, theta = pm_bounds_and_start('Independent_Kinetic')
    assert np.allclose(
        pm_probabilities('Independent_Kinetic', pool, theta)[0],
        pm_probabilities('Independent_Kinetic', extra, theta)[0][:4]
    )
    assert not np.allclose(
        pm_ctmc_activity(pool, 0.5, 1.0, 0.0),
        pm_ctmc_activity(extra, 0.5, 1.0, 0.0)[:4],
        rtol=1e-05
    )
    for kind in ['Calibrated_Bernoulli', 'FiniteBudget', 'Occupancy_Detection']:
        _, theta = pm_bounds_and_start(kind)
        rank = pm_probabilities(kind, pool, theta)[2]
        assert np.array_equal(np.argsort(rank), np.argsort(pool['z']))
    print(
        'PASS: float64 probabilities/history likelihoods, CTMC positivity/resource conservation, site permutation invariance, independent-pool invariance, monotone controls.',
            
            
            
        flush=True
    )

def probability_run(inputs, train_groups, val_groups):
    if not cfg.RUN_PROBABILITY:
        return
    out = cfg.OUT_DIR / 'probability_models'
    out.mkdir(exist_ok=True)
    source = inputs['source_zip']
    on_path = inputs['on_target_csv']
    with zipfile.ZipFile(source) as z:
        prepared = pd.read_csv(z.open('prepared_sequences.csv'), low_memory=False)
        obs = pd.read_csv(z.open('nominated_sites_predictions.csv'), low_memory=False)
        hybrid = pd.read_csv(z.open('hybrid_capture_predictions.csv'), low_memory=False)
        deeper = pd.read_csv(z.open('rhAmpSeq_secondary_predictions.csv'), low_memory=False)
    obs = pm_source_consistency(obs, 'GUIDE', out)
    hybrid = pm_source_consistency(hybrid, 'hybrid', out)
    deeper = pm_source_consistency(deeper, 'rhAmpSeq', out)
    hybrid['hybrid_label'] = pm_binary_label(hybrid.hybrid_label)
    deeper['rhAmpSeq_avg_delta_gt_0_2pct'] = pm_binary_label(deeper['rhAmpSeq_avg_delta_gt_0_2pct'])
    for r in [1, 2, 3]:
        obs['GUIDE-seq_detected_%d' % r] = pm_binary_label(obs['GUIDE-seq_detected_%d' % r])
    if prepared.key.duplicated().any():
        raise ValueError('概率候选池存在重复位点key；不同坐标同序列应保留，不得按输入去重竞争池')
    cp = cfg.OUT_DIR / 'current833_own_snapshot.json'
    model = None
    metadata = {}
    checkpoint_sha = file_hash(cp)

    def recalculate(d):
        if any((len(str(g)) != 23 or len(str(o)) != 23 for g, o in zip(d.guide23, d.DNA))):
            raise ValueError('概率输入必须有完整guide23和DNA23')
        codes = np.array(
            [encode_pair(str(g), str(o)) for g, o in zip(d.guide23, d.DNA)],
            dtype=np.uint8
        )
        d = d.copy()
        d['score'] = score_own_float64(model, codes)
        return d
    prepared = recalculate(prepared)
    on = pd.read_csv(
        on_path,
        low_memory=False).rename(columns={'mm': 'mismatches', 'gap': 'gaps', 'gRNA': 'guide'}
    )
    on = on[(on.mismatches == 0) & (on.gaps == 0)].drop_duplicates([
        'guide',
        'chromosome',
        'start',
        'end'
    ]).copy()
    if any((g[:20] != o[:20] for g, o in zip(on.guide23, on.DNA))):
        raise ValueError('on-target的guide/DNA前20位不相同，无法构建正确资源池')
    on = recalculate(on)
    on.to_csv(out / 'on_target_scores.csv', index=False)
    pools = pm_make_pools(prepared, on)
    guides = sorted(obs.guide.unique())
    if set(guides) != set(pools):
        raise ValueError('检出历史guide和完整候选池不一致')
    histories = {}
    overlap = []
    trainrefs = sorted(set((rule_B(g) for g in train_groups)))
    valrefs = sorted(set((rule_B(g) for g in val_groups)))
    for g in guides:
        t = obs[obs.guide == g]
        idx = pd.Index(pools[g]['keys']).get_indexer(t.key)
        if (idx < 0).any():
            raise ValueError('检出历史含候选池之外位点')
        history = pm_binary_matrix(t)
        if not np.isin(history, [0.0, 1.0]).all():
            raise ValueError('无效重复历史')
        histories[g] = (idx, history)
        seqs = prepared.loc[prepared.guide.eq(g), 'guide23'].str[:20].unique()
        if len(seqs) != 1:
            raise ValueError('同一guide标识对应多条序列')
        seq = seqs[0]
        distance = lambda refs: min([sum((
            a != b for a,
            b in zip(seq, v)
        )) for v in refs if len(v) == 20] or [20])
        audit = REFERENCE_AUDIT[g]
        assert seq == audit['guide20']
        overlap.append(audit)
    overlap = pd.DataFrame(overlap)
    overlap.to_csv(out / 'teacher_guide_overlap.csv', index=False)
    protocol = dict(
        fit='LOGO: all three GUIDE histories of held guide excluded',
        guides=guides,
        models=cfg.MODEL_NAMES,
        extra_parameters=cfg.KINDS_PARAMETERS,
        source=str(source),
        source_sha256=inputs['source_sha256'],
        on_target_sha256=inputs['on_target_sha256'],
        teacher_checkpoint=str(cp),
        teacher_sha256=checkpoint_sha,
        teacher='primary paper checkpoint seed42; frozen exported OWN coefficients; float64 accumulation',
            
            
            
        original_checkpoint_sha256=SNAPSHOT['original_checkpoint_sha256'],
        score_precision='exported float32 coefficients accumulated in float64',
        historical_teacher=inputs['old_lock'].get('teacher'),
        recompute_all_pool_scores=True,
        historical_scores_not_mixed=True,
        probability_constants=dict(FIXED_ATTEMPTS=cfg.FIXED_ATTEMPTS, ENCOUNTER_SCALE=cfg.ENCOUNTER_SCALE, BACKGROUND_WEIGHT=cfg.BACKGROUND_WEIGHT),
            
            
            
        primary_endpoint='hybrid_capture',
        rhAmpSeq='author-selected secondary subset',
        all_endpoint_labels_excluded_from_fitting=True,
        network_unseen_guide_claim=False,
        prior_source_independence_verified=False,
        probability_fit_guide_count=len(guides)
    )
    protocol_path = out / 'protocol_before_fitting.json'
    if protocol_path.is_file() and json.loads(protocol_path.read_text(encoding='utf-8')) != protocol:
        raise RuntimeError('概率模块的教师/数据/常量改变，禁止复用旧拟合缓存。请用新的输出目录。')
    json_write(protocol_path, protocol)
    raw = prepared[['key', 'guide', 'score']].copy()
    raw['rank_score'] = raw.score
    raw['latent_probability'] = expit(raw.score)
    for r in [1, 2, 3]:
        raw['q%d' % r] = raw.latent_probability
    scores = {'Raw_PAM01': raw.drop(columns='score')}
    constant = []
    for held in guides:
        train = {g: v for g, v in histories.items() if g != held}
        q = np.concatenate([v[1] for v in train.values()]).mean(0)
        p = pools[held]
        d = pd.DataFrame(dict(
            key=p['keys'],
            guide=held,
            rank_score=0.0,
            latent_probability=float(q.mean())
        ))
        for r in [1, 2, 3]:
            d['q%d' % r] = q[r - 1]
        constant.append(d)
    scores['Constant_train_prevalence'] = pd.concat(constant, ignore_index=True)
    with threadpool_limits(limits=1):
        for kind in cfg.MODEL_NAMES:
            pieces = []
            for held in guides:
                training = {g: v for g, v in histories.items() if g != held}
                print('[概率LOGO]', kind, 'held', held, flush=True)
                theta, spec = pm_fit_model(kind, training, pools, out, held)
                q, latent, rank = pm_probabilities(kind, pools[held], theta)
                p = pools[held]
                n = p['n_off']
                if kind in ['Calibrated_Bernoulli', 'FiniteBudget', 'Occupancy_Detection']:
                    rank = p['z'].copy()
                d = pd.DataFrame(dict(
                    key=p['keys'],
                    guide=held,
                    rank_score=rank[:n],
                    latent_probability=latent[:n]
                ))
                for r in [1, 2, 3]:
                    d['q%d' % r] = q[:n, r - 1]
                pieces.append(d)
                if kind == 'SharedCas9_CTMC':
                    _, diag = pm_ctmc_activity(p, theta[0], theta[1], theta[2], True)
                    json_write(out / ('resource_' + str(held) + '.json'), diag)
            scores[kind] = pd.concat(pieces, ignore_index=True)
        for name, d in scores.items():
            d.to_csv(out / ('predictions_' + name + '.csv'), index=False)
        pm_evaluate_models(scores, obs, hybrid, deeper, out)
        pm_pool_sensitivity(pools, out)
    fits = [json.loads(p.read_text(encoding='utf-8')) for p in sorted(out.glob('fit_*.json'))]
    pd.DataFrame([{
        k: v for k,
        v in d.items() if k in (
            'model',
            'held_guide',
            'objective',
            'success',
            'message',
            'iterations',
            'boundary_parameters',
            'parameter_count'
        )
    } for d in fits]).to_csv(
        out / 'fit_diagnostics.csv',
        index=False
    )
    unseen = set(overlap.loc[overlap.unseen_teacher_within2, 'guide'])
    eligible = hybrid[hybrid.guide.isin(unseen)]
    if len(eligible) and any((
        len(v.hybrid_label.unique()) == 2 for _,
        v in eligible.groupby('guide')
    )):
        sub = out / 'teacher_unseen_subset'
        sub.mkdir(exist_ok=True)
        pm_evaluate_models(
            scores,
            obs[obs.guide.isin(unseen)],
            eligible,
            deeper[deeper.guide.isin(unseen)],
            sub
        )
    if file_hash(source) != inputs['source_sha256'] or file_hash(on_path) != inputs['on_target_sha256'] or file_hash(cp) != checkpoint_sha:
        raise RuntimeError('概率阶段源文件/教师发生变化')
    probability_figures(out)
    json_write(
        out / 'completion.json',
        dict(
            completed=True,
            n_guides=len(guides),
            no_neural_training=True,
            fit_labels='other guides GUIDE histories only',
            not_proof_of_molecular_competition=True
        )
    )

def probability_figures(out):
    plt = ensure_plotting()
    if plt is None:
        return
    m = pd.read_csv(out / 'metrics_summary.csv')
    cal = pd.read_csv(out / 'heldout_GUIDE_calibration.csv')
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.7), constrained_layout=True)
    for ax, metric in zip(axs, ['Macro_AUPRC', 'Macro_AP']):
        t = m[m.endpoint.eq('hybrid_capture')].set_index('model')
        ax.barh(t.index, t[metric])
        ax.set_xlabel(metric)
        ax.set_title('LOGO independent hybrid-capture endpoint')
        ax.invert_yaxis()
    save_figure(fig, out / 'fig_probability_independent_endpoint')
    q = cal.groupby('model')[['Brier', 'marginal_logloss']].mean()
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.7), constrained_layout=True)
    for ax, col in zip(axs, ['Brier', 'marginal_logloss']):
        ax.barh(q.index, q[col])
        ax.invert_yaxis()
        ax.set_xlabel(col + ' (lower better)')
        ax.set_title('Held-out-guide detection calibration')
    save_figure(fig, out / 'fig_probability_calibration')
    s = pd.read_csv(out / 'candidate_pool_sensitivity.csv')
    fig, ax = plt.subplots(figsize=(11, 5.5), constrained_layout=True)
    for kind, b in s.groupby('model'):
        q = b.groupby('scenario').mean_abs_delta_probability.mean()
        ax.plot(q.index, q.values, marker='o', label=kind)
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylabel('Mean absolute probability shift')
    ax.set_title('Candidate-pool stress test: fixed parameters, no label use')
    ax.legend(fontsize=8)
    save_figure(fig, out / 'fig_probability_pool_sensitivity')

def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    tmp.replace(path)

def score_own_float64(model, codes):
    codes = np.asarray(codes, dtype=int)
    if codes.ndim != 2 or codes.shape[1] != 23 or codes.min() < 1 or (codes.max() > 36):
        raise ValueError('要求规范化23位置ID')
    a = GUIDE_COEF[np.arange(20)[None, :], codes[:, :20] - 1]
    b = PAM_COEF[np.arange(2)[None, :], codes[:, 21:23] - 1]
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('包含未导出的gap/特殊编码；禁止把缺失系数填0')
    return a.sum(1, dtype=np.float64) + b.sum(1, dtype=np.float64)

def coefficient_self_test():
    wt = 'GAACAAGGTGTTCCCACCCGTGG'
    mm = 'AAACAAGGTGTTCCCACCCGTGG'
    x = np.array([encode_pair(wt, wt), encode_pair(wt, mm)])
    q = score_own_float64(None, x)
    np.testing.assert_allclose(q[1] - q[0], -0.5217595100402832, atol=2e-05, rtol=0)
    y = x.copy()
    y[:, 20] = 1
    np.testing.assert_array_equal(score_own_float64(None, y), q)
    np.testing.assert_array_equal(score_own_float64(None, x[::-1]), q[::-1])
    assert SNAPSHOT['original_checkpoint_sha256'] == 'c8ee012523c64b52cb7197f9ba4c1e4fbf911dfccf8271627885f5030d9cc449'
    print('PASS: current-primary checkpoint identity, coefficient substitution parity, ignored PAM1, batch isolation.')

def main():
    cfg.RUN_PROBABILITY = True
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=str(cfg.ROOT))
    ap.add_argument('--out')
    ap.add_argument('--probability-zip')
    ap.add_argument('--on-target-csv')
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    coefficient_self_test()
    probability_self_test()
    if args.self_test:
        return
    cfg.ROOT = Path(args.root)
    cfg.PROBABILITY_RESULT_ZIP = args.probability_zip
    cfg.PROBABILITY_ON_TARGET_CSV = args.on_target_csv
    cfg.OUT_DIR = Path(args.out) if args.out else cfg.ROOT / 'models/CurrentPrimary833_ProbabilityAligned' / datetime.now().strftime('%Y%m%d_%H%M%S')
    cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)
    cp = cfg.OUT_DIR / 'current833_own_snapshot.json'
    encoded = json.dumps(SNAPSHOT, ensure_ascii=False, sort_keys=True)
    if cp.exists() and cp.read_text(encoding='utf-8') != encoded:
        raise ValueError('输出目录系数不同；使用新目录')
    cp.write_text(encoded, encoding='utf-8')
    inputs = probability_inputs()
    probability_run(inputs, [], [])
    json_write(
        cfg.OUT_DIR / 'alignment_completed.json',
        dict(
            main_checkpoint_sha256=SNAPSHOT['original_checkpoint_sha256'],
            allocated_parameters=833,
            seed=42,
            neural_training=False,
            probability_layers_refitted=True,
            coordinate_download=False,
            other833_results_reused=False
        )
    )
    target = cfg.OUT_DIR / 'CurrentPrimary833_probability_results.zip'
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in cfg.OUT_DIR.rglob('*'):
            if f.is_file() and f != target and (not f.name.startswith('predictions_')) and (f.suffix in (
                '.csv',
                '.json',
                '.png',
                '.pdf',
                '.svg'
            )):
                z.write(f, f.relative_to(cfg.OUT_DIR))
    print('DONE:', target)
