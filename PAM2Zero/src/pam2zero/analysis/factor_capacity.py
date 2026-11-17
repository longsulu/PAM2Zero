import os
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
os.environ['OMP_NUM_THREADS'] = '1'
import argparse, json, io, itertools, zipfile, hashlib
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sklearn import metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT = Path.cwd()
SEEDS = [42, 123, 3407, 2025, 2026]
PAIRS = [
    'AA',
    'AG',
    'AC',
    'AT',
    'GA',
    'GG',
    'GC',
    'GT',
    'CA',
    'CG',
    'CC',
    'CT',
    'TA',
    'TG',
    'TC',
    'TT'
]
CODES = [1, 2, 3, 4, 8, 7, 9, 10, 14, 15, 13, 16, 20, 21, 22, 19]
MATCH = np.array([x[0] == x[1] for x in PAIRS])
MM = np.flatnonzero(~MATCH)
ENC = {
    'AA': 1,
    'AG': 2,
    'AC': 3,
    'AT': 4,
    'A_': 5,
    'A-': 6,
    'GG': 7,
    'GA': 8,
    'GC': 9,
    'GT': 10,
    'G_': 11,
    'G-': 12,
    'CC': 13,
    'CA': 14,
    'CG': 15,
    'CT': 16,
    'C_': 17,
    'C-': 18,
    'TT': 19,
    'TA': 20,
    'TG': 21,
    'TC': 22,
    'T_': 23,
    'T-': 24,
    '__': 25,
    '_A': 26,
    '_G': 27,
    '_C': 28,
    '_T': 29,
    '_-': 30,
    '--': 31,
    '-A': 32,
    '-G': 33,
    '-C': 34,
    '-T': 35,
    '-_': 36
}
TESTS = [
    'CRISPOR_test',
    'GUIDEseq_test',
    'HEK293T_test',
    'II6_test',
    'K562_test',
    'guideseqCT_test',
    'pdh_test'
]
VALID = [26, 44, 16, 12, 11, 19, 7]

def unpack():
    raw = json.loads((Path(__file__).resolve().parents[1] / 'resources/factor_capacity_exports.json').read_text())
    return {key: pd.read_csv(io.StringIO(value)) for key, value in raw.items()}

def grid(d, key):
    return d.pivot(
        index='position',
        columns='pair',
        values=key).reindex(index=range(1, 21),
        columns=PAIRS).to_numpy(float
    )

def additive(D, support, weighted=False):
    i, j = np.where(support > 0)
    X = np.zeros((len(i), 36))
    X[np.arange(len(i)), i] = 1
    X[np.arange(len(i)), 20 + j] = 1
    w = np.sqrt(support[i, j] / support[i, j].mean()) if weighted else np.ones(len(i))
    beta = np.linalg.lstsq(X * w[:, None], D[i, j] * w, rcond=None)[0]
    pred = beta[:20, None] + beta[None, 20:]
    return (pred, int(np.linalg.matrix_rank(X)))

def recover_transform(D, C):
    sign = np.where(MATCH, 1.0, -1.0)[None, :]
    target = np.abs(C)

    def transformed(x, V):
        s, gm, gx = np.exp(x)
        g = np.where(MATCH, gm, gx)[None, :]
        return sign * g * (np.logaddexp(0.0, V / s) + 1e-06)

    def fun(x):
        return (transformed(x, D) - C).ravel()
    fit = least_squares(
        fun,
        np.log([0.5, 1.0, 1.0]),
        max_nfev=3000,
        ftol=1e-13,
        xtol=1e-13,
        gtol=1e-13
    )
    err = np.max(abs(fun(fit.x)))
    if err > 2e-05:
        raise ValueError('Cannot reconstruct exported OWN transform: ' + str(err))
    return (lambda V: transformed(fit.x, V), float(err), np.exp(fit.x))

def score(codes, guide, pam):
    out = np.empty(len(codes), float)
    for start in range(0, len(codes), 65536):
        x = codes[start:start + 65536].astype(int) - 1
        a = guide[np.arange(20)[None, :], x[:, :20]]
        b = pam[np.arange(2)[None, :], x[:, 21:23]]
        if not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Unexported gap/special code; no zero filling')
        out[start:start + len(x)] = a.sum(1) + b.sum(1)
    return out

def evaluate(y, z, groups, threshold):
    rows = []
    for g in sorted(set(groups)):
        ix = np.flatnonzero(groups == g)
        a = y[ix]
        s = z[ix]
        if len(np.unique(a)) != 2:
            continue
        p, r, _ = metrics.precision_recall_curve(a, s)
        p[r == 0] = 1.0
        pred = s >= threshold
        rows.append(dict(
            guide=g,
            N=len(a),
            Positive=int(a.sum()),
            AUPRC=metrics.auc(r, p),
            AP=metrics.average_precision_score(a, s),
            AUROC=metrics.roc_auc_score(a, s),
            MCC_locked=metrics.matthews_corrcoef(a, pred),
            F1_locked=metrics.f1_score(a, pred, zero_division=0),
            ACC_locked=metrics.accuracy_score(a, pred)
        ))
    per = pd.DataFrame(rows)
    summary = {k: float(per[k].mean()) for k in [
        'AUPRC',
        'AP',
        'AUROC',
        'MCC_locked',
        'F1_locked',
        'ACC_locked'
    ]}
    summary.update(N=len(y), Positive=int(y.sum()), Valid_guides=len(per))
    return (summary, per)

def encode(sg, dna):
    sg = str(sg).upper().strip()
    dna = str(dna).upper().strip()
    if len(sg) != len(dna) or len(sg) > 24:
        return None
    ids = []
    for a, b in zip(sg, dna):
        if a == 'N':
            a = b
        elif b == 'N':
            b = a
        if a + b not in ENC:
            return None
        ids.append(ENC[a + b])
    ids = [31] * (24 - len(ids)) + ids
    if ids[0] not in (25, 31) or any((c not in CODES for c in ids[1:])):
        return None
    return ids[1:]

def load_test(path):
    df = pd.read_csv(path)
    required = {'sg', 'DNA', 'label'}
    if not required.issubset(df):
        raise ValueError('Missing sg/DNA/label: ' + str(path))
    xs = []
    ys = []
    gs = []
    counts = dict(input_rows=len(df), on_target=0, unsupported=0)
    for sg, dna, label in df[['sg', 'DNA', 'label']].itertuples(index=False, name=None):
        sg = str(sg).upper().strip()
        dna = str(dna).upper().strip()
        if sg[:-3] == dna[:-3]:
            counts['on_target'] += 1
            continue
        x = encode(sg, dna)
        if x is None:
            counts['unsupported'] += 1
            continue
        if float(label) not in (0, 1):
            raise ValueError('Nonbinary label')
        xs.append(x)
        ys.append(int(label))
        gs.append(sg[:-3])
    counts['kept'] = len(xs)
    return (np.asarray(xs, np.uint8), np.asarray(ys, int), np.asarray(gs, object), counts)

def figures(out, mats, stats, svd, examples, perf, reported):
    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 9,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'pdf.fonttype': 42
    })

    def save(fig, name):
        fig.tight_layout()
        for ext in ['png', 'pdf']:
            fig.savefig(out / (name + '.' + ext), dpi=230, bbox_inches='tight')
        plt.close(fig)
    mean = np.mean([x['D'] for x in mats.values()], axis=0)
    proj = np.mean([x['A'] for x in mats.values()], axis=0)
    support = next(iter(mats.values()))['support']
    fig, axs = plt.subplots(1, 3, figsize=(13, 4.7))
    for ax, V, t in zip(
        axs,
        [mean, proj, mean - proj],
        ['833 learned latent D', 'Best supported additive projection', 'Position x type residual']
    ):
        lim = np.nanmax(abs(V))
        im = ax.imshow(
            np.ma.masked_where(support == 0, V).T,
            aspect='auto',
            cmap='RdBu_r',
            vmin=-lim,
            vmax=lim
        )
        ax.set_yticks(range(16), PAIRS)
        ax.set_xticks([0, 4, 9, 14, 19], [1, 5, 10, 15, 20])
        ax.set_xlabel('Guide position (20 near PAM)')
        ax.set_title(t)
        fig.colorbar(im, ax=ax, shrink=0.7)
    save(fig, 'fig01_latent_decomposition')
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.4))
    for sd, g in svd.groupby('seed'):
        axs[0].plot(g['rank'], g.residual_explained, marker='o', alpha=0.65, label=str(sd))
    axs[0].set(
        xlabel='Rank added to additive latent D',
        ylabel='Fraction of residual variance recovered',
        ylim=(0, 1.03),
        title='Only fully supported positions 2-20'
    )
    axs[0].legend(title='Seed', fontsize=7)
    if len(examples):
        e = examples.iloc[0]
        j = PAIRS.index(e.type_a)
        k = PAIRS.index(e.type_b)
        diff = np.array([x['D'][:, j] - x['D'][:, k] for x in mats.values()])
        a = np.array([x['A'][:, j] - x['A'][:, k] for x in mats.values()])
        mu = diff.mean(0)
        sd = diff.std(0, ddof=1)
        pos = np.arange(1, 21)
        axs[1].errorbar(
            pos[1:],
            mu[1:],
            yerr=sd[1:],
            marker='o',
            capsize=2,
            label='833: type difference'
        )
        axs[1].plot(pos[1:], a.mean(0)[1:], '--', label='Additive projection')
        axs[1].axhline(0, c='gray', lw=0.8)
        axs[1].set(
            xlabel='Guide position',
            ylabel=e.type_a + ' minus ' + e.type_b + ' (latent D)',
            title='Same mismatch types can reverse their order'
        )
        axs[1].legend(fontsize=8)
    save(fig, 'fig02_rank_and_reversal')
    if len(perf):
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.3))
        q = perf[(perf.dataset == 'TrueOT_all_dedup') & perf.condition.str.startswith('alpha_')].copy()
        q['alpha'] = q.condition.str[6:].astype(float)
        q = q.groupby('alpha').agg(mean=('AUPRC', 'mean'), sd=('AUPRC', 'std')).reset_index()
        axs[0].errorbar(q.alpha, q['mean'], yerr=q.sd, marker='o', capsize=3)
        axs[0].set(
            xlabel='Retained latent residual alpha',
            ylabel='TrueOT guide-macro AUPRC',
            title='Frozen intervention; no retraining'
        )
        z = reported[(reported.cohort == 'original_protocol') & reported.strategy.eq('FactorizedD170_own')].set_index('dataset')
        full = reported[(reported.cohort == 'original_protocol') & reported.strategy.eq('Main_fullSelect_own')].set_index('dataset')
        names = [n for n in TESTS if n in z.index]
        delta = full.loc[
            names,
            'Macro_AUPRC_mean'].to_numpy() - z.loc[names,
            'Macro_AUPRC_mean'].to_numpy(
        )
        axs[1].bar(
            np.arange(len(names)),
            delta,
            color=['#226b92' if v > 0 else '#d37b31' for v in delta]
        )
        axs[1].axhline(0, c='gray', lw=0.8)
        axs[1].set_xticks(
            np.arange(len(names)),
            [n.replace('_test', '') for n in names],
            rotation=35,
            ha='right'
        )
        axs[1].set(ylabel='833 - retrained170 AUPRC', title='Existing separate-training comparison')
        save(fig, 'fig03_intervention_and_retraining')

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', default=str(ROOT))
    ap.add_argument('--out')
    ap.add_argument('--structural-only', action='store_true')
    ap.add_argument('--evaluate-seven', action='store_true')
    ap.add_argument('--trueot', type=Path)
    a = ap.parse_args()
    out = Path(a.out) if a.out else Path(a.root) / 'models/Factor170_833_CapacityAudit' / datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=True)
    raw = unpack()
    mats = {}
    records = []
    sv = []
    flips = []
    perfs = []
    pers = []
    audit = []
    for seed in SEEDS:
        d = raw['nodes'][raw['nodes'].seed.eq(seed)]
        D = grid(d, 'learned_D')
        C = grid(d, 'own_logit_contribution')
        S = grid(d, 'support')
        A, rank = additive(D, S)
        W, _ = additive(D, S, True)
        transform, err, pars = recover_transform(D, C)
        supported = S > 0
        R = D - A
        sst = float(np.sum((D[supported] - D[supported].mean()) ** 2))
        sse = float(np.sum(R[supported] ** 2))
        rect = A[2, 2] - A[2, 1] - A[1, 2] + A[1, 1]
        assert abs(rect) < 1e-10
        rev = 0
        for j, k in itertools.combinations(MM, 2):
            ok = (S[:, j] > 0) & (S[:, k] > 0)
            v = (D[:, j] - D[:, k])[ok]
            if v.min() < -0.0001 and v.max() > 0.0001:
                rev += 1
        records.append(dict(
            seed=seed,
            supported_cells=int(supported.sum()),
            supported_additive_design_rank=rank,
            raw_projection_RMSE=float(np.sqrt(np.mean(R[supported] ** 2))),
            raw_residual_variance_fraction=sse / sst,
            weighted_projection_RMSE=float(np.sqrt(np.average((D - W)[supported] ** 2, weights=S[supported]))),
                
                
                
            type_pairs_with_supported_reversal=rev,
            type_pairs_total=66,
            transform_recovery_max_error=err,
            recovered_input_scale=pars[0]
        ))
        B = D[1:]
        B0 = B.mean(1)[:, None] + B.mean(0)[None, :] - B.mean()
        U, v, V = np.linalg.svd(B - B0, full_matrices=False)
        for k in [0, 1, 2, 4, 8, 15]:
            sv.append(dict(
                seed=seed,
                rank=k,
                residual_explained=float(np.sum(v[:k] ** 2) / np.sum(v ** 2)),
                latent_RMSE=float(np.sqrt(np.mean((B - (B0 + U[:, :k] * v[:k] @ V[:k])) ** 2)))
            ))
        pam = np.full((2, 36), np.nan)
        for r in raw['pam'][raw['pam'].seed.eq(seed)].itertuples():
            pam[int(r.pam_position) - 2, int(r.code) - 1] = r.contribution
        coeffs = {}
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            name = 'alpha_' + str(alpha)
            G = np.full((20, 36), np.nan)
            G[:, np.array(CODES) - 1] = C if alpha == 1 else transform(A + alpha * R)
            coeffs[name] = G
        G = np.full((20, 36), np.nan)
        G[:, np.array(CODES) - 1] = transform(W)
        coeffs['weighted_projection'] = G
        mats[seed] = dict(D=D, A=A, W=W, support=S, coefficients=coeffs, pam=pam)
        for key, val in [('raw_D', D), ('additive_projection', A), ('residual', R)]:
            pd.DataFrame(
                val,
                index=range(1, 21),
                columns=PAIRS).to_csv(out / ('%s_seed%s.csv' % (key, seed))
            )
    for j, k in itertools.combinations(MM, 2):
        dif = np.array([mats[s]['D'][:, j] - mats[s]['D'][:, k] for s in SEEDS])
        ok = np.all(
            [(mats[s]['support'][:, j] > 0) & (mats[s]['support'][:, k] > 0) for s in SEEDS],
            axis=0
        )
        posplus = np.where(ok & (dif.min(0) > 0.0001))[0]
        posminus = np.where(ok & (dif.max(0) < -0.0001))[0]
        if len(posplus) and len(posminus):
            i = posplus[np.argmax(dif.mean(0)[posplus])]
            l = posminus[np.argmin(dif.mean(0)[posminus])]
            flips.append(dict(
                type_a=PAIRS[j],
                type_b=PAIRS[k],
                position_positive=i + 1,
                position_negative=l + 1,
                positive_mean=dif[:, i].mean(),
                negative_mean=dif[:, l].mean(),
                minimum_seed_margin=min(dif[:, i].min(), -dif[:, l].max())
            ))
    examples = pd.DataFrame(flips).sort_values(
        'minimum_seed_margin',
        ascending=False) if flips else pd.DataFrame(
    )
    stats = pd.DataFrame(records)
    svd = pd.DataFrame(sv)
    stats.to_csv(out / 'structural_summary.csv', index=False)
    svd.to_csv(out / 'residual_low_rank_curve.csv', index=False)
    examples.to_csv(out / 'five_seed_supported_order_reversals.csv', index=False)
    cohorts = []
    if a.trueot and not a.structural_only:
        external = pd.read_csv(a.trueot)
        required = {"input_key", "label", "guide"}
        if not required.issubset(external.columns):
            raise ValueError("Provide the previously filtered TrueOT CSV with input_key, label and guide")
        cohorts.append((
            "TrueOT_all_dedup",
            np.asarray([[int(c) for c in key.split(".")] for key in external.input_key], np.uint8),
            external.label.to_numpy(int),
            external.guide.to_numpy(object)
        ))
    root = Path(a.root)
    paths = [root / 'data/test_data' / (n + '.csv') for n in TESTS]
    run7 = not a.structural_only and (a.evaluate_seven or all((x.is_file() for x in paths)))
    if a.evaluate_seven and (not all((x.is_file() for x in paths))):
        raise FileNotFoundError('Missing seven-test CSVs; specify --root or omit --evaluate-seven')

    def run_cohort(name, x, y, g):
        print('EVALUATE', name, 'N=', len(y), flush=True)
        for seed in SEEDS:
            t = raw['thresholds']
            t = t[pd.to_numeric(
                t.seed,
                errors='coerce').eq(seed) & t.strategy.eq('Main_fullSelect_own'
            )]
            assert len(t) == 1
            threshold = float(t.threshold.iloc[0])
            q = mats[seed]
            for condition, G in q['coefficients'].items():
                z = score(x, G, q['pam'])
                met, pg = evaluate(y, z, g, threshold)
                perfs.append(dict(
                    dataset=name,
                    seed=seed,
                    condition=condition,
                    threshold_origin='original833_TTISS_locked',
                    **met
                ))
                pg = pg.assign(dataset=name, seed=seed, condition=condition)
                pers.append(pg)
    for c in cohorts:
        run_cohort(*c)
    if run7:
        for name, path, expected in zip(TESTS, paths, VALID):
            x, y, g, c = load_test(path)
            audit.append(dict(dataset=name, **c))
            run_cohort(name, x, y, g)
            assert perfs[-1]['Valid_guides'] == expected, (
                name,
                perfs[-1]['Valid_guides'],
                expected
            )
    if not perfs:
        figures(out, mats, stats, svd, examples, pd.DataFrame(), raw['reported'])
        (out / 'protocol.json').write_text(json.dumps(
            dict(neural_training=False, structural_only=True, datasets_evaluated=[]),
            indent=2
        ))
        print(stats.to_string(index=False))
        return
    perf = pd.DataFrame(perfs)
    perf.to_csv(out / 'frozen_intervention_seed_metrics.csv', index=False)
    pd.concat(
        pers,
        ignore_index=True).to_csv(out / 'frozen_intervention_per_guide.csv',
        index=False
    )
    agg = perf.groupby([
        'dataset',
        'condition'])[['AUPRC',
        'AP',
        'AUROC',
        'MCC_locked',
        'F1_locked',
        'ACC_locked']].agg(['mean',
        'std'
    ])
    agg.columns = ['_'.join(c) for c in agg.columns]
    agg.reset_index().to_csv(out / 'frozen_intervention_summary.csv', index=False)
    pg = pd.concat(pers, ignore_index=True)
    paired = []
    rng = np.random.default_rng(42)
    for dataset, df in pg.groupby('dataset'):
        q = df.groupby(['guide', 'condition']).AUPRC.mean().unstack()
        for c in [x for x in q if x != 'alpha_1.0']:
            delta = (q['alpha_1.0'] - q[c]).dropna().to_numpy()
            b = delta[rng.integers(0, len(delta), size=(5000, len(delta)))].mean(1)
            paired.append(dict(
                dataset=dataset,
                contrast='original833_minus_' + c,
                n_guides=len(delta),
                delta_AUPRC=delta.mean(),
                CI_low=np.quantile(b, 0.025),
                CI_high=np.quantile(b, 0.975),
                multiplicity_adjusted=False
            ))
    pd.DataFrame(paired).to_csv(out / 'paired_guide_intervention_CI.csv', index=False)
    pd.DataFrame(audit).to_csv(out / 'seven_input_audit.csv', index=False)
    figures(out, mats, stats, svd, examples, perf, raw['reported'])
    json.dump(
        dict(neural_training=False, embedded_analysis_payload_sha256=hashlib.sha256((Path(__file__).resolve().parents[1] / 'resources/factor_capacity_exports.json').read_bytes()).hexdigest(), source_checkpoint_sha256={str(r.seed): r.source_sha256 for r in raw['source_identity'].itertuples()}, primary_intervention='A+alpha*(D-A), alpha=[0,.25,.5,.75,1]; fixed shared downstream/PAM', projection_fit='only training-supported learned D cells; no test labels', retrained170_data='existing five-seed reported summary; no170 checkpoint assumed', evaluation_scope='local seven-test and/or supplied TrueOT inputs', classification_threshold='original833 TTISS locked; not refit for intervention', uncertainty='seed SD and descriptive guide-bootstrap intervals; no multiplicity correction', warning='position-by-type conditionality is not inter-position mismatch synergy'),
            
            
            
        open(out / 'protocol.json', 'w'),
        indent=2
    )
    lines = [
        '# 170 vs833：位置和类型是否可以分开？',
        '',
        '本次没有神经训练。170重训练结果引用此前汇总；投影833是独立的只读干预。',
        '',
        '## 已计算的结构结果',
        stats.to_string(index=False),
        '',
        '## 五种子共同存在的顺序翻转例子',
        examples.head(8).to_string(index=False),
        '',
        '## TrueOT只读干预结果',
        perf[perf.dataset.eq('TrueOT_all_dedup')].groupby('condition')[['AUPRC', 'AP', 'AUROC']].mean().to_string(),
            
            
            
        '',
        '170的D=mu+位置项+类型项。同属错配的类型强弱顺序不能跨位置改变；softplus虽然会产生非线性幅度变化，但单调性保留该顺序约束。833允许类型与位置绑定。',
        '位置×类型条件性不是两个位置之间的协同互作；不能用这个实验声称恢复了真实错配组合能量。',
        'alpha曲线是固定模型干预，不是重新训练的170；七集结果若存在应与TrueOT一起解释，不能只挑有利终点。',
        'MCC/F1/ACC沿用原833阈值，仅为固定阈值干预诊断，不等于投影模型重新校准后的最佳分类表现。'
    ]
    (out / '分析说明_CN.md').write_text('\n'.join(lines), encoding='utf-8')
    print(stats.to_string(index=False))
    print(perf.groupby(['dataset', 'condition']).AUPRC.mean().to_string())
    target = out / 'Factor170_833_analysis_results.zip'
    with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED) as z:
        for f in out.iterdir():
            if f.is_file() and f != target:
                z.write(f, f.name)
    print('DONE', target)
if __name__ == '__main__':
    main()
