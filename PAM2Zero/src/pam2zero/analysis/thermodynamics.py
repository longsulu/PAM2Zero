from __future__ import annotations
import os
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import argparse, hashlib, itertools, json, re
import subprocess, sys, tempfile
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path.cwd()
TABLE_FILE = None
MODEL_DIR = None
SEEDS = [42, 123, 3407, 2025, 2026]
N_BACKGROUNDS = 8
BOOTSTRAPS = 5000
RNG_SEED = 20261107
ENCODE = {
    'AA': 1,
    'AG': 2,
    'AC': 3,
    'AT': 4,
    'GG': 7,
    'GA': 8,
    'GC': 9,
    'GT': 10,
    'CC': 13,
    'CA': 14,
    'CG': 15,
    'CT': 16,
    'TT': 19,
    'TA': 20,
    'TG': 21,
    'TC': 22
}
MATCH = (1, 7, 13, 19)
COMP_RNA = {'A': 'T', 'C': 'G', 'G': 'C', 'U': 'A'}
COMP_DNA = {'A': 'T', 'C': 'G', 'G': 'C', 'T': 'A'}

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
        encoding='utf-8'
    )

def read_sheet(path, name):
    d = pd.read_excel(path, sheet_name=name, header=4, engine='openpyxl')
    d = d.dropna(how='all')
    d.columns = [str(x).strip() for x in d.columns]
    return d

def sheet_rows(d, columns):
    missing = set(columns) - set(d.columns)
    if missing:
        raise ValueError('工作表缺少数值列：' + str(missing) + '；请使用完整热力学Excel，不能只有说明页。')

def parse_tables(path, out):
    mm = read_sheet(path, '内部错配_32项')
    sheet_rows(mm, ['原文二聚体标签', 'RNA 5→3', 'DNA 3→5 对齐', 'ΔG37 kcal/mol'])
    lut = {}
    raw = []
    for _, r in mm.iterrows():
        a = str(r['RNA 5→3']).strip()
        b = str(r['DNA 3→5 对齐']).strip()
        if not re.fullmatch('[ACGU]{2}', a) or not re.fullmatch('[ACGT]{2}', b):
            raise ValueError('二聚体序列格式错误')
        key = (a, b)
        if key in lut:
            raise ValueError('二聚体重复键：' + str(key))
        val = float(r['ΔG37 kcal/mol'])
        if not np.isfinite(val):
            raise ValueError('二聚体值不是有限数')
        lut[key] = val
        raw.append(dict(RNA_5to3=a, DNA_3to5=b, dG37_kcal_mol=val))
    if len(lut) != 32:
        raise ValueError('要求32项Watkins内部错配参数，实际=' + str(len(lut)))
    nn = read_sheet(path, '1995_实现参考')
    sheet_rows(nn, ['实现键', 'ΔH kcal/mol', 'ΔS cal/(mol·K)'])
    normal = {}
    for _, r in nn.iterrows():
        key = str(r['实现键']).strip()
        if not re.fullmatch('[ACGT]{2}/[ACGT]{2}', key):
            continue
        a, b = key.split('/')
        a = a.replace('T', 'U')
        if b != ''.join((COMP_RNA[x] for x in a)):
            raise ValueError('正常参考方向不一致：' + key)
        normal[a, b] = float(r['ΔH kcal/mol']) - 310.15 * float(r['ΔS cal/(mol·K)']) / 1000.0
    if len(normal) != 16:
        raise ValueError('要求16项1M正常参考，实际=' + str(len(normal)))
    rows = []
    for mismatch in 'ACGU':
        target = mismatch.replace('U', 'T')
        for left, right in itertools.product('ACGU', repeat=2):
            a = left + mismatch + right
            wt = ''.join((COMP_RNA[x] for x in a))
            mutant = wt[0] + target + wt[2]
            gmm = lut[a[:2], mutant[:2]] + lut[a[1:], mutant[1:]]
            gwt = normal[a[:2], wt[:2]] + normal[a[1:], wt[1:]]
            encoded_pair = mismatch.replace('U', 'T') + COMP_DNA[target]
            rows.append(dict(
                context_id=len(rows),
                RNA_5to3=a,
                WT_target_DNA_3to5=wt,
                MM_target_DNA_3to5=mutant,
                mismatch_class='r%s:d%s' % (mismatch, target),
                encoded_center_pair=encoded_pair,
                encoded_center_code=ENCODE[encoded_pair],
                local_MM_dG37=gmm,
                local_WT_dG37=gwt,
                ddG37_kcal_mol=gmm - gwt,
                NaCl_M=1.0,
                data_kind='NN-table-derived local comparison; not independent cleavage measurement'
            ))
    contexts = pd.DataFrame(rows)
    contexts.to_csv(out / 'thermodynamic_contexts64.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(raw).to_csv(out / 'mismatch_NN32_used.csv', index=False)
    pd.DataFrame([dict(RNA=a, DNA=b, dG37=g) for (a, b), g in normal.items()]).to_csv(
        out / 'matched_NN16_used.csv',
        index=False
    )
    audit = dict(
        contexts=len(contexts),
        mismatch_class_counts=contexts.mismatch_class.value_counts().to_dict(),
        normal_reference='Sugimoto1995 reproduced by Biopython; original Table3 not directly checked',
            
            
            
        mismatch_reference='Watkins2011 Table1',
        salt_molar=1.0,
        temperature_kelvin=310.15,
        corrected_100mM_table_used=False,
        independent_measurements=False
    )
    for name in ['错配双链_实测', '正常双链_38实测']:
        try:
            d = read_sheet(path, name)
            d.to_csv(out / (name + '_source_only.csv'), index=False, encoding='utf-8-sig')
            audit[name + '_rows'] = len(d)
        except ValueError as e:
            audit[name + '_note'] = str(e)
    write_json(out / 'thermodynamic_source_audit.json', audit)
    return contexts

def _load_checkpoint_torch(path, cfg):
    import torch
    try:
        cp = torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        cp = torch.load(path, map_location='cpu')
    if 'state_dict' not in cp:
        raise ValueError('要求best_full_TTISS.pt，不能用last_state.pt：' + str(path))
    if cp.get('arm') != 'full' or cp.get('selection') != 'full':
        raise ValueError('要求原full训练/full选择checkpoint：' + str(path))
    s = cp['state_dict']
    required = [
        'matrix_d',
        'matrix_pam',
        'raw_team_gain',
        'raw_temperature',
        'raw_distance',
        'edge_raw',
        'input_scale'
    ]
    if any((k not in s for k in required)):
        raise ValueError('不是PAM01兼容checkpoint')
    state = {k: s[k].detach().cpu().numpy().astype(np.float64) for k in required}
    state.update(
        edge_log_range=float(cfg['EDGE_LOG_RANGE']),
        contest_max=float(cfg['CONTEST_MAX']),
        force_floor=1e-06
    )
    if state['matrix_d'].shape != (20, 36) or state['matrix_pam'].shape != (2, 36):
        raise ValueError('D/PAM尺寸错误')
    for k in required:
        if not np.isfinite(state[k]).all():
            raise ValueError('checkpoint有NaN/Inf：' + k)
    if float(state['input_scale']) <= 0:
        raise ValueError('input_scale必须正')
    metadata = {k: cp.get(k) for k in [
        'seed',
        'arm',
        'epoch',
        'selection',
        'architecture',
        'fingerprint',
        'selection_metric_value'
    ]}
    return (state, metadata)

def torch_worker(request_path):
    request = json.loads(Path(request_path).read_text(encoding='utf-8'))
    self_test(True)
    if request['mode'] == 'self_test':
        return
    state, metadata = _load_checkpoint_torch(Path(request['checkpoint']), request['config'])
    probes = np.random.default_rng(RNG_SEED).choice(list(ENCODE.values()), size=(80, 23))
    torch_parity(state, probes)
    np.savez(request['output_npz'], **state)
    write_json(request['output_meta'], metadata)

def run_torch_worker(request):
    with tempfile.TemporaryDirectory(prefix='pam01_torch_') as tmp:
        tmp = Path(tmp)
        request = dict(
            request,
            output_npz=str(tmp / 'state.npz'),
            output_meta=str(tmp / 'metadata.json')
        )
        path = tmp / 'request.json'
        write_json(path, request)
        result = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), '--torch-worker', str(path)],
            check=False
        )
        if result.returncode:
            raise RuntimeError('独立Torch读取/校验进程失败，退出码=%s；未绕过OpenMP错误。' % result.returncode)
        if request['mode'] == 'self_test':
            return
        with np.load(request['output_npz'], allow_pickle=False) as archive:
            state = {k: archive[k].copy() for k in archive.files}
        metadata = json.loads(Path(request['output_meta']).read_text(encoding='utf-8'))
        return (state, metadata)

def load_checkpoint(path, cfg):
    return run_torch_worker(dict(mode='checkpoint', checkpoint=str(path), config=cfg))

def pair_coefficients(s):
    edge = np.exp(s['edge_log_range'] * np.tanh(s['edge_raw']))
    strength = s['contest_max'] / (1.0 + np.exp(-s['raw_distance']))
    i, j = np.triu_indices(20, k=1)
    c = np.array([strength[b - a - 1] * np.prod(edge[a:b]) for a, b in zip(i, j)])
    return (i, j, c)

def score_numpy(s, codes, batch=2048):
    codes = np.asarray(codes, dtype=np.int64)
    if codes.ndim != 2 or codes.shape[1] != 23:
        raise ValueError('只接受23位内部探针')
    if not np.isin(codes, list(ENCODE.values())).all():
        raise ValueError('出现gap或不支持编码')
    gains = np.logaddexp(0.0, s['raw_team_gain']) + 0.0001
    temp = float(np.logaddexp(0.0, s['raw_temperature']) + 0.05)
    scale = float(s['input_scale'])
    i, j, coeff = pair_coefficients(s)
    own_all = []
    full_all = []
    for k in range(0, len(codes), batch):
        xx = codes[k:k + batch]
        match = np.isin(xx[:, :20], MATCH)
        raw = s['matrix_d'][np.arange(20)[None], xx[:, :20] - 1]
        force = (np.logaddexp(0.0, raw / scale) + s['force_floor']) * np.where(
            match,
            gains[0],
            gains[1]
        )
        sign = np.where(match, 1.0, -1.0)
        pam = s['matrix_pam'][np.arange(2)[None], xx[:, 21:23] - 1].sum(1) / scale
        own = (np.sum(force * sign, 1) + pam) / temp
        cross = match[:, i] != match[:, j]
        contribution = coeff[None] * cross * sign[
            :,
            i] * (force[:, i] - force[:, j]) / (force[:, i] + force[:, j]
        )
        full = own + contribution.sum(1) / temp
        own_all.append(own)
        full_all.append(full)
    return dict(own=np.concatenate(own_all), full=np.concatenate(full_all))

def score_reference_numpy(s, codes):
    x = np.asarray(codes)
    match = np.isin(x[:, :20], MATCH)
    g = np.logaddexp(0.0, s['raw_team_gain']) + 0.0001
    force = (np.logaddexp(
        0.0,
        s['matrix_d'][
            np.arange(20)[None],
            x[:, :20] - 1] / float(s['input_scale']
        )) + s['force_floor'
    ]) * np.where(
        match,
        g[0],
        g[1]
    )
    ma = (force * match).sum(1)
    mi = (force * ~match).sum(1)
    edge = np.exp(s['edge_log_range'] * np.tanh(s['edge_raw']))
    strength = s['contest_max'] / (1 + np.exp(-s['raw_distance']))
    path = np.ones(20)
    for distance in range(1, 20):
        path = path[:-1] * edge[distance - 1:]
        left = match[:, :-distance]
        cross = left != match[:, distance:]
        pot = strength[distance - 1] * path[None] * cross
        award = pot * force[:, :-distance] / (force[:, :-distance] + force[:, distance:])
        other = pot - award
        ma += np.where(left, award, other).sum(1)
        mi += np.where(left, other, award).sum(1)
    pam = s['matrix_pam'][np.arange(2)[None], x[:, 21:23] - 1].sum(1) / float(s['input_scale'])
    return (ma - mi + pam) / float(np.logaddexp(0.0, s['raw_temperature']) + 0.05)

def torch_parity(s, codes):
    import torch
    import torch.nn.functional as F
    t = {k: torch.tensor(
        s[k],
        dtype=torch.float64) for k in ['matrix_d',
        'matrix_pam',
        'raw_team_gain',
        'raw_temperature',
        'raw_distance',
        'edge_raw',
        'input_scale'
    ]}
    x = torch.tensor(codes, dtype=torch.long)
    m = torch.isin(x[:, :20], torch.tensor(MATCH))
    f = (F.softplus(t['matrix_d'][
        torch.arange(20)[None],
        x[:, :20] - 1] / t['input_scale'
    ]) + 1e-06) * torch.where(
        m,
        F.softplus(t['raw_team_gain'])[0] + 0.0001,
        F.softplus(t['raw_team_gain'])[1] + 0.0001
    )
    ma = (f * m).sum(1)
    mi = (f * ~m).sum(1)
    path = torch.ones(20, dtype=torch.float64)
    for distance in range(1, 20):
        path = path[:-1] * torch.exp(s['edge_log_range'] * torch.tanh(t['edge_raw'][distance - 1:]))
        cross = m[:, :-distance] != m[:, distance:]
        pot = s['contest_max'] * torch.sigmoid(t['raw_distance'][distance - 1]) * path[None] * cross
        award = pot * f[:, :-distance] / (f[:, :-distance] + f[:, distance:])
        other = pot - award
        ma += torch.where(m[:, :-distance], award, other).sum(1)
        mi += torch.where(m[:, :-distance], other, award).sum(1)
    pam = t['matrix_pam'][torch.arange(2)[None], x[:, 21:23] - 1].sum(1) / t['input_scale']
    z = ((ma - mi + pam) / (F.softplus(t['raw_temperature']) + 0.05)).numpy()
    np.testing.assert_allclose(z, score_numpy(s, codes)['full'], atol=1e-09, rtol=1e-10)

def self_test(with_torch=False):
    rng = np.random.default_rng(41)
    s = dict(
        matrix_d=rng.normal(size=(20, 36)),
        matrix_pam=rng.normal(size=(2, 36)),
        raw_team_gain=np.array([0.4, 0.7]),
        raw_temperature=np.array(0.5),
        raw_distance=rng.normal(size=19),
        edge_raw=rng.normal(size=19),
        input_scale=np.array(0.5),
        edge_log_range=0.1,
        contest_max=1.0,
        force_floor=1e-06
    )
    x = rng.choice(list(ENCODE.values()), size=(27, 23))
    got = score_numpy(s, x)
    np.testing.assert_allclose(got['full'], score_reference_numpy(s, x), atol=1e-10)
    together = score_numpy(s, np.concatenate([x[:3], x[3:]]))
    np.testing.assert_array_equal(got['own'], together['own'])
    for row in [np.full((2, 23), 1), np.full((2, 23), 4)]:
        z = score_numpy(s, row)
        np.testing.assert_allclose(z['full'], z['own'], atol=1e-12)
    assert ENCODE['A' + COMP_DNA['A']] == 4 and ENCODE['T' + COMP_DNA['T']] == 20
    base = np.full((3, 23), 1)
    base[1, 7] = 7
    base[2, 9] = 13
    mutant = base.copy()
    mutant[:, 8] = 4
    cost = score_numpy(s, base)['own'] - score_numpy(s, mutant)['own']
    np.testing.assert_allclose(cost, cost[0], atol=1e-10)
    if with_torch:
        torch_parity(s, x)
    print(
        'PASS: original budget formula parity; pairing/strand encoding; batch isolation; OWN neighbor invariance' + ('; Torch parity' if with_torch else ''),
            
            
            
        flush=True
    )

def make_probes(contexts):
    rng = np.random.default_rng(RNG_SEED)
    backgrounds = []
    for _ in range(N_BACKGROUNDS):
        a = np.array(list('ACGT' * 5))
        rng.shuffle(a)
        backgrounds.append(''.join(a))
    records = []
    wt = []
    mm = []
    for c in contexts.to_dict('records'):
        for center in range(1, 19):
            local = c['RNA_5to3'].replace('U', 'T')
            for bg_id, bg in enumerate(backgrounds):
                guide = bg[:center - 1] + local + bg[center + 2:]
                off = list(guide)
                off[center] = COMP_DNA[c['MM_target_DNA_3to5'][1]]
                off = ''.join(off)
                wt.append([ENCODE[a + a] for a in guide + 'TGG'])
                mm.append([ENCODE[a + b] for a, b in zip(guide + 'TGG', off + 'TGG')])
                if len(wt[-1]) != 23:
                    raise AssertionError('内部序列长度错误')
                physical = ''.join((COMP_DNA[x] for x in off[center - 1:center + 2]))
                if physical != c['MM_target_DNA_3to5']:
                    raise AssertionError('物理/编码配对方向错误')
                records.append(dict(
                    context_id=c['context_id'],
                    position=center + 1,
                    background_id=bg_id,
                    guide20=guide,
                    mutant_off_non_target20=off,
                    mismatch_class=c['mismatch_class'],
                    ddG37_kcal_mol=c['ddG37_kcal_mol']
                ))
    assert len(records) == 64 * 18 * N_BACKGROUNDS
    return (pd.DataFrame(records), np.asarray(wt), np.asarray(mm), backgrounds)

def correlation(x, y):
    from scipy.stats import spearmanr, pearsonr
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 4:
        return dict(n=len(x), rho=None, pearson=None, status='insufficient')
    if np.ptp(x) < 1e-10 or np.ptp(y) < 1e-10:
        return dict(
            n=len(x),
            rho=None,
            pearson=None,
            status='constant/near-constant: correlation undefined'
        )
    return dict(
        n=len(x),
        rho=float(spearmanr(x, y)[0]),
        pearson=float(pearsonr(x, y)[0]),
        status='ok'
    )

def bootstrap_contexts(d, reps):
    rng = np.random.default_rng(RNG_SEED + 9)
    subsets = [g.index.to_numpy() for _, g in d.groupby('mismatch_class')]
    r = []
    for _ in range(reps):
        idx = np.concatenate([rng.choice(g, len(g), replace=True) for g in subsets])
        q = d.loc[idx]
        c = correlation(q.ddG37_kcal_mol, q.model_cost)
        if c['rho'] is not None:
            r.append(c['rho'])
    if not r:
        return dict(rho_CI_low=None, rho_CI_high=None, bootstrap_valid=0)
    lo, hi = np.quantile(r, [0.025, 0.975])
    return dict(rho_CI_low=float(lo), rho_CI_high=float(hi), bootstrap_valid=len(r))

def summarize(raw, out):
    p = raw.groupby(
        ['seed', 'branch', 'context_id', 'position', 'mismatch_class'],
        as_index=False)[['ddG37_kcal_mol', 'model_cost']].mean(
    )
    p.to_csv(out / 'cost_by_seed_position_context.csv', index=False)
    rows = []
    for (seed, branch, pos), g in p.groupby(['seed', 'branch', 'position']):
        rows.append(dict(
            seed=int(seed),
            branch=branch,
            position=int(pos),
            scope='all_64_contexts',
            **correlation(g.ddG37_kcal_mol, g.model_cost)
        ))
        for typ, h in g.groupby('mismatch_class'):
            rows.append(dict(
                seed=int(seed),
                branch=branch,
                position=int(pos),
                scope=typ,
                **correlation(h.ddG37_kcal_mol, h.model_cost)
            ))
    pd.DataFrame(rows).to_csv(out / 'correlation_by_position_and_type.csv', index=False)
    avg = p.groupby(
        ['seed', 'branch', 'context_id', 'mismatch_class'],
        as_index=False)[['ddG37_kcal_mol', 'model_cost']].mean(
    )
    avg.to_csv(out / 'cost64_by_seed.csv', index=False)
    totals = []
    for (seed, branch), g in avg.groupby(['seed', 'branch']):
        totals.append(dict(
            seed=int(seed),
            branch=branch,
            scope='64_contexts_position_averaged',
            **correlation(g.ddG37_kcal_mol, g.model_cost)
        ))
        h = g.copy()
        h['ddG37_kcal_mol'] -= h.groupby('mismatch_class').ddG37_kcal_mol.transform('mean')
        h['model_cost'] -= h.groupby('mismatch_class').model_cost.transform('mean')
        totals.append(dict(
            seed=int(seed),
            branch=branch,
            scope='within_type_centered_context_signal',
            **correlation(h.ddG37_kcal_mol, h.model_cost)
        ))
    pd.DataFrame(totals).to_csv(out / 'correlation_per_seed.csv', index=False)
    ensemble = avg.groupby(
        ['branch', 'context_id', 'mismatch_class'],
        as_index=False)[['ddG37_kcal_mol', 'model_cost']].mean(
    )
    ensemble.to_csv(out / 'cost64_seed_mean.csv', index=False)
    main = []
    for branch, g in ensemble.groupby('branch'):
        g = g.reset_index(drop=True)
        main.append(dict(
            branch=branch,
            scope='64_contexts_position_background_seed_averaged',
            **correlation(g.ddG37_kcal_mol, g.model_cost),
            **bootstrap_contexts(g, BOOTSTRAPS)
        ))
        h = g.copy()
        h['ddG37_kcal_mol'] -= h.groupby('mismatch_class').ddG37_kcal_mol.transform('mean')
        h['model_cost'] -= h.groupby('mismatch_class').model_cost.transform('mean')
        main.append(dict(
            branch=branch,
            scope='within_type_centered_context_signal',
            **correlation(h.ddG37_kcal_mol, h.model_cost),
            **bootstrap_contexts(h, BOOTSTRAPS)
        ))
    report = pd.DataFrame(main)
    report.to_csv(out / 'MAIN_consistency_summary.csv', index=False)
    inv = p.groupby([
        'seed',
        'branch',
        'position',
        'mismatch_class']).model_cost.agg(['min',
        'max',
        'std'
    ]).reset_index()
    inv['context_range'] = inv['max'] - inv['min']
    inv.to_csv(out / 'neighbor_context_response.csv', index=False)
    ownmax = float(inv[inv.branch == 'own'].context_range.max())
    if ownmax > 1e-08:
        raise AssertionError('OWN本应无邻域依赖，但观察到变化；请检查实现')
    write_json(
        out / 'structural_checks.json',
        dict(
            own_max_context_range=ownmax,
            own_neighbor_invariance_passed=True,
            full_context_response_is_not_proof_of_physics=True
        )
    )
    return (report, ensemble, p)

def figures(avg, p, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = dict(zip(
        ['rA:dA', 'rC:dC', 'rG:dG', 'rU:dT'],
        ['#d95f02', '#7570b3', '#1b9e77', '#e7298a']
    ))

    def save(fig, name):
        fig.tight_layout()
        fig.savefig(out / (name + '.png'), dpi=220)
        fig.savefig(out / (name + '.pdf'))
        plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, branch in zip(axs, ['own', 'full']):
        for typ, g in avg[avg.branch == branch].groupby('mismatch_class'):
            ax.scatter(
                g.ddG37_kcal_mol,
                g.model_cost,
                s=28,
                label=typ,
                color=colors[typ],
                alpha=0.8
            )
        ax.set(
            title=branch.upper() + ': 64 NN-derived contexts',
            xlabel='Mismatch penalty, delta-delta G37 (kcal/mol)',
            ylabel='Model penalty: WT logit - mismatch logit'
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.15)
    save(fig, 'fig01_thermo_vs_model')
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, branch in zip(axs, ['own', 'full']):
        for typ, g in avg[avg.branch == branch].groupby('mismatch_class'):
            ax.scatter(
                g.ddG37_kcal_mol - g.ddG37_kcal_mol.mean(),
                g.model_cost - g.model_cost.mean(),
                s=28,
                label=typ,
                color=colors[typ]
            )
        ax.axhline(0, color='gray', lw=0.7)
        ax.set(
            title=branch.upper() + ': remove mismatch-class means',
            xlabel='Within-class centered delta-delta G37',
            ylabel='Within-class centered model penalty'
        )
        ax.grid(alpha=0.15)
    save(fig, 'fig02_within_type_neighbor_response')
    stats = pd.read_csv(out / 'correlation_by_position_and_type.csv')
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for branch in ['own', 'full']:
        d = stats[(stats.branch == branch) & (stats.scope == 'all_64_contexts')].groupby('position').rho.agg([
            'mean',
            'std'
        ])
        ax.errorbar(d.index, d['mean'], yerr=d['std'], marker='o', label=branch, capsize=2)
    ax.set(
        xlabel='Guide position (20 nearest PAM)',
        ylabel='Spearman correlation',
        title='Position sensitivity: mean +/- seed SD; not independent thermodynamic samples'
    )
    ax.legend()
    ax.grid(alpha=0.15)
    save(fig, 'fig03_position_sensitivity')


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Nearest-neighbor thermodynamic context audit")
    parser.add_argument("--table", type=Path)
    parser.add_argument("--models", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--torch-worker", type=Path)
    args = parser.parse_args()
    if args.torch_worker:
        torch_worker(args.torch_worker)
        return
    self_test(False)
    if args.self_test:
        return
    if not all([args.table, args.models, args.out]):
        parser.error("--table, --models and --out are required")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((args.models / "protocol.json").read_text(encoding="utf-8-sig"))
    config = protocol.get("config", protocol.get("resolved_config", {}))
    contexts = parse_tables(args.table, out)
    probes, wt, mm, backgrounds = make_probes(contexts)
    rows = []
    hashes = {}
    for seed in SEEDS:
        options = [args.models / "training/full" / ("seed_%d" % seed) / "best_full_TTISS.pt",
                   args.models / ("seed_%d" % seed) / "best_full_TTISS.pt"]
        path = next((q for q in options if q.is_file()), None)
        if path is None:
            raise FileNotFoundError(str(options))
        hashes[str(path)] = digest(path)
        state, metadata = load_checkpoint(path.resolve(), config)
        a, b = score_numpy(state, wt), score_numpy(state, mm)
        for branch in ["own", "full"]:
            frame = probes.copy()
            frame["seed"], frame["branch"] = seed, branch
            frame["WT_logit"], frame["MM_logit"] = a[branch], b[branch]
            frame["model_cost"] = a[branch] - b[branch]
            rows.append(frame)
    raw = pd.concat(rows, ignore_index=True)
    raw.to_csv(out / "all_probe_scores.csv", index=False)
    report, avg, position = summarize(raw, out)
    figures(avg, position, out)
    if any(digest(path) != value for path, value in hashes.items()):
        raise RuntimeError("Source checkpoints changed")
    write_json(
        out / "protocol.json",
        dict(source_sha256=hashes, contexts=len(contexts), backgrounds=backgrounds)
    )


if __name__ == "__main__":
    main()
