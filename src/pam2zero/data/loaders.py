from __future__ import annotations
from pam2zero import config as cfg
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pickle
import torch
from sklearn.model_selection import train_test_split


def _read_pkl_bundle(base_path):
    out = {}
    for feature in ('pos_list', 'labels', 'sg_list'):
        full = Path(str(base_path).replace('.pkl', f'_{feature}.pkl'))
        if not full.is_file():
            raise FileNotFoundError(f'缺少真实 PKL 文件: {full}')
        with open(full, 'rb') as f:
            out[feature] = np.asarray(pickle.load(f)).squeeze()
        if feature == 'pos_list' and out[feature].ndim == 1 and (out[feature].shape == (24,)):
            out[feature] = out[feature].reshape(1, 24)
        elif feature != 'pos_list':
            out[feature] = out[feature].reshape(-1)
    return out


def _load_train_ttiss_original():
    from pam2zero.data.encoding import _off_target_mask, _sg_str, check_codes, rule_B
    print('=' * 92)
    print('LOAD ALL CHANGEseq + TTISS; internal split only for compatibility')
    print('=' * 92)
    train_raw = _read_pkl_bundle(cfg.TRAIN_DATA_PATH)
    val_raw = _read_pkl_bundle(cfg.VAL_DATA_PATH)
    codes_raw = np.asarray(train_raw['pos_list'])
    labels_raw = np.asarray(train_raw['labels'], dtype=np.int64)
    sg_raw = np.asarray(train_raw['sg_list'])
    if len(codes_raw) != len(labels_raw) or len(codes_raw) != len(sg_raw):
        raise ValueError('CHANGEseq pos/labels/sg 的行数不一致')
    np.random.seed(42)
    tr_idx, inter_idx = train_test_split(
        np.arange(len(labels_raw)),
        test_size=0.2,
        stratify=labels_raw
    )
    tr_keep = _off_target_mask(codes_raw[tr_idx])
    inter_keep = _off_target_mask(codes_raw[inter_idx])
    # Recombine both subsets to retain every off-target training row.
    kept_ids = np.concatenate([tr_idx[tr_keep], inter_idx[inter_keep]])
    train_codes = np.asarray(codes_raw[kept_ids], dtype=np.uint8)
    train_labels = np.asarray(labels_raw[kept_ids], dtype=np.int64)
    train_sg = np.asarray([rule_B(s) for s in sg_raw[kept_ids]], dtype=object)
    if len(train_sg) != len(train_labels):
        raise RuntimeError('train sgRNA alignment FAILED')
    train_gid, unique_groups = pd.factorize(train_sg, sort=True)
    train_gid = train_gid.astype(np.int32, copy=False)
    if (train_gid < 0).any():
        raise RuntimeError('训练 sgRNA 含缺失分组')
    v_keep = _off_target_mask(val_raw['pos_list'])
    val_codes = np.asarray(val_raw['pos_list'][v_keep], dtype=np.uint8)
    val_labels = np.asarray(val_raw['labels'][v_keep], dtype=np.int64)
    val_groups = np.asarray([_sg_str(s) for s in val_raw['sg_list'][v_keep]], dtype=object)
    check_codes('CHANGEseq', train_codes)
    check_codes('TTISS', val_codes)
    print(f'CHANGEseq 原始 {len(labels_raw):,} -> 移除 on-target 后 {len(train_labels):,}')
    print(f'正样本 {train_labels.sum():,} / {len(train_labels):,} = {train_labels.mean():.6%}')
    print(f'训练 sgRNA 组数 {len(unique_groups):,}，TTISS 原协议组数 {len(np.unique(val_groups)):,}')
    print('TTISS 只用于学习率调整、保存最优模型和早停')
    return (
        train_codes,
        train_labels,
        train_gid,
        np.asarray(unique_groups, dtype=str),
        val_codes,
        val_labels,
        val_groups
    )


def compute_group_weights(group_ids, groups, power, floor, cap, out_dir):
    counts = np.bincount(group_ids, minlength=len(groups)).astype(np.int64)
    if np.any(counts == 0):
        raise ValueError('空训练组')
    ref = float(np.median(counts))
    raw = (ref / counts.astype(np.float64)) ** float(power)
    raw = np.clip(raw, float(floor), float(cap))
    normalization = float(np.dot(raw, counts) / counts.sum())
    weights = (raw / normalization).astype(np.float32)
    weighted_mass = counts.astype(np.float64) * weights
    weighted_mass /= weighted_mass.sum()
    stats = pd.DataFrame({
        'sgRNA_group': groups.astype(str),
        'n_samples': counts,
        'sample_fraction': counts / counts.sum(),
        'group_weight': weights,
        'weighted_loss_mass': weighted_mass
    })
    stats.sort_values(
        'n_samples',
        ascending=False).to_csv(out_dir / 'train_group_weights.csv',
        index=False,
        encoding='utf-8-sig'
    )
    print(f'训练组权重: power={power}, raw clip=[{floor}, {cap}], sample-mean(weight)={float((weights * counts).sum() / counts.sum()):.6f}')
    print(f'每组样本数: min={counts.min():,}, median={np.median(counts):.0f}, max={counts.max():,}')
    print(f'归一化后权重: min={weights.min():.5f}, max={weights.max():.5f}')
    print(f'损失份额: min={weighted_mass.min():.5f}, max={weighted_mass.max():.5f}; ideal={1 / len(groups):.5f}')
    return torch.tensor(weights, dtype=torch.float32)


def _load_test_csv_original(csv_path):
    from pam2zero.data.encoding import _sg_str, basepair_2_id_encode_original, check_codes
    df = pd.read_csv(csv_path)
    for col in ('DNA', 'sg', 'label'):
        if col not in df.columns:
            raise ValueError(f'{csv_path} 缺少列 {col}')
    codes, labels, groups = ([], [], [])
    removed = 0
    for sg, off, y in df[['sg', 'DNA', 'label']].itertuples(index=False, name=None):
        sg, off = (_sg_str(sg), _sg_str(off))
        if sg[:-3] == off[:-3]:
            removed += 1
            continue
        codes.append(basepair_2_id_encode_original(sg, off))
        labels.append(int(y))
        groups.append(sg[:-3])
    codes = np.asarray(codes, dtype=np.uint8)
    labels = np.asarray(labels, dtype=np.int64)
    groups = np.asarray(groups, dtype=object)
    check_codes(csv_path.stem, codes)
    print(f'{csv_path.stem}: N={len(labels):,}, positive={labels.sum():,}, groups={len(np.unique(groups)):,}, on_target_removed={removed:,}')
    return (codes, labels, groups)


def load_train_ttiss_balanced():
    from pam2zero.data.encoding import normalize_to23
    tr, y, gid, groups, val, vy, vg = _load_train_ttiss_original()
    tr, keep = normalize_to23(tr, 'CHANGEseq')
    y = y[keep]
    sg = np.asarray(groups)[gid[keep]]
    gid, groups = pd.factorize(sg, sort=True)
    val, keep = normalize_to23(val, 'TTISS')
    vy, vg = (vy[keep], vg[keep])
    if len(np.unique(y)) < 2 or len(np.unique(vy)) < 2:
        raise ValueError('23位置过滤后训练/验证标签必须同时包含正负类')
    return (tr, y, gid.astype(np.int32), np.asarray(groups, dtype=str), val, vy, vg)


def load_test_csv_original(csv_path):
    from pam2zero.data.encoding import normalize_to23
    codes, y, groups = _load_test_csv_original(csv_path)
    codes, keep = normalize_to23(codes, csv_path.stem)
    return (codes, y[keep], groups[keep])


def read_sample_table(path, sheet=None, load=True):
    from pam2zero.data.encoding import identify_columns
    suffix = path.suffix.lower()
    if suffix == '.csv':
        frame = pd.read_csv(path, low_memory=False, nrows=None if load else 4)
        cols = identify_columns(frame)
        return (frame, dict(format='csv', sheet=None, header_excel_row=1, columns=list(cols)))
    if suffix not in ('.xlsx', '.xlsm'):
        raise ValueError('只支持csv/xlsx/xlsm：' + str(path))
    try:
        book = pd.ExcelFile(path, engine='openpyxl')
    except ImportError:
        raise RuntimeError('读取Excel需要openpyxl，请在CRISPR_GNN环境运行：python -m pip install openpyxl')
    candidates = []
    inspected = []
    with book:
        sheets = [sheet] if sheet else book.sheet_names
        for sh in sheets:
            if sh not in book.sheet_names:
                raise ValueError('不存在工作表%s；可选：%s' % (sh, book.sheet_names))
            preview = pd.read_excel(book, sheet_name=sh, header=None, nrows=50)
            inspected.append(dict(
                sheet=sh,
                first_rows=preview.head(6).fillna('').astype(str).values.tolist()
            ))
            for header in range(len(preview)):
                names = [str(v).strip() if not pd.isna(v) else '' for v in preview.iloc[header]]
                try:
                    cols = identify_columns(pd.DataFrame(columns=names))
                except ValueError:
                    continue
                candidates.append((sh, header, cols))
                break
        if not candidates:
            raise ValueError('未找到含guide、DNA、label三列的样本工作表。可能是Table S1来源表，或需要设置GUIDE_COLUMN/DNA_COLUMN/LABEL_COLUMN。表头预览：' + json.dumps(
                inspected,
                ensure_ascii=False
            ))
        if len(candidates) != 1:
            raise ValueError('多个工作表都含样本列，不能静默合并重复版本：%s。请用 --sheet 指定，并配合 --dataset 指定文件。' % [
                (s, h + 1) for s,
                h,
                c in candidates
            ])
        sh, header, cols = candidates[0]
        frame = pd.read_excel(book, sheet_name=sh, header=header, nrows=None if load else 4)
    return (frame, dict(format=suffix, sheet=sh, header_excel_row=header + 1, columns=list(cols)))


def prepare_dataset(path, out, sheet=None):
    from pam2zero.data.encoding import clean_sequence, encode_pair, identify_columns
    df, table_meta = read_sample_table(path, sheet=sheet)
    gc, dc, lc = identify_columns(df)
    if df.empty:
        raise ValueError('样本工作表为空：' + str(path))
    bulges = pd.to_numeric(df['#Bulge'], errors='coerce') if '#Bulge' in df else None
    audits = []
    records = []
    for i, (s, o, label) in enumerate(df[[gc, dc, lc]].itertuples(index=False, name=None)):
        sg, off = (clean_sequence(s), clean_sequence(o))
        reason = ''
        filled = False
        try:
            yy = float(label)
            if not np.isfinite(yy) or yy not in (0.0, 1.0):
                raise ValueError()
            y = int(yy)
        except (ValueError, TypeError):
            y = -1
            reason = 'invalid_binary_label'
        if not reason and bulges is not None and (not pd.isna(bulges.iloc[i])) and (bulges.iloc[i] > 0):
            reason = 'gap_or_bulge'
        if not reason and any((c in sg + off for c in '-_')):
            reason = 'gap_or_bulge'
        if not reason and len(sg) == 20 and cfg.GUIDE20_REFERENCE_PAM:
            sg += cfg.GUIDE20_REFERENCE_PAM
            filled = True
        if not reason and (len(sg) != 23 or len(off) != 23):
            reason = 'requires_guide20_PAM3_and_DNA20_PAM3'
        if not reason and (set(sg[:20]) - set('ACGT') or set(off) - set('ACGT') or set(sg[20:]) - set('ACGTN')):
            reason = 'ambiguous_or_invalid_base'
        if not reason and sg[:20] == off[:20]:
            reason = 'on_target'
        entry = dict(
            csv_row=i + table_meta['header_excel_row'] + 1,
            source_sheet=table_meta['sheet'],
            guide=sg[:20],
            sgRNA=sg,
            DNA=off,
            label=y,
            reference_PAM_filled=filled,
            reason=reason
        )
        for source_col in [
            'Dataset',
            'chromosome',
            'start',
            'end',
            'Cell type',
            'Delivery of Cas9',
            'Delivery of gRNA',
            'Alignment Pattern',
            '#MM',
            '#Bulge',
            'Editing efficiency in original study',
            '#Identical labels/#Total Occurrence'
        ]:
            if source_col in df:
                entry['source_' + source_col] = df.iloc[i][source_col]
        if not reason:
            code = encode_pair(sg, off)
            entry['input_key'] = '.'.join(map(str, code))
            records.append((len(audits), code, sg[:20], y))
        audits.append(entry)
    audit = pd.DataFrame(audits)
    eligible = audit[audit.reason.eq('')]
    conflict = set(eligible.groupby('input_key').label.nunique().loc[lambda x: x > 1].index)
    seen = set()
    codes = []
    labels = []
    groups = []
    rows = []
    for ai, code, guide, y in records:
        key = audit.at[ai, 'input_key']
        if key in conflict:
            audit.at[ai, 'reason'] = 'conflicting_labels_same_full_input'
            continue
        if key in seen:
            audit.at[ai, 'reason'] = 'duplicate_same_full_input'
            continue
        seen.add(key)
        audit.at[ai, 'reason'] = 'kept'
        codes.append(code)
        labels.append(y)
        groups.append(guide)
        rows.append(ai)
    out.mkdir(parents=True, exist_ok=True)
    audit.to_csv(out / 'row_filter_and_dedup_audit.csv', index=False, encoding='utf-8-sig')
    stats = dict(
        input_rows=len(df),
        kept_rows=len(codes),
        conflicting_full_input_keys=len(conflict),
        inferred_reference_PAM_rows=int(audit.reference_PAM_filled.sum()),
        reason_counts=audit.reason.value_counts().to_dict(),
        detected_columns=dict(guide=gc, DNA=dc, label=lc),
        source_table=table_meta
    )
    (out / 'dataset_audit.json').write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding='utf-8'
    )
    print(path.name, stats, flush=True)
    if not codes:
        raise ValueError('过滤后没有可评估样本；请看 row_filter_and_dedup_audit.csv')
    kept = audit.iloc[rows].reset_index(drop=True)
    return (
        np.array(codes, dtype=np.uint8),
        np.array(labels, dtype=np.int64),
        np.array(groups),
        kept
    )


def reference_guides(root, skip=False):
    from pam2zero.data.encoding import clean_sequence
    from pam2zero.utils.io import sha256
    if skip:
        return ({}, {})
    paths = {
        'CHANGEseq': root / 'data/train_data/CHANGEseq_encoded_sg_list.pkl',
        'TTISS': root / 'data/val_data/ttiss_encoded_sg_list.pkl'
    }
    sets = {}
    hashes = {}
    for name, p in paths.items():
        if not p.is_file():
            raise FileNotFoundError('独立性检查需要 %s；如明确只做未审计测试，可加 --skip-overlap。' % p)
        with open(p, 'rb') as f:
            arr = pickle.load(f)
        guides = set()
        for s in np.asarray(arr).reshape(-1):
            if isinstance(s, bytes):
                s = s.decode('utf-8')
            s = clean_sequence(s)
            guide = s[:-3] if len(s) > 20 else s
            if len(guide) != 20 or set(guide) - set('ACGT'):
                raise ValueError('%s含非标准20位guide：%s' % (p, guide))
            guides.add(guide)
        sets[name] = guides
        hashes[str(p)] = sha256(p)
    for name in [
        'CRISPOR_test',
        'GUIDEseq_test',
        'HEK293T_test',
        'II6_test',
        'K562_test',
        'guideseqCT_test',
        'pdh_test'
    ]:
        p = root / 'data/test_data' / (name + '.csv')
        if not p.is_file():
            continue
        df = pd.read_csv(p, usecols=lambda c: c == 'sg')
        if 'sg' not in df:
            raise ValueError('%s缺少sg列' % p)
        sets[name] = {clean_sequence(s)[:-3] for s in df.sg}
        hashes[str(p)] = sha256(p)
    return (sets, hashes)


def overlap_cohorts(groups, sets, out):
    if not sets:
        return {'all_dedup_UNAUDITED': np.ones(len(groups), dtype=bool)}
    rows = []
    for g in np.unique(groups):
        row = {'guide': g}
        for name, refs in sets.items():
            a = np.array([list(s) for s in sorted(refs)])
            dist = np.count_nonzero(a != np.array(list(g))[None, :], axis=1)
            row[name + '_min_mismatches'] = int(dist.min()) if len(dist) else 20
            row[name + '_exact'] = bool(g in refs)
        rows.append(row)
    audit = pd.DataFrame(rows).set_index('guide')
    audit.to_csv(out / 'guide_overlap_audit.csv', encoding='utf-8-sig')
    clean_train = (audit.CHANGEseq_min_mismatches > 2) & (audit.TTISS_min_mismatches > 2)
    clean_all = clean_train.copy()
    for name in sets:
        clean_all &= audit[name + '_min_mismatches'] > 2
    return {
        'all_dedup': np.ones(len(groups), dtype=bool),
        'unseen_train_val_90pct': np.array([clean_train.loc[g] for g in groups]),
        'unseen_all_available_90pct': np.array([clean_all.loc[g] for g in groups])
    }


def dataset_identity_audit(tr, train_groups, val, vg):
    from pam2zero.utils.io import json_write
    out = cfg.OUT_DIR / 'data_identity'
    out.mkdir(exist_ok=True)
    decode = {v: k for k, v in cfg.ENCODE_DICT_36.items()}
    lut = np.array([decode[i][0] for i in range(1, 37)])

    def guides(x):
        return np.array([''.join(r) for r in lut[x[:, :20].astype(int) - 1]], dtype=object)
    trainrefs = set(guides(tr))
    valrefs = set(guides(val))
    rows = []

    def unique_inputs(x):
        return np.unique(np.ascontiguousarray(x).view(np.dtype((np.void, 23))).reshape(-1))
    trainkeys = unique_inputs(tr)
    valkeys = unique_inputs(val)
    for name, x in [('TTISS', val)] + [(n, None) for n in cfg.TEST_NAMES]:
        if x is None:
            x, _, _ = load_test_csv_original(cfg.TEST_DIR / (name + '.csv'))
        gs = guides(x)
        keys = np.ascontiguousarray(x).view(np.dtype((np.void, 23))).reshape(-1)
        overlap_train = np.isin(keys, trainkeys)
        overlap_val = np.isin(keys, valkeys)
        rows.append(dict(
            dataset=name,
            N=len(x),
            unique_guide20=len(set(gs)),
            guide20_overlap_train=len(set(gs) & trainrefs),
            guide20_overlap_TTISS=len(set(gs) & valrefs),
            rows_exact_input_in_train=int(overlap_train.sum()),
            rows_exact_input_in_TTISS=int(overlap_val.sum()),
            duplicate_input_rows=len(x) - len(np.unique(keys))
        ))
        pd.DataFrame(dict(
            sample_id=np.arange(len(x)),
            exact_input_in_train=overlap_train,
            exact_input_in_TTISS=overlap_val
        )).to_csv(
            out / (name + '_overlap_flags.csv.gz'),
            index=False,
            compression='gzip'
        )
    pd.DataFrame(rows).to_csv(out / 'identity_summary.csv', index=False)
    json_write(
        out / 'limitations.json',
        dict(
            original_protocol_not_altered=True,
            exact_overlap_is_not_automatically_label_leakage=True,
            independence_requires_D_prior_and_source_provenance=True,
            seven_tests_repeatedly_seen=True
        )
    )
