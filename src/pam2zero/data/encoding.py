from __future__ import annotations
from pam2zero import config as cfg
import numpy as np
import pandas as pd
import re


def check_codes(name, codes):
    codes = np.asarray(codes)
    if codes.ndim != 2 or codes.shape[1] != 24:
        raise ValueError(f'{name}: expected (N,24), got {codes.shape}')
    lo, hi = (int(codes.min()), int(codes.max()))
    if lo < 1 or hi > 36:
        raise ValueError(f'{name}: codes must be 1..36, got {lo}..{hi}')
    print(f'{name}: shape={codes.shape}, range={lo}..{hi}')


def _off_target_mask(codes):
    codes = np.asarray(codes)
    if codes.ndim != 2 or codes.shape[1] != 24:
        raise ValueError(f'期望 (N,24) pair codes, got {codes.shape}')
    matches = np.array([1, 7, 13, 19, 25, 31], dtype=codes.dtype)
    return np.any(~np.isin(codes[:, :-3], matches), axis=1)


def _sg_str(v):
    if isinstance(v, bytes):
        v = v.decode('utf-8')
    return str(v).strip().upper()


def rule_B(sg):
    sg = _sg_str(sg)
    return sg[:-3] if len(sg) > 20 else sg


def basepair_2_id_encode_original(sg, off):
    sg, off = (_sg_str(sg), _sg_str(off))
    encoded = []
    for s, o in zip(sg, off):
        if s == 'N':
            s = o
        elif o == 'N':
            o = s
        pair = s + o
        if pair not in cfg.ENCODE_DICT_36:
            raise KeyError(f'Unsupported pair={pair!r}, sg={sg!r}, DNA={off!r}')
        encoded.append(cfg.ENCODE_DICT_36[pair])
    if len(encoded) > 24:
        raise ValueError(f'编码长度不能超过 24，got {len(encoded)}')
    if len(encoded) < 24:
        encoded = [31] * (24 - len(encoded)) + encoded
    return encoded


def normalize_to23(codes, name, record=True, policy=None):
    a = np.asarray(codes)
    if a.ndim != 2 or a.shape[1] != 24:
        raise ValueError('%s: expected original (N,24), got %s' % (name, a.shape))
    # Remove only the left pad; PAM index 20 is ignored by the model.
    region = a[:, 1:24]
    left_pad = np.isin(a[:, 0], [25, 31])
    valid_types = np.isin(region, np.array(cfg.MATCH_CODES + cfg.MISMATCH_CODES)).all(1)
    keep = left_pad & valid_types
    row = dict(
        dataset=name,
        input_rows=len(a),
        kept_rows=int(keep.sum()),
        dropped_rows=int((~keep).sum()),
        missing_left_pad=int((~left_pad).sum()),
        non_standard_pair_in_23_positions=int((~valid_types).sum())
    )
    mode = cfg.UNSUPPORTED_ALIGNMENT_POLICY if policy is None else policy
    if mode not in ('error', 'filter'):
        raise ValueError('invalid alignment policy')
    if record:
        cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)
        cfg.ALIGNMENT_AUDIT[:] = [r for r in cfg.ALIGNMENT_AUDIT if r['dataset'] != name] + [row]
        pd.DataFrame(cfg.ALIGNMENT_AUDIT).to_csv(cfg.OUT_DIR / 'alignment_audit.csv', index=False)
        print('[23-position alignment: guide20 + PAM3]', row)
    if not keep.all():
        message = '%s: %d rows contain unsupported padding/gap in guide or PAM' % (
            name,
            int((~keep).sum())
        )
        if mode == 'error':
            raise ValueError(message)
        if record:
            print(
                'FILTER:',
                message,
                '; changed evaluation population, not comparable with full-set reference.'
            )
    if not keep.any():
        raise ValueError('%s: no valid 23-position samples' % name)
    return (region[keep].astype(np.uint8), keep)


def clean_sequence(v):
    if pd.isna(v):
        return ''
    return re.sub('\\s+', '', str(v)).upper().replace('U', 'T')


def column(df, names):
    normalize = lambda s: re.sub('[^a-z0-9]', '', str(s).lower())
    lookup = {normalize(c): c for c in df.columns}
    for n in names:
        if normalize(n) in lookup:
            return lookup[normalize(n)]
    raise ValueError('找不到列 %s；实际列=%s。请将序列/标签列改名为 gRNA,OT,label。不能把reads自动当作二分类标签。' % (
        names,
        list(df.columns)
    ))


def identify_columns(df):
    aliases = [
        (cfg.GUIDE_COLUMN, ['gRNA', 'sgRNA', 'sg', 'guide', 'guide_seq', 'guide_sequence', 'sgRNA_sequence', 'gRNA_seq', 'gRNA sequence', 'Align.sgRNA', 'Wildtype Sequence']),
            
            
            
        (cfg.DNA_COLUMN, ['OT', 'DNA', 'off_target', 'target', 'target_sequence', 'off_target_sequence', 'DNA sequence', 'OT sequence', 'OT_seq', 'Align.off-target', 'OT sequence with bulge info']),
            
            
            
        (
            cfg.LABEL_COLUMN,
            ['label', 'labels', 'is_offtarget', 'binary_label', 'active', 'TrueOT', 'TrueOT label']
        )
    ]
    result = []
    for override, names in aliases:
        result.append(column(df, [override]) if override else column(df, names))
    if len(set(result)) != 3:
        raise ValueError('guide、DNA、label必须是三个不同的列')
    return tuple(result)


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
