from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np
import pandas as pd
import hashlib
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
EMBEDDED_MANIFEST = json.loads((Path(__file__).resolve().parents[1] / 'resources/crisprfusion_manifest.json').read_text())
_base_LEARNING_RATE = 0.0003
_base_WEIGHT_DECAY = 0.0001
_base_ADAM_EPSILON = 1e-07
_base_PATIENCE_MAE = 25
_base_MIN_DELTA_MAE = 0.0001
_base_EDGE_LOG_RANGE = 0.1
_base_OFFICIAL_SEVEN = ['xCas', 'eSp', 'HF1', 'SpCas9', 'CRISPRON', 'HT_Cas9', 'HypaCas9']
_base_BASES = 'ACGT'

def _base_dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False),
        encoding='utf-8'
    )

def _base_sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def _base_git_blob_sha(path):
    h = hashlib.sha1()
    h.update(('blob %d\x00' % Path(path).stat().st_size).encode())
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def _base_find_repo(value=None):
    p = Path(value).expanduser() if value else Path(__file__).resolve().parent
    candidates = [p, p / 'CrisprFusion-main', p / 'CrisprFusion']
    for q in candidates:
        if (q / 'CrisprFusion.py').is_file() and (q / 'data').is_dir():
            return q.resolve()
    raise FileNotFoundError('Extract the downloaded ZIP; place the scripts next to CrisprFusion.py, or set REPO_ROOT / --repo.')

def _base_verify_repo(repo, datasets, strict=True):
    manifest = EMBEDDED_MANIFEST
    rows = []
    rows = []
    for name, want in manifest['code_sha256'].items():
        p = repo / name
        actual = _base_sha256(p) if p.is_file() else None
        rows.append(dict(
            file=name,
            kind='code_sha256',
            expected=want,
            actual=actual,
            match=actual == want
        ))
    for ds in datasets:
        for split in ['train_X', 'train_y', 'test_X', 'test_y']:
            name = 'data/%s_%s.npy' % (ds, split)
            p = repo / name
            if not p.is_file():
                raise FileNotFoundError('Missing supplied split file: ' + str(p))
            want = manifest['data_git_blobs'].get(name)
            if want is None:
                raise ValueError('Dataset not in checked repository manifest: ' + name)
            actual = _base_git_blob_sha(p)
            rows.append(dict(
                file=name,
                kind='git_blob_sha1',
                expected=want['git_blob_sha1'],
                actual=actual,
                match=actual == want['git_blob_sha1']
            ))
    if strict and (not all((r['match'] for r in rows))):
        bad = [r['file'] for r in rows if not r['match']]
        raise RuntimeError('Downloaded files differ from the verified upstream snapshot: ' + ', '.join(bad) + '. Do not claim exact source alignment. --allow-unverified explicitly records an unverified version.')
    return (manifest, rows)

def _base_encode_sequences(raw):
    arr = np.asarray(raw)
    if arr.ndim == 2 and arr.shape[1] == 1:
        arr = arr[:, 0]
    if arr.ndim != 1:
        raise ValueError('Expected one sequence string per row, got ' + str(arr.shape))
    strings = np.array([(x.decode('utf-8') if isinstance(
        x,
        bytes) else str(x
    )).upper() for x in arr])
    if any((len(x) != 23 or any((c not in _base_BASES for c in x)) for x in strings)):
        raise ValueError('Checked splits must contain exactly 23 A/C/G/T bases; no silent truncation/padding/filtering.')
    lut = {c: i for i, c in enumerate(_base_BASES)}
    x = np.array([[lut[c] for c in s] for s in strings], dtype=np.int64)
    return (x, strings)

def _base_load_dataset(repo, ds):
    data = []
    for split in ['train', 'test']:
        x, seq = _base_encode_sequences(np.load(
            repo / 'data' / ('%s_%s_X.npy' % (ds, split)),
            allow_pickle=False
        ))
        y = np.asarray(
            np.load(repo / 'data' / ('%s_%s_y.npy' % (ds, split)), allow_pickle=False),
            dtype=np.float64).reshape(-1
        )
        if len(y) != len(x) or not np.isfinite(y).all():
            raise ValueError(ds + ' ' + split + ': invalid label alignment')
        data.extend([x, y, seq])
    x, y, s, xt, yt, st = data
    return (x, y, s, xt, yt, st)

def _base_regression_metrics(y, pred):
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    p = np.asarray(pred, dtype=np.float64).reshape(-1)
    if y.shape != p.shape or not np.isfinite(p).all():
        raise ValueError('Nonfinite or misaligned predictions')
    variable = len(y) > 1 and np.ptp(y) > 0 and (np.ptp(p) > 0)
    return dict(
        N=len(y),
        MSE=float(mean_squared_error(y, p)),
        MAE=float(mean_absolute_error(y, p)),
        RMSE=float(np.sqrt(mean_squared_error(y, p))),
        Spearman=float(spearmanr(y, p)[0]) if variable else np.nan,
        Pearson=float(pearsonr(y, p)[0]) if variable else np.nan,
        R2=float(r2_score(y, p)) if len(y) > 1 else np.nan
    )

def _base_make_parameters(kind, seed, bias=0.0, dtype=np.float32):
    rng = np.random.RandomState(seed)
    p = {
        'guide': rng.normal(0, 0.01, (20, 4)).astype(dtype),
        'pam': rng.normal(0, 0.01, (3, 4)).astype(dtype),
        'bias': np.array([bias], dtype=dtype)
    }
    if kind != 'additive':
        p['interaction'] = rng.normal(0, 0.01, (4, 4)).astype(dtype)
        p['edge'] = np.zeros(19, dtype=dtype)
        if kind == 'path_pair':
            p['distance'] = np.full(19, math.log(0.1 / 0.9), dtype=dtype)
    return p

def _base_pair_indices(kind):
    if kind == 'adjacent_pair':
        return (np.arange(19), np.arange(1, 20))
    return np.triu_indices(20, 1)

class _base_NumpyChain:

    def __init__(self, kind, seed, bias=0.0, dtype=np.float32):
        if kind not in ['additive', 'adjacent_pair', 'path_pair']:
            raise ValueError(kind)
        self.kind = kind
        self.p = _base_make_parameters(kind, seed, bias, dtype)
        if kind != 'additive':
            self.i, self.j = _base_pair_indices(kind)
            self.d = self.j - self.i - 1
            self.mask = ((np.arange(19)[
                None,
                :] >= self.i[:,
                None]) & (np.arange(19)[None,
                :] < self.j[:,
                None
            ])).astype(dtype)

    def predict(self, x):
        return self.forward(x)[0]

    def forward(self, x):
        g = self.p['guide']
        pa = self.p['pam']
        g = g - g.mean(1, keepdims=True)
        pa = pa - pa.mean(1, keepdims=True)
        pred = g[
            np.arange(20)[None, :],
            x[:, :20]].sum(1) + pa[np.arange(3)[None, :],
            x[:, 20:]].sum(1) + self.p['bias'][0
        ]
        if self.kind == 'additive':
            return (pred, None)
        raw = self.p['interaction']
        J = raw - raw.mean(0, keepdims=True) - raw.mean(1, keepdims=True) + raw.mean()
        loge = _base_EDGE_LOG_RANGE * np.tanh(self.p['edge'])
        coeff = np.exp(self.mask @ loge)
        if self.kind == 'path_pair':
            b = 1 / (1 + np.exp(-self.p['distance']))
            coeff = coeff * b[self.d]
        else:
            b = None
        types = J[x[:, self.i], x[:, self.j]]
        pred = pred + (types * coeff[None, :]).sum(1)
        return (pred, (J, coeff, types, b))

    def gradients(self, x, y):
        pred, cache = self.forward(x)
        err = pred - y
        dy = 2 * err / len(x)
        grad = {k: np.zeros_like(v) for k, v in self.p.items()}
        for pos in range(20):
            np.add.at(grad['guide'][pos], x[:, pos], dy)
        for pos in range(3):
            np.add.at(grad['pam'][pos], x[:, 20 + pos], dy)
        grad['guide'] -= grad['guide'].mean(1, keepdims=True)
        grad['pam'] -= grad['pam'].mean(1, keepdims=True)
        grad['bias'][0] = dy.sum()
        if cache is not None:
            J, coeff, types, b = cache
            a = x[:, self.i]
            c = x[:, self.j]
            w = dy[:, None] * coeff[None, :]
            gj = np.bincount(
                (a * 4 + c).reshape(-1),
                weights=w.reshape(-1),
                minlength=16).reshape(4,
                4
            )
            gj = gj - gj.mean(0, keepdims=True) - gj.mean(1, keepdims=True) + gj.mean()
            grad['interaction'][...] = gj
            dcoeff = (dy[:, None] * types).sum(0)
            grad['edge'][...] = self.mask.T @ (dcoeff * coeff) * _base_EDGE_LOG_RANGE * (1 - np.tanh(self.p['edge']) ** 2)
            if b is not None:
                grad['distance'][...] = np.bincount(
                    self.d,
                    weights=dcoeff * coeff * (1 - b[self.d]),
                    minlength=19
                )
        return (float(np.mean(err ** 2)), grad)

    def arrays(self):
        return {k: v.copy() for k, v in self.p.items()}

    def set_arrays(self, p):
        for k in self.p:
            self.p[k][...] = p[k]

    def param_count(self):
        return sum((v.size for v in self.p.values()))

def _base_export_interpretation(kind, p, folder):
    g = p['guide'] - p['guide'].mean(1, keepdims=True)
    pa = p['pam'] - p['pam'].mean(1, keepdims=True)
    pd.DataFrame(
        g,
        index=np.arange(1, 21),
        columns=list(_base_BASES)).to_csv(folder / 'guide_single_effects.csv',
        index_label='position'
    )
    pd.DataFrame(
        pa,
        index=np.arange(21, 24),
        columns=list(_base_BASES)).to_csv(folder / 'PAM_effects.csv',
        index_label='position'
    )
    if kind != 'additive':
        J = p['interaction']
        J = J - J.mean(0, keepdims=True) - J.mean(1, keepdims=True) + J.mean()
        pd.DataFrame(
            J,
            index=list(_base_BASES),
            columns=list(_base_BASES)).to_csv(folder / 'type_interactions.csv',
            index_label='left_base'
        )
        pd.DataFrame(dict(
            edge=np.arange(1, 20),
            gain=np.exp(_base_EDGE_LOG_RANGE * np.tanh(p['edge']))
        )).to_csv(
            folder / 'edge_gains.csv',
            index=False
        )
        if kind == 'path_pair':
            pd.DataFrame(dict(
                distance=np.arange(1, 20),
                weight=1 / (1 + np.exp(-p['distance']))
            )).to_csv(
                folder / 'distance_weights.csv',
                index=False
            )
try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None
dump = _base_dump
sha256 = _base_sha256
git_blob_sha = _base_git_blob_sha
find_repo = _base_find_repo
verify_repo = _base_verify_repo
encode_sequences = _base_encode_sequences
load_dataset = _base_load_dataset
regression_metrics = _base_regression_metrics
make_parameters = _base_make_parameters
pair_indices = _base_pair_indices
NumpyChain = _base_NumpyChain
export_interpretation = _base_export_interpretation
LEARNING_RATE = _base_LEARNING_RATE
WEIGHT_DECAY = _base_WEIGHT_DECAY
ADAM_EPSILON = _base_ADAM_EPSILON
MIN_DELTA_MAE = _base_MIN_DELTA_MAE
PATIENCE_MAE = _base_PATIENCE_MAE
EDGE_LOG_RANGE = _base_EDGE_LOG_RANGE
OFFICIAL_SEVEN = _base_OFFICIAL_SEVEN
BASES = _base_BASES
