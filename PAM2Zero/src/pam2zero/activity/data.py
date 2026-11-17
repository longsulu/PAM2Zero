from __future__ import annotations
import ast, os
import numpy as np
import pandas as pd
from pam2zero.activity.settings import SHAPES

def load_shape(repo, dataset, split, expected_sequences, log_folder):
    paths = [repo / 'bedfeatures' / dataset / split / ('output_%s.fasta' % f) for f in SHAPES]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError('Missing author DNA-shape files:\n' + '\n'.join(missing))
    tree = ast.parse((repo / 'load_data_func.py').read_text(encoding='utf-8-sig'))
    names = ['load_shape_features_inorder', 'unify_DNA_shape_columns']
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    if len(nodes) != 2:
        raise RuntimeError('Upstream DNA-shape functions changed.')
    scope = {'np': np, 'pd': pd, 'os': os}
    module = ast.Module(body=nodes, type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, str(repo / 'load_data_func.py'), 'exec'), scope)
    frame = pd.DataFrame({'sgRNA': expected_sequences})
    cwd = os.getcwd()
    try:
        os.chdir(str(repo))
        for feature in SHAPES:
            frame = scope['load_shape_features_inorder'](
                frame,
                dataset,
                split,
                feature,
                lines_per_sgRNA=1
            )
        frame = scope['unify_DNA_shape_columns'](frame, SHAPES)
    finally:
        os.chdir(cwd)
    if not np.array_equal(frame.sgRNA.to_numpy(), expected_sequences):
        raise RuntimeError('Shape loading changed sample order/count.')
    shape = np.stack(frame.DNA_shape.to_numpy()).astype(np.float32)
    if shape.shape != (len(expected_sequences), 23, 5) or not np.isfinite(shape).all():
        raise ValueError('Invalid shape tensor: ' + str(shape.shape))
    audit = [
        dict(feature=f, minus_one_values=int((shape[:, :, i] == -1).sum()), constant_positions=int((shape[:, :, i].std(0) < 1e-08).sum())) for i,
            
            
            
        f in enumerate(SHAPES)
    ]
    pd.DataFrame(audit).to_csv(log_folder / ('%s_shape_values.csv' % split), index=False)
    return (shape, paths)

def fit_scaler(shape, indices):
    values = shape[indices].astype(np.float64)
    mean = values.mean(0)
    std = values.std(0)
    constant = std < 1e-08
    std[constant] = 1.0
    return dict(mean=mean, std=std, constant=constant)

def standardize(shape, scaler):
    return ((shape - scaler['mean']) / scaler['std']).astype(np.float32)

def preflight_shape_files(repo, datasets):
    missing = []
    for dataset in datasets:
        for split in ['train', 'test']:
            for feature in SHAPES:
                path = repo / 'bedfeatures' / dataset / split / ('output_%s.fasta' % feature)
                if not path.is_file():
                    missing.append(str(path))
    if missing:
        raise FileNotFoundError('Missing DNA-shape inputs before training:\n' + '\n'.join(missing))
