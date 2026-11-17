from __future__ import annotations
import argparse, time, zipfile
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from pam2zero.activity import core
from pam2zero.activity.settings import DATASETS, REPO_ROOT, VARIANTS
from pam2zero.activity.settings import BASE_KEYS, BATCH_SIZE, EPOCHS, PARAMETERS, SEEDS, SHAPES, torch

def train_fold(variant, x, s, y, tr, va, seed, folder, epochs, batch):
    from pam2zero.activity.model import Optimizer, ShapeChain, arrays, predict, restore
    folder.mkdir(parents=True, exist_ok=True)
    model = ShapeChain(variant, seed, float(y[tr].mean())).to(x.device)
    assert sum((v.numel() for v in model.parameters())) == PARAMETERS[variant]
    opt = Optimizer(model)
    rng = np.random.RandomState(seed + 271)
    targets = torch.as_tensor(y, dtype=torch.float32, device=x.device)
    y32 = y.astype(np.float32)
    best_mse = np.inf
    best_mae = np.inf
    best_epoch = 0
    wait = 0
    best = None
    history = []
    for epoch in range(1, epochs + 1):
        start = time.time()
        order = tr[rng.permutation(len(tr))]
        total = 0.0
        model.train()
        for at in range(0, len(order), batch):
            ids = torch.as_tensor(order[at:at + batch], dtype=torch.long, device=x.device)
            model.zero_grad(set_to_none=True)
            pred = model(x.index_select(0, ids), s.index_select(0, ids))
            mse = ((pred - targets.index_select(0, ids)) ** 2).mean()
            loss = mse
            if not bool(torch.isfinite(loss)):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            opt.step()
            total += float(mse.detach().cpu()) * len(ids)
        vp = predict(model, x, s, va)
        err = vp.astype(np.float32) - y32[va]
        vmse = float(np.mean(err ** 2, dtype=np.float64))
        vmae = float(np.mean(np.abs(err), dtype=np.float64))
        if vmse < best_mse:
            best_mse = vmse
            best_epoch = epoch
            best = arrays(model)
            np.savez(folder / 'best_val_MSE.npz', **best)
        if vmae < best_mae - core.MIN_DELTA_MAE:
            best_mae = vmae
            wait = 0
        else:
            wait += 1
        history.append(dict(
            epoch=epoch,
            train_MSE=total / len(tr),
            val_MSE=vmse,
            val_MAE=vmae,
            best_MSE_epoch=best_epoch,
            mae_wait=wait,
            seconds=time.time() - start
        ))
        if epoch == 1 or epoch % 10 == 0 or wait >= core.PATIENCE_MAE:
            print(
                '%s E%03d MSE=%.6f val_MSE=%.6f val_MAE=%.6f best=%d wait=%d sec=%.2f' % (variant, epoch, total / len(tr), vmse, vmae, best_epoch, wait, history[-1]['seconds']),
                    
                    
                    
                flush=True
            )
        if wait >= core.PATIENCE_MAE:
            break
    restore(model, best)
    pd.DataFrame(history).to_csv(folder / 'training_history.csv', index=False)
    info = dict(
        parameters=PARAMETERS[variant],
        best_MSE_epoch=best_epoch,
        stopped_epoch=epoch,
        best_val_MSE=best_mse
    )
    core.dump(folder / 'selection.json', info)
    return (model, info)

def export(model, folder, scaler):
    from pam2zero.activity.model import arrays
    p = arrays(model)
    core.export_interpretation('path_pair', {k: p[k] for k in BASE_KEYS}, folder)
    if model.variant != 'baseline':
        pd.DataFrame(
            p['shape_weight'],
            index=np.arange(1, 24),
            columns=SHAPES).to_csv(folder / 'shape_single_weights.csv',
            index_label='position'
        )
        pd.DataFrame(dict(feature=SHAPES, interaction_weight=p['shape_pair'])).to_csv(
            folder / 'shape_pair_weights.csv',
            index=False
        )

def write_partial_summary(rows, out):
    frame = pd.DataFrame(rows)
    cols = ['Spearman', 'Pearson', 'MSE', 'MAE', 'RMSE', 'R2']
    summary = frame.groupby(['dataset', 'model'])[cols].agg(['mean', 'std'])
    summary.columns = ['_'.join(k) for k in summary.columns]
    summary.reset_index().to_csv(out / 'summary_mean_std.csv', index=False)
    frame.groupby(['dataset', 'model']).size().rename('completed_repeats').reset_index().to_csv(
        out / 'completed_repeats.csv',
        index=False
    )

def main():
    from pam2zero.activity.data import fit_scaler, load_shape, preflight_shape_files, standardize
    from pam2zero.activity.model import predict
    from pam2zero.activity.validation import self_test
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', default=REPO_ROOT)
    ap.add_argument('--datasets', nargs='+', default=DATASETS)
    ap.add_argument('--all-seven', action='store_true')
    ap.add_argument(
        '--with-baseline',
        action='store_true',
        help='Also train the original 147-parameter control on all selected datasets'
    )
    ap.add_argument('--variants', nargs='+', choices=list(PARAMETERS), default=VARIANTS)
    ap.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    ap.add_argument('--epochs', type=int, default=EPOCHS)
    ap.add_argument('--batch', type=int, default=BATCH_SIZE)
    ap.add_argument('--out')
    ap.add_argument('--self-test', action='store_true')
    args = ap.parse_args()
    if args.with_baseline:
        args.variants = ['baseline', 'shape_fixed']
    if args.epochs < 1 or args.batch < 1 or len(set(args.seeds)) != len(args.seeds) or (len(set(args.variants)) != len(args.variants)):
        ap.error('Positive epochs/batch and unique seeds/variants required')
    self_test()
    if args.self_test:
        return
    if torch is None:
        raise RuntimeError('Run in the existing CRISPR_GNN environment with PyTorch installed.')
    repo = core.find_repo(args.repo)
    datasets = core.OFFICIAL_SEVEN if args.all_seven else args.datasets
    if len(set(datasets)) != len(datasets):
        ap.error('Datasets must be unique')
    manifest, source_audit = core.verify_repo(repo, datasets, strict=True)
    preflight_shape_files(repo, datasets)
    print(
        'PLAN: datasets=',
        datasets,
        'variants=',
        args.variants,
        'repeats=',
        len(args.seeds),
        'folds=5 total_fits=',
        len(datasets) * len(args.variants) * len(args.seeds) * 5,
        flush=True
    )
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda':
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
    out = Path(args.out).resolve() if args.out else repo / 'DChain_DNAShape_Fixed_All7_results' / datetime.now().strftime('%Y%m%d_%H%M%S')
    if out.exists():
        raise FileExistsError('Use a new output directory: ' + str(out))
    out.mkdir(parents=True)
    protocol = dict(
        upstream_commit=manifest['commit'],
        source_split_hashes_verified=True,
        datasets=datasets,
        variants=args.variants,
        repeat_seeds=args.seeds,
        parameters=PARAMETERS,
        device=device,
        inputs='Original-order guide20+PAM3 and author DNA-shape23x5 ONLY; no Sc/misc',
        input_order='No reversal; shape positions match sequence positions',
        cohort='Author supplied split/order unchanged; no deduplication, sampling, target transform or off-target weights',
            
            
            
        KFold=dict(n_splits=5, shuffle=True, random_state=42),
        epochs=args.epochs,
        batch=args.batch,
        optimizer=dict(name='TFA-style AdamW', lr=core.LEARNING_RATE, wd_per_step=core.WEIGHT_DECAY, epsilon=core.ADAM_EPSILON),
            
            
            
        early_stop=dict(monitor='val_MAE', patience=core.PATIENCE_MAE, min_delta=core.MIN_DELTA_MAE),
            
            
            
        checkpoint='Strict minimum validation MSE; reload before prediction',
        primary_metric='Spearman on arithmetic mean of 5 fold test predictions; report 5-repeat sample std',
            
            
            
        preprocessing='Per-fold training rows only: position/channel mean+std; preserve author -1 padding values as numeric inputs',
            
            
            
        shape_fixed='Fixed standardized shape, position-specific linear weights AND 5 shared diagonal shape-interaction weights',
            
            
            
        baseline_objective='MSE',
        fixed_objective='MSE',
        shape_pair='Sum of chain-weighted same-channel products over guide20 position pairs; PAM shape is additive',
            
            
            
        limitation='Fixed DNA-shape inputs; model weights are trained. The 267-parameter model includes 5 shape-interaction weights absent from earlier shape262',
            
            
            
        script_sha256=core.sha256(Path(__file__).resolve()),
        implementation='Standalone file; helpers and manifest embedded'
    )
    core.dump(out / 'protocol_LOCKED.json', protocol)
    protocol_hash = core.sha256(out / 'protocol_LOCKED.json')
    pd.DataFrame(source_audit).to_csv(out / 'upstream_file_audit.csv', index=False)
    repeats = []
    fold_rows = []
    cohorts = []
    feature_audit = []
    print('OUTPUT:', out, 'DEVICE:', device, flush=True)
    for ds in datasets:
        x, y, seq, xt, yt, seqt = core.load_dataset(repo, ds)
        dsdir = out / ds
        dsdir.mkdir()
        raw, files1 = load_shape(repo, ds, 'train', seq, dsdir)
        rawt, files2 = load_shape(repo, ds, 'test', seqt, dsdir)
        files = files1 + files2
        hashes = {str(p): core.sha256(p) for p in files}
        feature_audit.extend((dict(
            dataset=ds,
            file=str(p.relative_to(repo)),
            sha256=hashes[str(p)]
        ) for p in files))
        pd.DataFrame(feature_audit).to_csv(out / 'shape_file_hashes.csv', index=False)
        cohorts.append(dict(
            dataset=ds,
            train_N=len(y),
            test_N=len(yt),
            train_unique=len(np.unique(seq)),
            test_unique=len(np.unique(seqt)),
            exact_sequence_overlap=len(np.intersect1d(seq, seqt)),
            cohort_changed=False
        ))
        pd.DataFrame(cohorts).to_csv(out / 'data_audit.csv', index=False)
        folds = list(KFold(n_splits=5, shuffle=True, random_state=42).split(x))
        xx = torch.as_tensor(x, dtype=torch.long, device=device)
        xxt = torch.as_tensor(xt, dtype=torch.long, device=device)
        for repeat, seed in enumerate(args.seeds):
            predictions = {k: [] for k in args.variants}
            contributions = {k: {} for k in args.variants}
            for fold, (tr, va) in enumerate(folds):
                shared = dsdir / ('repeat_%d' % repeat) / ('fold_%d_preprocessing' % fold)
                shared.mkdir(parents=True)
                sc = fit_scaler(raw, tr)
                np.savez(shared / 'scaler.npz', **sc)
                np.savez_compressed(shared / 'indices.npz', train=tr, validation=va)
                ss = torch.as_tensor(standardize(raw, sc), device=device)
                sst = torch.as_tensor(standardize(rawt, sc), device=device)
                for variant in args.variants:
                    folder = dsdir / variant / ('repeat_%d' % repeat) / ('fold_%d' % fold)
                    print(
                        '\n%s %s repeat=%d seed=%d fold=%d' % (ds, variant, repeat, seed, fold),
                        flush=True
                    )
                    model, info = train_fold(
                        variant,
                        xx,
                        ss,
                        y,
                        tr,
                        va,
                        seed + 1009 * fold,
                        folder,
                        args.epochs,
                        args.batch
                    )
                    pred, parts = predict(model, xxt, sst, parts=True)
                    predictions[variant].append(pred)
                    for k, v in parts.items():
                        contributions[variant].setdefault(k, []).append(v)
                    export(model, folder, sc)
                    fold_rows.append(dict(
                        dataset=ds,
                        model=variant,
                        repeat=repeat,
                        seed=seed,
                        fold=fold,
                        **info,
                        **core.regression_metrics(yt, pred)
                    ))
                    pd.DataFrame(fold_rows).to_csv(out / 'fold_diagnostic_metrics.csv', index=False)
                    del model
                del ss, sst
            for variant in args.variants:
                pred = np.mean(np.stack(predictions[variant]), 0)
                parts = {k: np.mean(np.stack(v), 0) for k, v in contributions[variant].items()}
                assert np.allclose(pred, sum(parts.values()), atol=2e-05, rtol=1e-05)
                folder = dsdir / variant / ('repeat_%d' % repeat)
                pd.DataFrame(dict(
                    sequence=seqt,
                    y_true=yt,
                    y_pred=pred,
                    **{'contribution_' + k: v for k, v in parts.items()}
                )).to_csv(
                    folder / 'ensemble_test_predictions.csv',
                    index=False
                )
                row = dict(
                    dataset=ds,
                    model=variant,
                    repeat=repeat,
                    seed=seed,
                    parameters=PARAMETERS[variant],
                    **core.regression_metrics(yt, pred)
                )
                repeats.append(row)
                print('FIVE-FOLD ENSEMBLE', row, flush=True)
                pd.DataFrame(repeats).to_csv(out / 'repeat_ensemble_metrics.csv', index=False)
                write_partial_summary(repeats, out)
        for p in files:
            if core.sha256(p) != hashes[str(p)]:
                raise RuntimeError('Shape input changed: ' + str(p))
        del xx, xxt, raw, rawt
    frame = pd.DataFrame(repeats)
    cols = ['Spearman', 'Pearson', 'MSE', 'MAE', 'RMSE', 'R2']
    summary = frame.groupby(['dataset', 'model'])[cols].agg(['mean', 'std'])
    summary.columns = ['_'.join(k) for k in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(out / 'summary_mean_std.csv', index=False)
    paired = []
    for ds in datasets:
        for a, b in [('shape_fixed', 'baseline')]:
            if a not in args.variants or b not in args.variants:
                continue
            aa = frame[(frame.dataset == ds) & (frame.model == a)].set_index('seed')
            bb = frame[(frame.dataset == ds) & (frame.model == b)].set_index('seed')
            delta = aa.Spearman - bb.Spearman
            paired.append(dict(
                dataset=ds,
                comparison=a + ' minus ' + b,
                n_repeats=len(delta),
                delta_Spearman_mean=float(delta.mean()),
                delta_Spearman_std=float(delta.std(ddof=1)),
                positive_repeats=int((delta > 0).sum())
            ))
    pd.DataFrame(paired).to_csv(out / 'paired_Spearman_deltas.csv', index=False)
    _, after = core.verify_repo(repo, datasets, strict=True)
    assert [r['actual'] for r in source_audit] == [r['actual'] for r in after]
    assert core.sha256(out / 'protocol_LOCKED.json') == protocol_hash
    core.dump(
        out / 'readonly_verified.json',
        dict(
            author_source_and_data_unchanged=True,
            shape_files_unchanged=True,
            protocol_sha256=protocol_hash
        )
    )
    with zipfile.ZipFile(
        out / 'DChain_DNAShape_Fixed_All7_results.zip',
        'w',
        zipfile.ZIP_DEFLATED
    ) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.suffix in ['.csv', '.json']:
                z.write(p, str(p.relative_to(out)))
    print('\n', summary.to_string(index=False))
    print('\nOUTPUT:', out)
    print('Send DChain_DNAShape_Fixed_All7_results.zip; fold weights/scalers stay local.')
