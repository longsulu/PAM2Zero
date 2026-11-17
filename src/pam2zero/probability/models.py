from __future__ import annotations
import json
import numpy as np
import pandas as pd
from scipy.special import expit, logsumexp
from scipy.linalg import expm
from scipy.optimize import minimize
from sklearn import metrics
from pam2zero import config as cfg
from pam2zero.utils.json import json_write

def pm_seed_count(d):
    return np.array(
        [sum((a != b for a, b in zip(g[12:20], s[12:20]))) for g, s in zip(d.guide23, d.DNA)],
        dtype=int
    )

def pm_binary_matrix(d):
    return np.column_stack([d['GUIDE-seq_detected_%d' % r].astype(str).str.lower().map({
        'true': 1,
        'false': 0,
        '1': 1,
        '0': 0
    }).to_numpy(dtype=float) for r in [
        1,
        2,
        3
    ]])

def pm_source_consistency(d, label, out):
    n = np.array([sum((a != b for a, b in zip(g[:20], s[:20]))) for g, s in zip(d.guide23, d.DNA)])
    bad = n != d.mismatches.to_numpy()
    d.loc[bad].to_csv(out / (label + '_sequence_annotation_conflicts.csv'), index=False)
    return d.loc[~bad].copy()

def pm_make_pools(prepared, on):
    pools = {}
    for guide, d in prepared.groupby('guide'):
        t = on[on.guide == guide]
        if not len(t):
            raise RuntimeError('guide缺少on-target：' + guide)
        pools[guide] = dict(
            z=np.r_[d.score.to_numpy(), t.score.to_numpy()],
            h=np.r_[pm_seed_count(d), np.zeros(len(t), dtype=int)],
            keys=d.key.to_numpy(),
            n_off=len(d),
            n_on=len(t)
        )
    return pools

def pm_prepare_pool(pool, a):
    logw = a * pool['z']
    logD = logsumexp(np.r_[logw, np.log(cfg.ENCOUNTER_SCALE), np.log(cfg.BACKGROUND_WEIGHT)])
    assoc = np.exp(logw - logD)
    background = float(np.exp(np.log(cfg.BACKGROUND_WEIGHT) - logD))
    return (logw, assoc, background)

def pm_ctmc_activity(pool, a, b, logrelease, diagnostics=False):
    _, assoc, bg = pm_prepare_pool(pool, a)
    on = np.r_[np.bincount(pool['h'], weights=assoc, minlength=9), bg]
    cut = np.r_[np.exp(-b * np.arange(9) / 8.0), 0.0]
    release = float(np.exp(logrelease))
    n = 10
    A = np.zeros((1 + 3 * n, 1 + 3 * n))
    for j in range(n):
        c = 1 + j
        bound = 1 + n + j
        counter = 1 + 2 * n + j
        A[c, 0] += on[j]
        A[0, 0] -= on[j]
        A[0, c] += 1.0
        A[c, c] -= 1.0 + cut[j]
        A[bound, c] += cut[j]
        A[bound, bound] -= release
        A[0, bound] += release
        A[counter, c] += cut[j]
    p0 = np.zeros(len(A))
    p0[0] = 1.0
    p = expm(A).dot(p0)
    mass = float(p[:1 + 2 * n].sum())
    if abs(mass - 1) > 1e-08 or p.min() < -1e-09:
        raise RuntimeError('CTMC positivity/resource conservation failure')
    accumulated = np.maximum(p[1 + 2 * n:1 + 3 * n], 0)
    factor = np.divide(accumulated[:9], on[:9], out=np.zeros(9), where=on[:9] > 0)
    activity = np.maximum(assoc * factor[pool['h']], 1e-300)
    if diagnostics:
        return (
            activity,
            dict(
                resource_mass=mass,
                free=float(p[0]),
                bound=float(p[1:1 + n].sum()),
                product_bound=float(p[1 + n:1 + 2 * n].sum()),
                cuts=float(accumulated.sum()),
                on_target_cuts=float(activity[pool['n_off']:].sum())
            )
        )
    return activity

def pm_log_activity(kind, pool, theta):
    a = float(theta[0])
    if kind == 'SharedCas9_CTMC':
        return np.log(pm_ctmc_activity(pool, a, theta[1], theta[2]))
    if kind == 'Independent_Kinetic':
        cut = np.exp(-theta[1] * pool['h'] / 8.0)
        r = 1.0 + cut
        factor = cut / r * (1.0 + np.expm1(-r) / r)
        return a * pool['z'] - np.log(cfg.ENCOUNTER_SCALE) + np.log(factor)
    raise ValueError(kind)

def pm_poisson_probability(logmu):
    mu = np.exp(np.clip(logmu, -700, 40))
    return np.clip(-np.expm1(-mu), 1e-12, 1 - 1e-12)

def pm_probabilities(kind, pool, theta):
    z = pool['z']
    a = theta[0]
    if kind == 'Calibrated_SeedMM':
        rank = a * z - theta[1] * pool['h'] / 8.0
        q = expit(rank[:, None] + theta[2:5][None, :])
        return (q, q.mean(1), rank)
    if kind == 'Calibrated_Bernoulli':
        q = expit(a * z[:, None] + theta[1:4][None, :])
        latent = q.mean(1)
        return (q, latent, a * z)
    if kind == 'Occupancy_Detection':
        psi = expit(a * z + theta[1])
        d = expit(theta[2:5])
        return (psi[:, None] * d[None, :], psi, a * z)
    if kind == 'FiniteBudget':
        logw = a * z
        logD = logsumexp(np.r_[logw, np.log(cfg.BACKGROUND_WEIGHT)])
        pi = np.exp(logw - logD)
        d = expit(theta[1:4])
        q = -np.expm1(cfg.FIXED_ATTEMPTS * np.log1p(-np.minimum(
            pi[:, None] * d[None, :],
            1 - 1e-15
        )))
        return (np.clip(q, 1e-12, 1 - 1e-12), q.mean(1), a * z)
    la = pm_log_activity(kind, pool, theta)
    logs = theta[3:6] if kind == 'SharedCas9_CTMC' else theta[2:5]
    q = pm_poisson_probability(la[:, None] + logs[None, :])
    return (q, q.mean(1), la + logsumexp(logs) - np.log(3.0))

def pm_history_nll(kind, pool, theta, idx, y):
    if kind == 'Occupancy_Detection':
        logits = theta[0] * pool['z'][idx] + theta[1]
        logpsi = -np.logaddexp(0.0, -logits)
        lognotpsi = -np.logaddexp(0.0, logits)
        ld = -np.logaddexp(0.0, -theta[2:5])
        lnd = -np.logaddexp(0.0, theta[2:5])
        loghistory = np.sum(y * ld + (1 - y) * lnd, axis=1)
        ll = np.where(
            y.sum(1) > 0,
            logpsi + loghistory,
            np.logaddexp(lognotpsi, logpsi + loghistory)
        )
        return float(-ll.sum())
    q, _, _ = pm_probabilities(kind, pool, theta)
    q = np.clip(q[idx], 1e-12, 1 - 1e-12)
    return float(-(y * np.log(q) + (1 - y) * np.log1p(-q)).sum())

def pm_bounds_and_start(kind):
    slope = (0.05, 2.0)
    if kind == 'Calibrated_SeedMM':
        return ([slope, (0.0, 12.0)] + [(-15.0, 5.0)] * 3, np.array([0.5, 1.0, -4.0, -4.0, -4.0]))
    if kind == 'Calibrated_Bernoulli':
        return ([slope] + [(-15.0, 5.0)] * 3, np.array([0.5, -4.0, -4.0, -4.0]))
    if kind == 'Occupancy_Detection':
        return ([slope, (-15.0, 8.0)] + [(-8.0, 8.0)] * 3, np.array([0.5, -3.0, 0.0, 0.0, 0.0]))
    if kind == 'FiniteBudget':
        return ([slope] + [(-15.0, 10.0)] * 3, np.array([0.5, -4.0, -4.0, -4.0]))
    if kind == 'Independent_Kinetic':
        return ([slope, (0.0, 12.0)] + [(-12.0, 25.0)] * 3, np.array([0.5, 1.0, 5.0, 5.0, 5.0]))
    return (
        [slope, (0.0, 12.0), (-4.0, 4.0)] + [(-12.0, 25.0)] * 3,
        np.array([0.5, 1.0, 0.0, 5.0, 5.0, 5.0])
    )

def pm_fit_model(kind, training, pools, out, held):
    cache = out / ('fit_%s_%s.json' % (held, kind))
    if cfg.RESUME and cache.is_file():
        saved = json.loads(cache.read_text(encoding='utf-8'))
        return (np.asarray(saved['theta'], dtype=float), saved)
    bounds, start = pm_bounds_and_start(kind)
    nobs = sum((len(v[0]) for v in training.values())) * 3

    def objective(theta):
        val = sum((
            pm_history_nll(kind, pools[g], theta, idx, y) for g,
            (idx, y) in training.items()
        )) / nobs
        val += 1e-05 * float(np.mean((theta - start) ** 2))
        return val
    results = []
    for trial in range(cfg.N_STARTS):
        x = start.copy()
        if trial:
            x[0] = 1.0
            x[1] = min(x[1] + 1.0, bounds[1][1])
        r = minimize(
            objective,
            x,
            method='L-BFGS-B',
            bounds=bounds,
            options=dict(maxiter=cfg.MAX_ITER, ftol=1e-10, maxls=30)
        )
        if np.isfinite(r.fun):
            results.append(r)
    if not results:
        raise RuntimeError('No finite optimization result: ' + kind)
    r = min(results, key=lambda r: r.fun)
    bound_hits = [
        i for i,
        (v, (lo, hi)) in enumerate(zip(r.x, bounds)) if min(v - lo, hi - v) < 0.0001
    ]
    spec = dict(
        model=kind,
        held_guide=held,
        train_guides=sorted(training),
        theta=r.x.tolist(),
        bounds=bounds,
        objective=float(r.fun),
        success=bool(r.success),
        message=str(r.message),
        iterations=int(r.nit),
        boundary_parameters=bound_hits,
        parameter_count=len(r.x),
        identifiable_physical_rates=False,
        fit_uses_GUIDE_histories_only=True
    )
    json_write(out / ('fit_%s_%s.json' % (held, kind)), spec)
    return (r.x, spec)

def pm_macro_metrics(y, score, g):
    rows = []
    for guide in np.unique(g):
        t = g == guide
        yy = y[t]
        ss = score[t]
        if len(np.unique(yy)) < 2:
            continue
        p, r, _ = metrics.precision_recall_curve(yy, ss)
        p[r == 0] = 1
        rows.append(dict(
            guide=guide,
            N=int(t.sum()),
            Positive=int(yy.sum()),
            AUPRC=float(metrics.auc(r, p)),
            AP=float(metrics.average_precision_score(yy, ss)),
            AUROC=float(metrics.roc_auc_score(yy, ss))
        ))
    d = pd.DataFrame(rows)
    summary = dict(N=len(y), Positive=int(y.sum()), Valid_guides=len(d))
    for c in ['AUPRC', 'AP', 'AUROC']:
        summary['Macro_' + c] = float(d[c].mean()) if len(d) else float('nan')
    return (summary, d)

def pm_evaluate_models(scores, obs, hybrid, deeper, out):
    summary = []
    per = []
    calibration = []
    top = []
    for model, s in scores.items():
        for name, table, ycol in [
            ('hybrid_capture', hybrid, 'hybrid_label'),
            ('rhAmpSeq_selected_subset', deeper, 'rhAmpSeq_avg_delta_gt_0_2pct')
        ]:
            d = table[['key', 'guide', ycol]].merge(s, on=['key', 'guide'], validate='one_to_one')
            y = d[ycol].astype(str).str.lower().map({
                'true': 1,
                'false': 0,
                '1': 1,
                '0': 0
            }).to_numpy(dtype=int)
            m, p = pm_macro_metrics(y, d.rank_score.to_numpy(), d.guide.to_numpy())
            m.update(model=model, endpoint=name)
            summary.append(m)
            if len(p):
                p['model'] = model
                p['endpoint'] = name
                per.append(p)
        d = obs[['key', 'guide'] + ['GUIDE-seq_detected_%d' % r for r in [1, 2, 3]]].merge(
            s,
            on=['key', 'guide'],
            validate='one_to_one'
        )
        y = pm_binary_matrix(d)
        q = d[['q1', 'q2', 'q3']].to_numpy()
        q = np.clip(q, 1e-12, 1 - 1e-12)
        for guide, t in d.groupby('guide').groups.items():
            idx = d.index.get_indexer(t)
            yy = y[idx]
            qq = q[idx]
            calibration.append(dict(
                model=model,
                guide=guide,
                Brier=float(np.mean((yy - qq) ** 2)),
                marginal_logloss=float(-np.mean(yy * np.log(qq) + (1 - yy) * np.log1p(-qq)))
            ))
        for r in range(3):
            take = y[:, r] == 0
            hidden = d.loc[take].copy()
            yy = (y[take].sum(1) > 0).astype(int)
            m, p = pm_macro_metrics(yy, hidden.rank_score.to_numpy(), hidden.guide.to_numpy())
            m.update(model=model, endpoint='rep%d_unlisted_recovered_elsewhere' % (r + 1))
            summary.append(m)
            if len(p):
                p['model'] = model
                p['endpoint'] = m['endpoint']
                per.append(p)
            hidden['recovered'] = yy
            for guide, g in hidden.groupby('guide'):
                for k in [5, 10, 20]:
                    kk = min(k, len(g))
                    threshold = np.sort(g.rank_score.to_numpy())[-kk]
                    hi = g[g.rank_score > threshold]
                    tie = g[g.rank_score == threshold]
                    hits = hi.recovered.sum() + (kk - len(hi)) * tie.recovered.mean()
                    top.append(dict(
                        model=model,
                        held_rep=r + 1,
                        guide=guide,
                        K=kk,
                        hits=float(hits),
                        precision_at_K=float(hits / kk),
                        background=float(g.recovered.mean())
                    ))
    result = pd.DataFrame(summary)
    result.to_csv(out / 'metrics_summary.csv', index=False)
    pg = pd.concat(per, ignore_index=True) if per else pd.DataFrame()
    pg.to_csv(out / 'metrics_per_guide.csv', index=False)
    pd.DataFrame(calibration).to_csv(out / 'heldout_GUIDE_calibration.csv', index=False)
    pd.DataFrame(top).to_csv(out / 'recovery_at_K.csv', index=False)
    rng = np.random.RandomState(42)
    pairs = []
    for endpoint, t in pg.groupby('endpoint'):
        p = t.pivot(index='guide', columns='model', values='AUPRC')
        comparisons = [('Raw_PAM01', model) for model in cfg.MODEL_NAMES]
        comparisons.append(('Independent_Kinetic', 'SharedCas9_CTMC'))
        comparisons.append(('Calibrated_SeedMM', 'SharedCas9_CTMC'))
        for baseline, model in comparisons:
            a = p[[baseline, model]].dropna()
            v = (a[model] - a[baseline]).to_numpy()
            if not len(v):
                continue
            b = v[rng.randint(len(v), size=(5000, len(v)))].mean(1)
            pairs.append(dict(
                endpoint=endpoint,
                baseline=baseline,
                model=model,
                N_guides=len(v),
                delta_AUPRC=float(v.mean()),
                CI_low=float(np.quantile(b, 0.025)),
                CI_high=float(np.quantile(b, 0.975))
            ))
    pd.DataFrame(pairs).to_csv(out / 'paired_guide_bootstrap.csv', index=False)
    print(result.to_string(index=False), flush=True)

def pm_binary_label(series):
    out = series.astype(str).str.lower().map({
        'true': 1,
        'false': 0,
        '1': 1,
        '0': 0,
        '1.0': 1,
        '0.0': 0
    })
    if out.isna().any():
        raise ValueError('概率实验标签不是明确的0/1或布尔值')
    return out.astype(int)

def pm_pool_sensitivity(pools, out):
    from scipy.stats import spearmanr
    rows = []
    for held, pool in pools.items():
        for kind in [
            'Calibrated_Bernoulli',
            'FiniteBudget',
            'Occupancy_Detection',
            'Independent_Kinetic',
            'SharedCas9_CTMC',
            'Calibrated_SeedMM'
        ]:
            saved = json.loads((out / ('fit_%s_%s.json' % (
                held,
                kind
            ))).read_text(encoding='utf-8'))
            theta = np.asarray(saved['theta'])
            n = pool['n_off']
            q0, _, rank0 = pm_probabilities(kind, pool, theta)
            q0 = q0[:n].mean(1)
            rank0 = rank0[:n]
            scenarios = [('original', pool, 1.0)]
            for count in [100, 1000, 10000]:
                zz = min(-20.0, float(np.quantile(pool['z'], 0.05)) - 20.0)
                scenarios.append((
                    'add_low_score_' + str(count),
                    dict(pool, z=np.r_[pool['z'], np.full(count, zz)], h=np.r_[pool['h'], np.full(count, 8)]),
                        
                        
                        
                    1.0
                ))
            for count in [1, 5, 20]:
                zz = float(np.quantile(pool['z'], 0.95)) + 2.0
                scenarios.append((
                    'add_high_score_' + str(count),
                    dict(pool, z=np.r_[pool['z'], np.full(count, zz)], h=np.r_[pool['h'], np.zeros(count, dtype=int)]),
                        
                        
                        
                    1.0
                ))
            for multiplier in [0.1, 10.0]:
                scenarios.append(('background_x' + str(multiplier), pool, multiplier))
            scenarios.append((
                'remove_on_target_diagnostic',
                dict(pool, z=pool['z'][:n], h=pool['h'][:n], n_on=0),
                1.0
            ))
            original_background = cfg.BACKGROUND_WEIGHT
            try:
                for name, altered, multiplier in scenarios:
                    vars(cfg)['BACKGROUND_WEIGHT'] = original_background * multiplier
                    q, _, rank = pm_probabilities(kind, altered, theta)
                    q = q[:n].mean(1)
                    rank = rank[:n]
                    rho = float(spearmanr(
                        rank0,
                        rank)[0]) if len(np.unique(rank0)) > 1 and len(np.unique(rank
                    )) > 1 else np.nan
                    rows.append(dict(
                        guide=held,
                        model=kind,
                        scenario=name,
                        N_original=n,
                        N_added=len(altered['z']) - len(pool['z']),
                        background_multiplier=multiplier,
                        mean_abs_delta_probability=float(np.mean(abs(q - q0))),
                        max_abs_delta_probability=float(np.max(abs(q - q0))),
                        within_original_pool_Spearman=rho,
                        probability_sum_original_sites=float(q.sum()),
                        refit=False,
                        labels_used=False
                    ))
            finally:
                vars(cfg)['BACKGROUND_WEIGHT'] = original_background
    pd.DataFrame(rows).to_csv(out / 'candidate_pool_sensitivity.csv', index=False)
