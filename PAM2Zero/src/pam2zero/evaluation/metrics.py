from __future__ import annotations
from pam2zero import config as cfg
from collections import Counter
import math
from sklearn import metrics
import numpy as np
import pandas as pd


def compute_metrics_paper(labels, scores, groups):
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    groups = np.asarray(groups, dtype=object)
    rows = []
    for sg in np.unique(groups):
        mask = groups == sg
        y, s = (labels[mask], scores[mask])
        counts = Counter(y)
        if counts[0] == 0 or counts[1] == 0:
            continue
        roc = metrics.roc_auc_score(y, s)
        precision, recall, _ = metrics.precision_recall_curve(y, s)
        precision[recall == 0] = 1.0
        pr = metrics.auc(recall, precision)
        rows.append({
            'sgRNA': str(sg),
            'N': int(len(y)),
            'Positive': int(y.sum()),
            'AUPRC': float(pr),
            'AP': float(metrics.average_precision_score(y, s)),
            'AUROC': float(roc)
        })
    per_group = pd.DataFrame(rows)
    if per_group.empty:
        raise RuntimeError('没有同时包含正负样本的 sgRNA 组')
    summary = {
        'Valid_sgRNA': len(per_group),
        'Macro_AUPRC': float(per_group['AUPRC'].mean()),
        'Macro_AP': float(per_group['AP'].mean()),
        'Median_AUPRC': float(per_group['AUPRC'].median()),
        'Macro_AUROC': float(per_group['AUROC'].mean()),
        'Global_AUPRC': float(metrics.average_precision_score(labels, scores)),
        'Global_AUROC': float(metrics.roc_auc_score(
            labels,
            scores
        )) if np.unique(labels).size == 2 else float('nan')
    }
    return (summary, per_group)


def threshold_mcc(y, s):
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    o = np.argsort(-s, kind='mergesort')
    v = s[o]
    t = y[o]
    end = np.r_[np.flatnonzero(v[:-1] != v[1:]), len(v) - 1]
    tp = np.cumsum(t)[end].astype(float)
    fp = end + 1 - tp
    fn = t.sum() - tp
    tn = len(t) - t.sum() - fp
    den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = np.divide(tp * tn - fp * fn, den, out=np.zeros_like(tp), where=den > 0)
    i = int(np.argmax(mcc))
    return float(v[end[i]]) if mcc[i] > 0 else float(np.nextafter(v[0], np.inf))


def classification(y, s, threshold):
    pred = np.asarray(s) >= threshold
    tn, fp, fn, tp = map(int, metrics.confusion_matrix(y, pred, labels=[0, 1]).ravel())
    den = math.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    return dict(
        ACC=float((tp + tn) / len(y)),
        MCC=float((tp * tn - fp * fn) / den) if den else 0.0,
        Precision=float(tp / (tp + fp)) if tp + fp else 0.0,
        Recall=float(tp / (tp + fn)) if tp + fn else 0.0,
        F1=float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0,
        Specificity=float(tn / (tn + fp)) if tn + fp else 0.0,
        TP=int(tp),
        FP=int(fp),
        TN=int(tn),
        FN=int(fn)
    )


def rank_budget_metrics(y, s):
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    positive = int(y.sum())
    out = {}
    for k in cfg.RANK_KS:
        kk = min(k, len(y))
        boundary = np.partition(s, len(s) - kk)[len(s) - kk]
        hi = s > boundary
        tie = s == boundary
        hits = float(y[hi].sum() + (kk - int(hi.sum())) * y[tie].mean())
        out['Recall_at_' + str(k)] = hits / positive if positive else np.nan
        out['Precision_at_' + str(k)] = hits / kk
    order = np.argsort(-s, kind='mergesort')
    v = s[order]
    yy = y[order]
    ends = np.r_[np.flatnonzero(v[:-1] != v[1:]), len(v) - 1]
    tp = np.cumsum(yy)[ends]
    for target in [0.8, 0.9]:
        eligible = np.flatnonzero(tp >= positive * target)
        n = int(ends[eligible[0]] + 1) if positive and len(eligible) else np.nan
        out['Candidates_for_recall' + str(int(target * 100))] = n
    out['Unique_scores'] = int(len(ends))
    out['Score_unique_fraction'] = len(ends) / len(y)
    return out


def metric_tables(y, s, g, threshold):
    summary, per = compute_metrics_paper(y, s, g)
    extra = []
    for guide in per.sgRNA:
        mask = g == guide
        cm = classification(y[mask], s[mask], threshold)
        extra.append(dict(
            sgRNA=guide,
            **{k + '_locked': cm[k] for k in ['MCC', 'ACC', 'Recall', 'Precision', 'F1', 'Specificity']},
                
                
                
            **rank_budget_metrics(y[mask], s[mask])
        ))
    per = per.merge(pd.DataFrame(extra), on='sgRNA', validate='one_to_one')
    summary.update({k: float(per[k].mean()) for k in [
        'MCC_locked',
        'ACC_locked',
        'Recall_locked',
        'Precision_locked',
        'F1_locked',
        'Specificity_locked'] + [c for c in per if c.startswith((
            'Recall_at_',
            'Precision_at_',
            'Candidates_for_recall',
            'Score_unique'
        ))
    ]})
    summary.update({
        'Global_' + k + '_locked': v for k,
        v in classification(y, s, threshold).items()
    })
    summary.update({'Global_' + k + '_zero_logit': v for k, v in classification(y, s, 0.0).items()})
    return (summary, per)


def score_and_report(
    dataset,
    cohort,
    codes,
    y,
    g,
    specs,
    thresholds,
    rows,
    pers,
    scores,
    mask=None
):
    if mask is None:
        mask = np.ones(len(y), bool)
    if not mask.any():
        return
    for spec in specs:
        strategy = spec['strategy']
        seed = str(spec['seed'])
        s = scores[strategy, seed][mask]
        if not any((len(np.unique(y[mask][g[mask] == k])) == 2 for k in np.unique(g[mask]))):
            continue
        met, pg = metric_tables(y[mask], s, g[mask], thresholds[strategy, seed])
        rows.append(dict(
            dataset=dataset,
            cohort=cohort,
            strategy=strategy,
            seed=seed,
            N=int(mask.sum()),
            Positive=int(y[mask].sum()),
            **met
        ))
        pg = pg.assign(dataset=dataset, cohort=cohort, strategy=strategy, seed=seed)
        pers.append(pg)
