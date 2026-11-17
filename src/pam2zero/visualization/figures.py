from __future__ import annotations
from pam2zero import config as cfg
from pathlib import Path
import numpy as np
import pandas as pd


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


def paper_plots():
    plt = ensure_plotting()
    out = cfg.OUT_DIR / 'figures'
    out.mkdir(exist_ok=True)
    summary = pd.read_csv(cfg.OUT_DIR / 'summary_mean_std.csv')
    part = summary[summary.dataset.isin(cfg.TEST_NAMES) & summary.strategy.isin([
        cfg.PRIMARY,
        'Control_fullSelect_full',
        'Official_MD'
    ])]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    for ax, metric in zip(axs, ['Macro_AUPRC', 'Macro_AP']):
        for strategy, b in part.groupby('strategy'):
            b = b.set_index('dataset').reindex(cfg.TEST_NAMES)
            ax.errorbar(
                np.arange(7),
                b[metric + '_mean'],
                yerr=b[metric + '_std'].fillna(0),
                marker='o',
                capsize=3,
                label=strategy
            )
        ax.set_xticks(np.arange(7))
        ax.set_xticklabels(
            [n.replace('_test', '') for n in cfg.TEST_NAMES],
            rotation=45,
            ha='right'
        )
        ax.set_ylabel(metric)
        ax.legend(fontsize=7)
    save_figure(fig, out / 'main_comparison')
    selected = summary[summary.dataset.isin(cfg.TEST_NAMES) & summary.strategy.isin([
        cfg.PRIMARY,
        'Ablation_ownTrain_ownSelect_own',
        'Ablation_ownBudget_sameFullEpoch_own',
        cfg.SIMPLE_LABELS['simple_additive'],
        cfg.SIMPLE_LABELS['pure_D']] + ['Component_' + a + '_fullSelect_own' for a in cfg.COMPONENT_ARMS
    ])]
    q = selected.groupby('strategy')[[
        'Macro_AUPRC_mean',
        'Macro_AP_mean'
    ]].mean().sort_values('Macro_AUPRC_mean')
    fig, axs = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    for ax, c in zip(axs, q.columns):
        ax.barh(q.index, q[c])
        ax.set_xlabel('Equal-weight Mean7 ' + c)
        ax.tick_params(axis='y', labelsize=7)
    save_figure(fig, out / 'ablation_Mean7')
    interp = cfg.OUT_DIR / 'interpretation'
    d = pd.read_csv(interp / 'effective_node_contributions.csv')
    pairs = [cfg.ENCODE_DICT_36[k] for k in [
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
    ]]
    fig, axs = plt.subplots(3, 1, figsize=(13, 9), constrained_layout=True)
    for ax, c in zip(axs, ['initial_D', 'learned_D', 'own_logit_contribution']):
        tab = d.groupby(['code', 'position'])[c].mean().unstack().reindex(pairs)
        limit = max(abs(tab.min().min()), abs(tab.max().max()))
        im = ax.imshow(tab, aspect='auto', cmap='RdBu_r', vmin=-limit, vmax=limit)
        ax.set_yticks(range(16))
        ax.set_yticklabels([{v: k for k, v in cfg.ENCODE_DICT_36.items()}[i] for i in pairs])
        ax.set_xticks(range(20))
        ax.set_xticklabels(range(1, 21))
        ax.set_title(c + ' (seed mean)')
        fig.colorbar(im, ax=ax)
    save_figure(fig, out / 'D_effective_contributions')
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, file, col, index, reference in [
        (axs[0], 'edge_gains', 'gain', 'edge', 1.0),
        (axs[1], 'distance_strengths', 'strength', 'distance', cfg.CONTEST_INIT)
    ]:
        q = pd.read_csv(interp / (file + '.csv')).groupby(index)[col].agg(['mean', 'std'])
        ax.errorbar(q.index, q['mean'], yerr=q['std'], marker='o', capsize=2)
        ax.axhline(reference, ls='--', color='gray')
        ax.set_xlabel(index)
        ax.set_ylabel(col)
    save_figure(fig, out / 'competition_parameters')
    singles = pd.read_csv(interp / 'single_counterfactuals.csv')
    q = singles.groupby(['seed', 'branch', 'position']).delta_logit.mean().reset_index()
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    for branch, b in q.groupby('branch'):
        t = b.groupby('position').delta_logit.agg(['mean', 'std'])
        ax.errorbar(t.index, t['mean'], yerr=t['std'], marker='o', capsize=2, label=branch)
    ax.set_xlabel('Guide position (20 proximal PAM)')
    ax.set_ylabel('Mutation logit - WT logit')
    ax.legend()
    save_figure(fig, out / 'single_counterfactuals')
    doubles = pd.read_csv(interp / 'pair_finite_differences.csv')
    fig, axs = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, branch in zip(axs, ['own', 'full']):
        t = doubles[doubles.branch.eq(branch)].groupby([
            'position_i',
            'position_j'
        ]).synergy.mean().unstack()
        im = ax.imshow(t, aspect='auto', cmap='RdBu_r')
        ax.set_title(branch + ' pair finite differences')
        fig.colorbar(im, ax=ax)
    save_figure(fig, out / 'pair_nonadditivity')


def final_paper_extras():
    out = cfg.OUT_DIR / 'paper_tables'
    out.mkdir(exist_ok=True)
    df = pd.read_csv(cfg.OUT_DIR / 'all_run_metrics.csv')
    summary = pd.read_csv(cfg.OUT_DIR / 'summary_mean_std.csv')
    for metric in [
        'Macro_AUPRC',
        'Macro_AP',
        'Macro_AUROC',
        'MCC_locked',
        'F1_locked',
        'ACC_locked'
    ]:
        tab = summary[summary.dataset.isin(cfg.TEST_NAMES) & summary.cohort.eq('original_protocol')].pivot(
            index='strategy',
            columns='dataset',
            values=metric + '_mean'
        )
        tab = tab.reindex(columns=cfg.TEST_NAMES)
        tab['Mean7'] = tab.mean(axis=1)
        tab.to_csv(out / (metric + '_by_dataset.csv'))
        tab.to_latex(out / (metric + '_by_dataset.tex'), float_format='%.6f')
        plt = ensure_plotting()
        fig, ax = plt.subplots(figsize=(12, max(5, len(tab) * 0.25)), constrained_layout=True)
        im = ax.imshow(tab.values, aspect='auto', cmap='viridis')
        ax.set_xticks(range(len(tab.columns)))
        ax.set_xticklabels(tab.columns, rotation=40, ha='right', fontsize=7)
        ax.set_yticks(range(len(tab)))
        ax.set_yticklabels(tab.index, fontsize=6)
        ax.set_title(metric)
        fig.colorbar(im, ax=ax)
        save_figure(fig, cfg.OUT_DIR / 'figures' / (metric + '_all_conditions'))
    main = df[df.strategy.eq(cfg.PRIMARY) & df.dataset.isin(cfg.TEST_NAMES)]
    rows = []
    for strategy, b in df[df.dataset.isin(cfg.TEST_NAMES)].groupby('strategy'):
        for omitted in cfg.TEST_NAMES:
            q = b[b.dataset.ne(omitted)].groupby('seed').Macro_AUPRC.mean()
            rows.append(dict(
                strategy=strategy,
                omitted=omitted,
                mean=float(q.mean()),
                std=float(q.std())
            ))
    pd.DataFrame(rows).to_csv(out / 'leave_one_dataset_out_sensitivity.csv', index=False)
    (out / 'INTERPRETATION_LIMITS.txt').write_text(
        'D values before softplus are not signed forces. Effective node contributions include team gains and temperature. OWN has zero logit pair interaction by construction. FULL finite differences show model nonadditivity only. Edge and distance parameters can compensate and are not separately identifiable physical constants. Correlation/ablation cannot establish molecular causality.',
            
            
            
        encoding='utf8'
    )
