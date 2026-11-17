from __future__ import annotations
from pam2zero import config as cfg
import torch.nn.functional as F
from torch import nn
import numpy as np
import pandas as pd
import random
import time
import torch
from tqdm import tqdm


def train_arm(arm, seed, data, weights, fp):
    from pam2zero.evaluation.metrics import compute_metrics_paper
    from pam2zero.models.factory import make_model
    from pam2zero.evaluation.inference import own_logits, predict_both, score_branch
    from pam2zero.utils.io import safe_load, save_atomic
    tr, y, gid, groups, val, vy, vg = data
    run = cfg.OUT_DIR / 'training' / arm / ('seed_%s' % seed)
    run.mkdir(parents=True, exist_ok=True)
    last_path = run / 'last_state.pt'
    m = make_model(seed, arm)
    direct = arm in ('own', 'own_budget', 'simple_additive', 'pure_D')
    monitor = 'own' if direct else 'full'
    opt = torch.optim.AdamW([
        dict(params=[p for n, p in m.named_parameters() if n in ('matrix_d', 'matrix_pam') and p.requires_grad], lr=cfg.LR_D, weight_decay=0.0),
            
            
            
        dict(
            params=[p for n, p in m.named_parameters() if n not in ('matrix_d', 'matrix_pam') and p.requires_grad],
                
                
            lr=cfg.LR_OTHER,
            weight_decay=cfg.WEIGHT_DECAY
        )
    ])
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt,
        mode='max',
        factor=0.5,
        patience=5,
        min_lr=1e-06
    )
    start = 1
    best = {'full': -1.0, 'own': -1.0}
    best_epoch = {'full': -1, 'own': -1}
    history = []
    stale = 0
    budget = None
    target_epoch = None
    if arm == 'own_budget':
        full = cfg.OUT_DIR / 'training/full' / ('seed_%s' % seed)
        src = safe_load(full / 'last_state.pt')
        if not src['finished']:
            raise ValueError('own_budget需要先完成同seed full')
        budget = src['history']
        target_epoch = safe_load(full / 'best_full_TTISS.pt')['epoch']
    if cfg.RESUME and last_path.is_file():
        cp = safe_load(last_path)
        if cp['fingerprint'] != fp:
            raise ValueError('续训指纹不同；请更换输出目录')
        if cp['finished']:
            required = [run / ('best_' + monitor + '_TTISS.pt')]
            if arm == 'own_budget':
                required.append(run / 'same_full_selected_epoch.pt')
            if not all((p.is_file() for p in required)):
                raise FileNotFoundError('已完成模型缺少对应最佳权重')
            print('[复用本轮完成任务]', arm, seed, flush=True)
            return run
        m.load_state_dict(cp['model'])
        opt.load_state_dict(cp['optimizer'])
        sch.load_state_dict(cp['scheduler'])
        best, best_epoch, history, stale, start = (
            cp['best'],
            cp['best_epoch'],
            cp['history'],
            cp['stale'],
            cp['epoch'] + 1
        )
        random.setstate(cp['python_rng'])
        np.random.set_state(cp['numpy_rng'])
        torch.set_rng_state(cp['torch_rng'])
        if cfg.DEVICE.type == 'cuda' and cp.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(cp['cuda_rng'])
    tx = torch.from_numpy(np.ascontiguousarray(tr, dtype=np.uint8))
    ty = torch.from_numpy(y.astype(np.int64))
    tg = torch.from_numpy(gid.astype(np.int64))
    wg = weights.to(cfg.DEVICE)
    limit = len(budget) if budget is not None else cfg.EPOCHS
    for epoch in range(start, limit + 1):
        if budget is not None:
            for pg, lr in zip(opt.param_groups, budget[epoch - 1]['lr_used']):
                pg['lr'] = lr
        lr_used = [p['lr'] for p in opt.param_groups]
        order = torch.randperm(
            len(y),
            generator=torch.Generator().manual_seed(seed * 100000 + epoch)
        )
        m.train()
        total = torch.zeros((), device=cfg.DEVICE)
        seen = 0
        positive = 0
        t0 = time.time()
        for off in tqdm(
            range(0, len(y), cfg.BATCH_SIZE),
            desc='%s seed%s E%03d' % (arm, seed, epoch),
            leave=False,
            mininterval=2.0
        ):
            ix = order[off:off + cfg.BATCH_SIZE]
            yy = ty[ix]
            positive += int(yy.sum())
            seen += len(yy)
            xx = tx[ix].to(cfg.DEVICE, dtype=torch.long)
            yy = yy.to(cfg.DEVICE)
            gg = tg[ix].to(cfg.DEVICE)
            opt.zero_grad(set_to_none=True)
            logits = own_logits(m, xx) if direct else m(xx)
            ce = F.cross_entropy(logits, yy, reduction='none')
            loss = (cfg.FOCAL_ALPHA * (1 - (-ce).exp()).pow(cfg.FOCAL_GAMMA) * ce * wg[gg]).mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('非有限loss')
            loss.backward()
            nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], cfg.GRAD_CLIP)
            opt.step()
            total += loss.detach() * len(yy)
        if seen != len(y) or positive != int(y.sum()):
            raise AssertionError('全量训练覆盖失败')
        vs = {'own': score_branch(m, val, 'own')} if direct else predict_both(m, val)
        vm = {k: compute_metrics_paper(vy, v, vg)[0] for k, v in vs.items()}
        improved = vm[monitor]['Macro_AUPRC'] > best[monitor]
        if budget is None:
            sch.step(vm[monitor]['Macro_AUPRC'])

        def checkpoint(selection):
            return dict(
                config={k: vars(cfg)[k] for k in cfg.CONFIG_KEYS},
                numerics=cfg.SOURCE_NUMERICS,
                architecture=cfg.ARCHITECTURE,
                fingerprint=fp,
                arm=arm,
                seed=seed,
                epoch=epoch,
                selection=selection,
                validation=vm,
                selection_metric_value=vm[selection]['Macro_AUPRC'],
                state_dict={k: v.detach().cpu() for k, v in m.state_dict().items()}
            )
        for branch, v in vm.items():
            if v['Macro_AUPRC'] > best[branch]:
                best[branch] = v['Macro_AUPRC']
                best_epoch[branch] = epoch
                save_atomic(checkpoint(branch), run / ('best_' + branch + '_TTISS.pt'))
        if target_epoch == epoch:
            save_atomic(checkpoint('own'), run / 'same_full_selected_epoch.pt')
        stale = 0 if improved else stale + 1
        history.append(dict(
            epoch=epoch,
            loss=float(total) / len(y),
            lr_used=lr_used,
            seen=seen,
            positive_seen=positive,
            seconds=time.time() - t0,
            **{k + '_TTISS_AUPRC': v['Macro_AUPRC'] for k, v in vm.items()}
        ))
        pd.DataFrame(history).to_csv(run / 'history.csv', index=False)
        finished = epoch == limit or (budget is None and stale >= cfg.PATIENCE)
        save_atomic(
            dict(fingerprint=fp, epoch=epoch, finished=finished, model=m.state_dict(), optimizer=opt.state_dict(), scheduler=sch.state_dict(), best=best, best_epoch=best_epoch, history=history, stale=stale, python_rng=random.getstate(), numpy_rng=np.random.get_state(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if cfg.DEVICE.type == 'cuda' else None),
                
                
                
            last_path
        )
        print(
            arm,
            'seed',
            seed,
            'epoch',
            epoch,
            'loss',
            history[-1]['loss'],
            'TTISS',
            {k: round(v['Macro_AUPRC'], 6) for k, v in vm.items()},
            'best',
            best_epoch,
            'seconds',
            round(history[-1]['seconds'], 1),
            flush=True
        )
        if finished:
            break
    return run
