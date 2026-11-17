from __future__ import annotations
from pam2zero import config as cfg
import numpy as np
import pandas as pd
import time
import torch


@torch.no_grad()
def final_efficiency(tr):
    from pam2zero.evaluation.inference import own_logits, read_model
    m, _ = read_model(cfg.OUT_DIR / 'training/full' / 'seed_42' / 'best_full_TTISS.pt')
    rows = []
    for size in [1, 256, 4096]:
        idx = np.arange(size) % len(tr)
        x = torch.as_tensor(tr[idx], dtype=torch.long, device=cfg.DEVICE)
        for branch in ['own', 'full']:
            fn = (lambda: own_logits(m, x)) if branch == 'own' else lambda: m(x)
            for _ in range(3):
                fn()
            if cfg.DEVICE.type == 'cuda':
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            t = time.perf_counter()
            for _ in range(20):
                fn()
            if cfg.DEVICE.type == 'cuda':
                torch.cuda.synchronize()
            seconds = (time.perf_counter() - t) / 20
            rows.append(dict(
                branch=branch,
                batch=size,
                seconds_per_batch=seconds,
                samples_per_second=size / seconds,
                peak_GPU_bytes=torch.cuda.max_memory_allocated() if cfg.DEVICE.type == 'cuda' else None,
                    
                    
                    
                device=str(cfg.DEVICE),
                hardware=torch.cuda.get_device_name(0) if cfg.DEVICE.type == 'cuda' else 'CPU',
                scope='device-resident model forward; excludes loading/encoding/transfer',
                repeats=20
            ))
    pd.DataFrame(rows).to_csv(cfg.OUT_DIR / 'efficiency.csv', index=False)
