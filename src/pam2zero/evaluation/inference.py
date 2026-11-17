from __future__ import annotations
from pam2zero import config as cfg
import torch.nn.functional as F
import hashlib
import json
import numpy as np
import torch

@torch.no_grad()
def _predict_both_legacy(model, codes):
    model.eval()
    result = {'full': [], 'own': []}
    for start in range(0, len(codes), cfg.EVAL_BATCH_SIZE):
        x = torch.as_tensor(
            codes[start:start + cfg.EVAL_BATCH_SIZE],
            device=cfg.DEVICE,
            dtype=torch.long
        )
        c = model.contributions(x)
        temp = model.temperature()
        for branch, a, b in [
            ('full', c['match_total'], c['mismatch_total']),
            ('own', c['own_match'], c['own_mismatch'])
        ]:
            logits = torch.stack([b - 0.5 * c['pam_score'], a + 0.5 * c['pam_score']], 1) / temp
            result[branch].append((logits[:, 1] - logits[:, 0]).cpu().numpy())
    result = {k: np.concatenate(v) for k, v in result.items()}
    if any((not np.isfinite(v).all() for v in result.values())):
        raise FloatingPointError('预测NaN/Inf')
    return result

@torch.no_grad()
def predict_official(m, d, codes, batch):
    result = {'CrisprPr_M': [], 'CrisprPr_D': [], 'CrisprPr_ensemble': []}
    for start in range(0, len(codes), batch):
        a = codes[start:start + batch]
        a = np.concatenate([np.full((len(a), 1), 31, dtype=np.uint8), a], axis=1)
        x = torch.as_tensor(a, dtype=torch.int8, device=cfg.DEVICE)
        mo, do = (m(x), d(x))
        if mo.shape != (len(a), 1) or do.shape != (len(a), 1):
            raise ValueError('官方模型应返回(B,1)原始logit，实际%s / %s；请使用原发布的TorchScript权重。' % (
                mo.shape,
                do.shape
            ))
        if not torch.isfinite(mo).all() or not torch.isfinite(do).all():
            raise ValueError('官方模型输出NaN/Inf')
        for k, v in [('CrisprPr_M', mo), ('CrisprPr_D', do), ('CrisprPr_ensemble', (mo + do) / 2)]:
            result[k].append(v[:, 0].cpu().numpy())
    return {k: np.concatenate(v) for k, v in result.items()}

def _own_logits_legacy(model, codes):
    from pam2zero.models.baselines import FreeAdditive
    if isinstance(model, FreeAdditive):
        return model(codes)
    x = codes[:, :20].long()
    match = model.match_lut[x]
    gain = model.team_gains()
    f = (F.softplus(model.matrix_d[
        model.positions[None],
        x - 1
    ] / model.input_scale) + cfg.FORCE_FLOOR) * torch.where(
        match,
        gain[0],
        gain[1]
    )
    a = (f * match).sum(1)
    b = (f * ~match).sum(1)
    pam = model.matrix_pam[
        model.pam_positions[None],
        codes[
            :,
            21:23].long() - 1].sum(1) / model.input_scale if model.use_pam else a.new_zeros(len(a
        )
    )
    return torch.stack([b - 0.5 * pam, a + 0.5 * pam], 1) / model.temperature()

def read_model(path):
    from pam2zero.models.factory import make_model
    from pam2zero.utils.io import safe_load
    cp = safe_load(path)
    if cp.get('architecture') != cfg.ARCHITECTURE or cp.get('fingerprint') != cfg.PROTOCOL_FINGERPRINT:
        raise ValueError('模型配置/输入身份不符：' + str(path))
    arm = cp['arm']
    m = make_model(int(cp['seed']), arm)
    m.load_state_dict(cp['state_dict'], strict=True)
    m.eval()
    m.requires_grad_(False)
    return (m, cp)

@torch.no_grad()
def _score_branch_legacy(model, codes, branch):
    from pam2zero.models.baselines import FreeAdditive
    model.eval()
    out = []
    for i in range(0, len(codes), cfg.EVAL_BATCH_SIZE):
        x = torch.as_tensor(codes[i:i + cfg.EVAL_BATCH_SIZE], dtype=torch.long, device=cfg.DEVICE)
        z = model(x) if branch == 'full' or isinstance(
            model,
            FreeAdditive) else own_logits(model,
            x
        )
        out.append((z[:, 1] - z[:, 0]).cpu().numpy())
    result = np.concatenate(out)
    if not np.isfinite(result).all():
        raise FloatingPointError('预测含NaN/Inf')
    return result

def score_own_float64(model, codes):
    state = {k: v.detach().cpu().numpy().astype(np.float64) for k, v in model.state_dict().items()}
    return numpy_reference(state, codes)['own']

def dataset_token(codes):
    return hashlib.sha256(np.ascontiguousarray(codes).tobytes()).hexdigest()

def cached_score(spec, codes, token):
    from pam2zero.utils.io import file_hash
    from pam2zero.evaluation.suite import load_spec
    key = hashlib.sha256(json.dumps(
        dict(cp=file_hash(spec['path']), branch=spec['branch'], intervention=spec.get('intervention'), codes=token, script=file_hash(__file__)),
            
            
            
        sort_keys=True).encode(
    )).hexdigest()
    cache = cfg.OUT_DIR / 'prediction_cache'
    cache.mkdir(exist_ok=True)
    path = cache / (key + '.npy')
    if path.is_file():
        a = np.load(path)
        if a.shape != (len(codes),) or not np.isfinite(a).all():
            raise ValueError('损坏的预测缓存：' + str(path))
        return a
    m, _ = load_spec(spec)
    a = score_branch(m, codes, spec['branch'])
    del m
    tmp = path.with_suffix('.tmp.npy')
    np.save(tmp, a)
    tmp.replace(path)
    return a

def own_logits(model, codes):
    from pam2zero.models.baselines import FreeAdditive
    from pam2zero.models.competition import typed_source_scores
    if cfg.SOURCE_NUMERICS != 'typed_vector' or isinstance(model, FreeAdditive):
        return _own_logits_legacy(model, codes)
    z = typed_source_scores(model, codes)['own']
    return torch.stack([-z / 2, z / 2], 1)

@torch.no_grad()
def predict_both(model, codes):
    from pam2zero.models.competition import typed_source_scores
    if cfg.SOURCE_NUMERICS != 'typed_vector':
        return _predict_both_legacy(model, codes)
    model.eval()
    out = {'full': [], 'own': []}
    for i in range(0, len(codes), cfg.EVAL_BATCH_SIZE):
        x = torch.as_tensor(codes[i:i + cfg.EVAL_BATCH_SIZE], dtype=torch.long, device=cfg.DEVICE)
        v = typed_source_scores(model, x)
        for k in out:
            out[k].append(v[k].cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}

@torch.no_grad()
def score_branch(model, codes, branch):
    from pam2zero.models.baselines import FreeAdditive
    if cfg.SOURCE_NUMERICS != 'typed_vector' or isinstance(model, FreeAdditive):
        return _score_branch_legacy(model, codes, branch)
    return predict_both(model, codes)[branch]

from pam2zero.models.numpy_reference import numpy_reference
