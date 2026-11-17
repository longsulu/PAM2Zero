from __future__ import annotations
from pathlib import Path
import hashlib
import json
import numpy as np
import random
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def file_hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save_atomic(value, path):
    tmp = path.with_name(path.name + '.tmp')
    torch.save(value, tmp)
    tmp.replace(path)


def safe_load(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def dump_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str),
        encoding='utf-8'
    )


def json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    tmp.replace(path)


def sha256(path):
    return file_hash(path)
