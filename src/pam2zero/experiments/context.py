import hashlib
import json
from pathlib import Path
import torch
from pam2zero import config as cfg
from pam2zero.utils.io import safe_load, file_hash


def configure(root, out, cpu=False):
    cfg.ROOT = Path(root).resolve()
    cfg.OUT_DIR = Path(out).resolve()
    cfg.OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg.TRAIN_DATA_PATH = cfg.ROOT / "data/train_data/CHANGEseq_encoded.pkl"
    cfg.VAL_DATA_PATH = cfg.ROOT / "data/val_data/ttiss_encoded.pkl"
    cfg.TEST_DIR = cfg.ROOT / "data/test_data"
    cfg.initialize_device(cpu)


def load_initialization(prior_path, seeds):
    from pam2zero.models.factory import align_prior
    import numpy as np
    raw = torch.as_tensor(np.load(prior_path), dtype=torch.float32)
    if raw.shape != (24, 36) or not torch.isfinite(raw).all():
        raise ValueError("Expected a finite 24 x 36 official D matrix")
    aligned = align_prior(raw)
    aligned[21:24] = 0
    guide = aligned[1:21].clone()
    active = guide[:, [c - 1 for c in cfg.ORDINARY_CODES]].abs().flatten()
    active = active[active > 0]
    cfg.FINAL_SCALE = float(active.median().clamp_min(1e-4)) if len(active) else 1.0
    cfg.FINAL_INITIAL_STATES = {
        seed: {"prior_template": aligned.clone(), "initial_d": guide.clone()}
        for seed in seeds
    }
    return file_hash(prior_path)


def load_source(source):
    from pam2zero.experiments.paper_suite import final_source_paths, source_config
    directory, paths = final_source_paths(cfg.ROOT, Path(source).resolve())
    configs = []
    cfg.FINAL_INITIAL_STATES = {}
    for seed, path in paths.items():
        checkpoint = safe_load(path)
        state = checkpoint["state_dict"]
        if state["matrix_d"].shape != (20, 36) or state["matrix_pam"].shape != (2, 36):
            raise ValueError("The source must be the final guide20/PAM2 model")
        if torch.count_nonzero(state["initial_pam"]):
            raise ValueError("The source PAM must have been initialized to zero")
        cfg.FINAL_INITIAL_STATES[seed] = state
        configs.append(source_config(path, checkpoint))
    cfg.FINAL_SCALE = float(cfg.FINAL_INITIAL_STATES[42]["input_scale"])
    cfg.SOURCE_NUMERICS = "typed_vector" if safe_load(paths[42]).get("architecture") == "PAM2ZeroTypedPairs-v1" else safe_load(paths[42]).get(
        "numerics",
        "legacy_logits"
    )
    for key in cfg.CONFIG_KEYS:
        values = [config[key] for config in configs if key in config]
        if values and any(value != values[0] for value in values):
            raise ValueError("Source settings differ across seeds: " + key)
        if values:
            setattr(cfg, key, values[0])
    return paths


def fingerprint(settings):
    return hashlib.sha256(json.dumps(settings, sort_keys=True, default=str).encode()).hexdigest()
