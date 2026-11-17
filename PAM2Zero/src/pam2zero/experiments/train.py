import argparse
from pathlib import Path
from pam2zero import config as cfg


def main():
    from pam2zero.experiments.context import configure, load_initialization, fingerprint
    from pam2zero.data.loaders import load_train_ttiss_balanced, compute_group_weights
    from pam2zero.experiments.training import train_arm
    from pam2zero.utils.io import json_write, file_hash
    from pam2zero.utils.provenance import code_hashes
    parser = argparse.ArgumentParser(description="Train the final zero-PAM model")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=cfg.SEEDS)
    parser.add_argument("--epochs", type=int, default=cfg.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    parser.add_argument(
        "--numerics",
        choices=["typed_vector", "legacy_logits"],
        default="typed_vector"
    )
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    configure(args.root, args.out, args.cpu)
    cfg.SEEDS = args.seeds
    cfg.EPOCHS, cfg.BATCH_SIZE = args.epochs, args.batch_size
    cfg.SOURCE_NUMERICS = args.numerics
    prior_hash = load_initialization(args.prior, cfg.SEEDS)
    inputs = [Path(str(stem).replace('.pkl', '_' + field + '.pkl'))
              for stem in [cfg.TRAIN_DATA_PATH, cfg.VAL_DATA_PATH]
              for field in ['pos_list', 'labels', 'sg_list']]
    settings = dict(
        input_sha256={str(path): file_hash(path) for path in inputs},
        code_sha256=code_hashes(),
        model="PAM2Zero",
        prior_sha256=prior_hash,
        numerics=cfg.SOURCE_NUMERICS,
        
                    config={key: vars(cfg)[key] for key in cfg.CONFIG_KEYS}, seeds=cfg.SEEDS)
    cfg.PROTOCOL_FINGERPRINT = fingerprint(settings)
    path = cfg.OUT_DIR / "protocol.json"
    if path.exists():
        import json
        if json.loads(path.read_text())["fingerprint"] != cfg.PROTOCOL_FINGERPRINT:
            raise ValueError("Output contains another protocol; choose another output directory")
    json_write(path, dict(fingerprint=cfg.PROTOCOL_FINGERPRINT, **settings))
    data = load_train_ttiss_balanced()
    weights = compute_group_weights(
        data[2],
        data[3],
        cfg.GROUP_POWER,
        cfg.GROUP_FLOOR,
        cfg.GROUP_CAP,
        cfg.OUT_DIR
    )
    for seed in cfg.SEEDS:
        train_arm("full", seed, data, weights, cfg.PROTOCOL_FINGERPRINT)
