import argparse
from pathlib import Path
import pandas as pd
from pam2zero import config as cfg


def main():
    from pam2zero.experiments.context import configure, load_source, fingerprint
    from pam2zero.data.loaders import load_train_ttiss_balanced, compute_group_weights
    from pam2zero.experiments.training import train_arm
    from pam2zero.evaluation.inference import read_model, score_branch
    from pam2zero.evaluation.metrics import compute_metrics_paper
    from pam2zero.utils.io import json_write
    parser = argparse.ArgumentParser(description="Retrain only the 170-parameter factorized control")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-models", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    configure(args.root, args.out, args.cpu)
    load_source(args.source_models)
    settings = dict(
        model="factor_D",
        config={key: vars(cfg)[key] for key in cfg.CONFIG_KEYS},
        seeds=cfg.SEEDS
    )
    cfg.PROTOCOL_FINGERPRINT = fingerprint(settings)
    path = cfg.OUT_DIR / "protocol.json"
    if path.exists():
        import json
        if json.loads(path.read_text())["fingerprint"] != cfg.PROTOCOL_FINGERPRINT:
            raise ValueError("Output contains another protocol")
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
    rows = []
    for seed in cfg.SEEDS:
        folder = train_arm("factor_D", seed, data, weights, cfg.PROTOCOL_FINGERPRINT)
        model, checkpoint = read_model(folder / "best_full_TTISS.pt")
        score = score_branch(model, data[4], "own")
        summary, _ = compute_metrics_paper(data[5], score, data[6])
        rows.append(dict(seed=seed, parameters=170, **summary))
        table = model.matrix_d.detach().cpu().numpy()
        pd.DataFrame(
            table,
            index=range(1, 21),
            columns=range(1, 37)).to_csv(folder / "reconstructed_D.csv"
        )
    pd.DataFrame(rows).to_csv(cfg.OUT_DIR / "factorized_TTISS_own.csv", index=False)
