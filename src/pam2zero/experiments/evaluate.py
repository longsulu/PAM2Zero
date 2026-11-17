import argparse
from pathlib import Path
import pandas as pd
from pam2zero import config as cfg


def main():
    from pam2zero.experiments.context import configure, load_source
    from pam2zero.models.factory import make_model
    from pam2zero.data.loaders import _read_pkl_bundle, load_test_csv_original
    from pam2zero.data.encoding import _off_target_mask, normalize_to23, _sg_str
    from pam2zero.evaluation.inference import score_branch
    from pam2zero.evaluation.metrics import compute_metrics_paper, threshold_mcc, classification
    from pam2zero.utils.io import safe_load, file_hash, json_write
    parser = argparse.ArgumentParser(description="Evaluate existing five-seed final checkpoints")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source-models", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", default=cfg.TEST_NAMES)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    configure(args.root, args.out, args.cpu)
    paths = load_source(args.source_models)
    hashes = {str(path): file_hash(path) for path in paths.values()}
    val = _read_pkl_bundle(cfg.VAL_DATA_PATH)
    keep = _off_target_mask(val["pos_list"])
    vx, aligned = normalize_to23(val["pos_list"][keep], "TTISS")
    vy = val["labels"][keep][aligned]
    rows = []
    models = []
    for seed, path in paths.items():
        model = make_model(seed, "full")
        model.load_state_dict(safe_load(path)["state_dict"], strict=True)
        model.eval()
        threshold = threshold_mcc(vy, score_branch(model, vx, "own"))
        models.append((seed, model, threshold))
    for name in args.datasets:
        x, y, groups = load_test_csv_original(cfg.TEST_DIR / (name + ".csv"))
        for seed, model, threshold in models:
            score = score_branch(model, x, "own")
            summary, per = compute_metrics_paper(y, score, groups)
            summary.update(classification(y, score, threshold))
            rows.append(dict(dataset=name, seed=seed, threshold=threshold, **summary))
            per.to_csv(cfg.OUT_DIR / (name + "_seed_" + str(seed) + "_per_guide.csv"), index=False)
    frame = pd.DataFrame(rows)
    frame.to_csv(cfg.OUT_DIR / "run_metrics.csv", index=False)
    numeric = [name for name in frame.select_dtypes(include="number").columns if name != "seed"]
    frame.groupby("dataset")[numeric].agg([
        "mean",
        "std"
    ]).to_csv(cfg.OUT_DIR / "summary_mean_std.csv")
    if any(file_hash(path) != value for path, value in hashes.items()):
        raise RuntimeError("A source checkpoint changed")
    json_write(
        cfg.OUT_DIR / "evaluation_protocol.json",
        dict(branch="own", thresholds="TTISS pooled MCC", source_sha256=hashes)
    )
