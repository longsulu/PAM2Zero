# PAM2Zero

A compact and decomposable model for CRISPR-Cas9 off-target prediction.

The final model uses all 20 guide positions and the last two PAM positions. The first PAM position is ignored. The retained PAM parameters are initialized to zero and updated during training.

The model allocates 833 parameters. The deployed OWN output uses 795 active parameters; the FULL branch supports competition-assisted training and TTISS checkpoint selection.

## Installation

Use Python 3.8 or later with a compatible PyTorch installation.

Run the following commands from the project root, which contains `pyproject.toml`:

```bash
python -m pip install -e .
python -m pam2zero self-test
```

## Data downloads

The processed training data, validation data, seven test datasets, and initial D matrix are provided by the original CrisprPr repository.

| Input | Official download location |
|---|---|
| CHANGE-seq training data | https://github.com/mochew/CrisprPr/tree/main/data/train_data |
| TTISS validation data | https://github.com/mochew/CrisprPr/tree/main/data/val_data |
| Seven test datasets | https://github.com/mochew/CrisprPr/tree/main/data/test_data |
| Initial D matrix | https://github.com/mochew/CrisprPr/blob/main/models/D_models/crisot_score_param_new.npy |

Original repository: https://github.com/mochew/CrisprPr

CrisprPr paper: https://doi.org/10.1093/bib/bbag140

### Download using Git LFS

The upstream data files use Git LFS. Install Git LFS from https://git-lfs.com/, then run:

```bash
git lfs install
git clone https://github.com/mochew/CrisprPr.git CrisprPr-upstream
cd CrisprPr-upstream
git checkout d207ce0b5e5f7c1d9ba47d5d0f19edc8f2beafc2
git lfs pull --include="data/train_data/*,data/val_data/*,data/test_data/*,models/D_models/crisot_score_param_new.npy"
```

The commit above pins the upstream version used for these download instructions.

Copy the required files into the PAM2Zero directories shown below. A tiny text file beginning with `version https://git-lfs.github.com/spec/v1` is an LFS pointer, not the actual dataset. Run `git lfs pull` to download the file contents.

## Required directory layout

```text
PAM2Zero/
├── pyproject.toml
├── README.md
├── README_CN.md
├── src/
│   └── pam2zero/
├── data/
│   ├── train_data/
│   │   ├── CHANGEseq_encoded_pos_list.pkl
│   │   ├── CHANGEseq_encoded_labels.pkl
│   │   └── CHANGEseq_encoded_sg_list.pkl
│   ├── val_data/
│   │   ├── ttiss_encoded_pos_list.pkl
│   │   ├── ttiss_encoded_labels.pkl
│   │   └── ttiss_encoded_sg_list.pkl
│   └── test_data/
│       ├── CRISPOR_test.csv
│       ├── GUIDEseq_test.csv
│       ├── HEK293T_test.csv
│       ├── II6_test.csv
│       ├── K562_test.csv
│       ├── guideseqCT_test.csv
│       └── pdh_test.csv
├── models/
│   └── D_models/
│       └── crisot_score_param_new.npy
└── results/
```

The six PKL files and the initial D matrix are required for training. The seven CSV files are required for test evaluation.

The names `CHANGEseq_encoded.pkl` and `ttiss_encoded.pkl` used internally are bundle stems. They resolve to the three suffixed files listed above; no additional combined PKL file is needed.

Preserve the original sample order across each PKL triplet. Use the upstream processed CSV files without changing their column layout.

The original D matrix must have shape `24 × 36`. Do not manually reverse the matrix or modify its PAM rows. Alignment and zero initialization of the retained PAM channel are handled by the training code.

## Train the final model

Run from the PAM2Zero project root:

```bash
python -m pam2zero train --root . --prior models/D_models/crisot_score_param_new.npy --out results/final833_training --seeds 42 123 3407 2025 2026
```

This trains five independent random seeds, not five cross-validation folds. It trains only the final model, without component ablations.

Default settings include:

- Maximum epochs: 100.
- Batch size: 4096.
- TTISS FULL Macro AUPRC for learning-rate scheduling, early stopping, and primary checkpoint selection.
- Group weighting with `GROUP_POWER=0.5`, `GROUP_FLOOR=0.25`, and `GROUP_CAP=4.0`.

See `src/pam2zero/config.py` for the complete configuration. The resolved experiment settings are saved in `protocol.json`.

Primary checkpoints are written to:

```text
results/final833_training/
├── protocol.json
└── training/full/
    ├── seed_42/best_full_TTISS.pt
    ├── seed_123/best_full_TTISS.pt
    ├── seed_3407/best_full_TTISS.pt
    ├── seed_2025/best_full_TTISS.pt
    └── seed_2026/best_full_TTISS.pt
```

OWN-selected checkpoints and resume states are also saved.

## Evaluate the trained models

After all five training runs finish:

```bash
python -m pam2zero evaluate --root . --source-models results/final833_training --out results/final833_evaluation
```

`--source-models` must point to the training output directory, not an individual `.pt` file. The current evaluator requires all five standard seeds.

The evaluator selects the FULL-selected checkpoints and computes their OWN outputs on the seven test datasets.

Ranking metrics include guide-macro AUPRC, AP, and AUROC. The classification threshold is selected using pooled MCC on TTISS and then locked for test evaluation. Pooled classification metrics should be distinguished from guide-macro ranking metrics.

Outputs include:

```text
results/final833_evaluation/
├── run_metrics.csv
├── summary_mean_std.csv
├── CRISPOR_test_seed_42_per_guide.csv
├── ...
└── evaluation_protocol.json
```

- `run_metrics.csv`: results for each dataset and seed.
- `summary_mean_std.csv`: five-seed means and sample standard deviations.
- `*_per_guide.csv`: per-guide ranking metrics.
- `evaluation_protocol.json`: evaluation settings and checkpoint hashes.

Training and evaluation are separate commands. The basic evaluation command does not run ablations or official-model comparisons.

## Use an existing data directory

Data do not need to be copied into this repository.

Set `--root` to the directory containing the upstream `data/` folder:

```bash
python -m pam2zero train --root PATH_TO_CRISPRPR --prior PATH_TO_CRISPRPR/models/D_models/crisot_score_param_new.npy --out results/final833_training --seeds 42 123 3407 2025 2026

python -m pam2zero evaluate --root PATH_TO_CRISPRPR --source-models results/final833_training --out results/final833_evaluation
```

Replace `PATH_TO_CRISPRPR` with the actual directory. Quote paths containing spaces.

Relative paths for `--prior`, `--out`, and `--source-models` are resolved against the current working directory, not against `--root`.

## Optional experiments

To reuse the five trained main models and run the paper-oriented ablation suite:

```bash
python -m pam2zero paper-suite --root . --source-models results/final833_training --out results/paper_suite
```

This is not required for basic training and evaluation. Optional external analyses require their corresponding input files.

## Troubleshooting

| Problem | Solution |
|---|---|
| `No module named pam2zero` | Run `python -m pip install -e .` from the project root using the same Python environment. |
| D matrix not found | Check `models/D_models/` or provide the actual absolute path using `--prior`. |
| PKL file not found | Check the six filenames and the directory supplied through `--root`. |
| PKL/NPY contains Git LFS text | Run `git lfs pull` in the upstream repository. |
| Five-seed checkpoints not found | Complete all five runs and pass their training output directory. |
| Output protocol mismatch | Use a new output directory when changing the experiment configuration or inputs. |

## Data attribution

Processed off-target datasets, the pair-code convention, and the initial D prior originate from CrisprPr. Please cite CrisprPr and the original experimental studies when using these resources.

Third-party datasets and pretrained checkpoints are not bundled with this repository. See `THIRD_PARTY.md` for source attribution.