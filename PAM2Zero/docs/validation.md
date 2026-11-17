# Validation status

The modular release was checked without retraining the paper models.

Completed in the preparation environment:

- Python compilation and static checks for unresolved global names.
- Eleven NumPy/data/metric tests.
- Frozen manuscript seed-42 coefficient identity and substitution checks.
- Probability-layer likelihood, positivity, resource-conservation and pool-invariance checks.
- Activity NumPy reference, component sums and train-only shape-scaling checks.
- Capacity audit reproduced 308 supported cells, additive design rank 35, 53 supported reversal pairs per seed and the archived residual fractions.

Five PyTorch test methods are included but were skipped because PyTorch is unavailable in the preparation environment. They cover parameter counts, gradients, NumPy parity, batch isolation, ignored first PAM position and OWN invariance to edge changes. Run them in the existing CRISPR_GNN environment before using this refactor for a full experiment.

The raw datasets, pretrained Torch checkpoints and GPU are not present here. Full dataset/checkpoint reproduction and new training were not executed. Paper-suite source import retains the TTISS reproduction check and never replaces failed inputs with historical aggregate results.
