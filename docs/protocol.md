# Evaluation protocol

- Main model: guide20 plus the final two PAM positions, zero-initialized trainable PAM.
- Seeds: 42, 123, 3407, 2025 and 2026.
- CHANGE-seq: retain every supported off-target training example. The inherited internal split is recombined and only determines historical row order.
- Training group weighting: inverse-square-root group size, clipped to [0.25, 4.0], then normalized by the sample-weighted mean. This is preserved from the final archived protocol.
- Loss: 0.25 times the focal modulation `(1 - exp(-CE))**2 * CE`, multiplied by group weight. Test metrics are unweighted.
- AdamW: D/PAM and other learning rates 0.001, weight decay 0, batch size 4096, gradient clipping 1, at most 100 epochs.
- TTISS FULL macro AUPRC: model selection and ReduceLROnPlateau, factor 0.5, patience 5, minimum learning rate 1e-6. Early stopping patience 12.
- Primary deployment readout: OWN from the FULL-selected checkpoint. FULL is a secondary readout.
- Grouping: training guide suffix rule, original TTISS identifiers, and test guide prefixes with the final three bases removed.
- Macro AUPRC: per valid guide, trapezoidal integration after setting precision to 1 where recall equals 0. Macro AP is reported separately. Historical `Global_AUPRC` is an AP-valued field; the paper suite also exports `Global_AP`.
- Only guides containing both labels enter macro ranking metrics. Mean7 averages the seven dataset values equally.
- Classification threshold: maximize pooled TTISS MCC, then keep it fixed on tests.
- TrueOT: full-input deduplication, all conflicting label keys excluded, nested distant-guide cohorts. Duplicate CSV/XLSX copies count once.
- Read-only perturbations and independently retrained ablations are distinct analyses.
- The probability extension uses a frozen sequence score and leave-one-guide-out fitting of observation layers. This is separate from neural training.
- Activity regression uses the upstream train/test splits, five-fold test-prediction averaging and five repeated seeds.

The source release supports both the typed-vector and legacy-logit arithmetic orders. Float32 ties can affect ranking. Checkpoint reuse selects its recorded architecture-compatible mode and checks saved TTISS metrics before importing the main model.
