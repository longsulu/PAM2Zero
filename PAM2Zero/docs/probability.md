# Probability inputs

Required inputs are already processed local sequence tables. No network access or coordinate reconstruction is performed.

The coordinate-v2 ZIP must contain:

- `LOCK_BEFORE_EXTERNAL_SCORING.json`
- `audit.json`
- `prepared_sequences.csv`
- `nominated_sites_predictions.csv`
- `hybrid_capture_predictions.csv`
- `rhAmpSeq_secondary_predictions.csv`

The lock must identify coordinate normalization version 2. Candidate tables require `key`, `guide`, `guide23`, `DNA` and their endpoint-specific labels. The nominated-site table requires `mismatches` and `GUIDE-seq_detected_1`, `_2`, `_3`. The on-target CSV requires `guide23`, `DNA`, `chromosome`, `start`, `end`.

`src/pam2zero/resources/seed42_own_coefficients.json` stores only exported scores and the source checkpoint identity. It is not a PyTorch checkpoint. The fixed source SHA256 is `c8ee012523c64b52cb7197f9ba4c1e4fbf911dfccf8271627885f5030d9cc449`.

Calibration, finite allocation and occupancy preserve within-guide ordering. Kinetic and seed-feature controls may reorder sites. Brier/log loss and ranking metrics answer different questions. Candidate-pool sensitivity and convergence diagnostics are retained in output tables.
