# Data layout

`--root` points to the existing CrisprPr data project, not necessarily this repository.

```text
ROOT/
  data/
    train_data/
      CHANGEseq_encoded_pos_list.pkl
      CHANGEseq_encoded_labels.pkl
      CHANGEseq_encoded_sg_list.pkl
    val_data/
      ttiss_encoded_pos_list.pkl
      ttiss_encoded_labels.pkl
      ttiss_encoded_sg_list.pkl
    test_data/
      CRISPOR_test.csv
      GUIDEseq_test.csv
      HEK293T_test.csv
      II6_test.csv
      K562_test.csv
      guideseqCT_test.csv
      pdh_test.csv
      [optional TrueOT CSV or XLSX]
  models/
    D_models/crisot_score_param_new.npy
    PAM2Zero_TypedPairEdges_v1/baseline/
      seed_42/best.pt
      seed_123/best.pt
      seed_3407/best.pt
      seed_2025/best.pt
      seed_2026/best.pt
```

The `*_encoded.pkl` paths in configuration are bundle stems: the actual files are the three suffixed pickle files above. Position arrays may have an extra singleton dimension, which the reader removes while preserving row alignment.

The off-target input has 24 columns: left pad, guide20 and PAM3. Normalization removes only the left pad. The model ignores normalized PAM column 20 and reads columns 21 and 22. Unsupported gap/bulge rows are recorded by alignment audits.

The seven CSVs use the existing CrisprPr field order and encoding convention. TrueOT uses detected guide, target-DNA and binary-label columns; explicit column settings are available in `config.py` if needed. Its exclusions, duplicate keys and conflicting labels are saved in audit tables.

Activity data and shape files remain in the original CrisprFusion repository. `activity --repo` verifies that checkout against the archived upstream manifest. Do not place shape files into the off-target pickle directories.
