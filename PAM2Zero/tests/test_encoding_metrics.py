import numpy as np
import unittest
from pam2zero.data.encoding import encode_pair, basepair_2_id_encode_original, normalize_to23, _off_target_mask
from pam2zero.evaluation.metrics import compute_metrics_paper, threshold_mcc, classification

class CoreTests(unittest.TestCase):

    def test_pair_order_and_padding(self):
        guide = 'G' * 20 + 'TGG'
        target = 'G' * 19 + 'A' + 'TGG'
        encoded = basepair_2_id_encode_original(guide, target)
        assert len(encoded) == 24 and encoded[0] == 31
        normalized, keep = normalize_to23(np.array([encoded]), 'example', record=False)
        assert keep.tolist() == [True]
        assert normalized[0].tolist() == list(encode_pair(guide, target))
        assert normalized[0, 19] == 8

    def test_n_replacement_is_homologous_encoding(self):
        assert encode_pair('NGG', 'AGG') == (1, 7, 7)
        assert encode_pair('AGG', 'NGG') == (1, 7, 7)

    def test_alignment_filters_unsupported_rows(self):
        x = np.full((3, 24), 1, dtype=np.uint8)
        x[:, 0] = 31
        x[1, 0] = 1
        x[2, 8] = 6
        result, keep = normalize_to23(x, 'example', record=False)
        assert result.shape == (1, 23)
        assert keep.tolist() == [True, False, False]
        with unittest.TestCase().assertRaises(ValueError):
            normalize_to23(x, 'example', record=False, policy='error')

    def test_on_target_filter_ignores_pam_differences(self):
        x = np.full((2, 24), 1, dtype=np.uint8)
        x[:, 0] = 31
        x[0, -1] = 2
        x[1, 10] = 2
        assert _off_target_mask(x).tolist() == [False, True]

    def test_macro_excludes_single_label_groups(self):
        summary, per = compute_metrics_paper([0, 1, 0, 0], [0, 1, 1, 0], ['a', 'a', 'b', 'b'])
        assert summary['Valid_sgRNA'] == 1
        assert summary['Macro_AUPRC'] == 1
        assert per.sgRNA.tolist() == ['a']

    def test_trapezoidal_auprc_is_not_ap(self):
        summary, _ = compute_metrics_paper([1, 0, 0, 0], [0, 0, 0, 0], ['a'] * 4)
        assert summary['Macro_AP'] == 0.25
        assert summary['Macro_AUPRC'] == 0.625

    def test_threshold_is_fit_once_on_validation(self):
        y = np.array([0, 0, 1, 1])
        threshold = threshold_mcc(y, np.array([0.0, 0.1, 0.8, 0.9]))
        assert classification(y, np.array([0.0, 0.1, 0.8, 0.9]), threshold)['MCC'] == 1
