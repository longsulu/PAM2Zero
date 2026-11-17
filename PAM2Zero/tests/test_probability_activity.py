import unittest
import numpy as np
from pam2zero.probability.aligned import coefficient_self_test, probability_self_test
from pam2zero.activity.validation import self_test as activity_self_test
from pam2zero.models.numpy_reference import numpy_reference
from pam2zero import config as cfg

class CoreTests(unittest.TestCase):

    def test_frozen_checkpoint_coefficients(self):
        coefficient_self_test()

    def test_probability_conservation_and_pool_invariance(self):
        probability_self_test()

    def test_activity_reference_and_training_only_scaler(self):
        activity_self_test()

    def test_numpy_own_is_additive_and_ignores_first_pam(self):
        rng = np.random.default_rng(21)
        state = dict(matrix_d=rng.normal(size=(20, 36)), matrix_pam=rng.normal(size=(2, 36)), raw_team_gain=np.zeros(2), raw_temperature=np.array(0.4), raw_distance=np.full(19, -2.0), edge_raw=rng.normal(size=19), input_scale=np.array(0.5))
        x = np.ones((4, 23), dtype=int)
        x[1, 5] = 2
        x[2, 12] = 3
        x[3, 5], x[3, 12] = (2, 3)
        own = numpy_reference(state, x)['own']
        assert abs(own[3] - own[1] - own[2] + own[0]) < 1e-12
        changed = x.copy()
        changed[:, 20] = 7
        np.testing.assert_array_equal(numpy_reference(state, changed)['own'], own)
        off = dict(state, edge_raw=np.zeros(19), raw_distance=np.full(19, -15.0))
        np.testing.assert_array_equal(numpy_reference(off, x)['own'], own)
