import unittest
import numpy as np
try:
    import torch
except ImportError:
    torch = None
if torch is not None:
    from pam2zero import config as cfg
    from pam2zero.models.factory import make_model
    from pam2zero.models.competition import typed_source_scores
    from pam2zero.evaluation.inference import own_logits
    from pam2zero.models.numpy_reference import numpy_reference

def make_fixture():
    cfg.initialize_device(cpu=True)
    prior = torch.zeros(24, 36)
    prior[1:21] = torch.linspace(-1, 1, 720).reshape(20, 36)
    cfg.FINAL_INITIAL_STATES = {42: dict(prior_template=prior, initial_d=prior[1:21].clone())}
    cfg.FINAL_SCALE = 0.5
    cfg.SOURCE_NUMERICS = 'typed_vector'
    return make_model(42, 'full')

@unittest.skipIf(torch is None, 'PyTorch is not installed')
class TorchModelTests(unittest.TestCase):

    def test_parameter_counts(self):
        initialized_model = make_fixture()
        for arm, count in [('full', 833), ('own', 795), ('frozen_D', 113), ('no_pam', 761), ('unit_edges', 814), ('constant_distance', 814), ('simple_additive', 793), ('pure_D', 792), ('factor_D', 170)]:
            model = make_model(42, arm)
            assert sum((value.numel() for value in model.parameters() if value.requires_grad)) == count

    def test_zero_pam_and_gradients(self):
        initialized_model = make_fixture()
        model = initialized_model
        assert torch.count_nonzero(model.initial_pam) == 0
        x = torch.tensor(np.random.default_rng(2).choice(cfg.ORDINARY_CODES, size=(5, 23)))
        model(x).square().sum().backward()
        for value in model.parameters():
            assert value.grad is not None and torch.isfinite(value.grad).all()

    def test_numpy_parity_and_batch_isolation(self):
        initialized_model = make_fixture()
        model = initialized_model.double()
        x = torch.tensor(np.random.default_rng(5).choice(cfg.ORDINARY_CODES, size=(7, 23)))
        state = {key: value.detach().numpy() for key, value in model.state_dict().items()}
        expected = numpy_reference(state, x.numpy())
        actual = typed_source_scores(model, x)
        for branch in ['own', 'full']:
            np.testing.assert_allclose(actual[branch].detach().numpy(), expected[branch], rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(model(x), torch.cat([model(row[None]) for row in x]), rtol=1e-12, atol=1e-12)

    def test_pam_first_ignored(self):
        initialized_model = make_fixture()
        model = initialized_model
        x = torch.ones(3, 23, dtype=torch.long)
        x[1, 6] = 2
        changed = x.clone()
        changed[:, 20] = 7
        torch.testing.assert_close(model(x), model(changed), rtol=0, atol=0)

    def test_edges_do_not_enter_own(self):
        initialized_model = make_fixture()
        model = initialized_model
        x = torch.tensor(np.random.default_rng(13).choice(cfg.ORDINARY_CODES, size=(4, 23)))
        before = own_logits(model, x).detach().clone()
        with torch.no_grad():
            model.edge_raw.add_(1)
            model.raw_distance.add_(2)
        torch.testing.assert_close(own_logits(model, x), before, rtol=0, atol=0)
