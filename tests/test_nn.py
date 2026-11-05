"""Gradient check of the NumPy two-head network (finite differences) + basic sanity."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import unittest
import numpy as np
from p6.models import TwoHeadMLP, hour_feats


class TestGradients(unittest.TestCase):
    def test_backprop_matches_finite_differences(self):
        rng = np.random.default_rng(0)
        N, C = 7, 3
        net = TwoHeadMLP(N, C, [i % C for i in range(N)], emb=4, hid=6, use_hour=True, seed=1)
        place = rng.integers(0, N, 9); tgt = (place + rng.integers(1, N, 9)) % N
        hf = hour_feats(rng.integers(0, 24, 9))
        dplace = rng.integers(0, N, 8); dhf = hour_feats(rng.integers(0, 24, 8)); y = rng.normal(size=8)
        loss, g = net.loss_and_grads((place, hf, tgt), (dplace, dhf, y), drop=0.0)
        eps = 1e-6
        for k, v in net.p.items():
            flat = v.reshape(-1)
            for i in rng.choice(flat.size, min(5, flat.size), replace=False):
                old = flat[i]
                flat[i] = old + eps; lp, _ = net.loss_and_grads((place, hf, tgt), (dplace, dhf, y))
                flat[i] = old - eps; lm, _ = net.loss_and_grads((place, hf, tgt), (dplace, dhf, y))
                flat[i] = old
                num = (lp - lm) / (2 * eps)
                self.assertAlmostEqual(g[k].reshape(-1)[i], num, places=5, msg=f"{k}[{i}]")

    def test_current_place_has_zero_probability(self):
        net = TwoHeadMLP(5, 2, [0, 1, 0, 1, 0], emb=3, hid=4, seed=0)
        place = np.arange(5); h2, _ = net.forward(place, hour_feats(np.zeros(5, int)))
        lg = net.logits(h2, place)
        self.assertTrue(np.all(np.isinf(lg[np.arange(5), place])))


if __name__ == "__main__":
    unittest.main()
