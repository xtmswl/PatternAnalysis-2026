"""Numerical baseline checks; passing is not segmentation-performance evidence."""

import unittest

import torch
import torch.nn.functional as F

from modules import UNet2D, SoftDiceLoss


class BaselineTests(unittest.TestCase):
    def test_expected_shape_and_finite_gradients(self):
        torch.manual_seed(42)
        model = UNet2D()
        logits = model(torch.randn(2, 1, 32, 48))
        self.assertEqual(tuple(logits.shape), (2, 6, 32, 48))
        labels = torch.randint(0, 6, (2, 32, 48))
        target = F.one_hot(labels, 6).permute(0, 3, 1, 2).float()
        loss = SoftDiceLoss()(logits, target)
        loss.backward()
        self.assertTrue(0 < loss.item() < 1)
        for p in model.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
        self.assertGreater(model.head.weight.grad.abs().sum().item(), 0)

    def test_dice_against_hand_calculation(self):
        # Uniform p=.5, equal class support: each Dice=.5 (up to smoothing).
        logits = torch.zeros(1, 2, 2, 2, requires_grad=True)
        labels = torch.tensor([[[0, 1], [0, 1]]])
        target = F.one_hot(labels, 2).permute(0, 3, 1, 2).float()
        self.assertAlmostEqual(SoftDiceLoss()(logits, target).item(), .5, places=5)
        correct = (target * 2 - 1) * 20
        wrong = -correct
        self.assertLess(SoftDiceLoss()(correct, target).item(), 1e-5)
        self.assertGreater(SoftDiceLoss()(wrong, target).item(), .999)

    def test_empty_foreground_and_3d_targets_stay_finite(self):
        logits = torch.randn(1, 6, 4, 8, 8, requires_grad=True)
        target = F.one_hot(torch.zeros(1, 4, 8, 8, dtype=torch.long), 6).permute(0, 4, 1, 2, 3).float()
        loss = SoftDiceLoss(include_background=False)(logits, target)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_shape_errors_fail_clearly(self):
        with self.assertRaises(ValueError): UNet2D()(torch.zeros(1, 1, 31, 32))
        with self.assertRaises(ValueError): SoftDiceLoss()(torch.zeros(1, 6, 2, 2), torch.zeros(1, 1, 2, 2))


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
