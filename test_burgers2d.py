import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from burgers2d_benchmark import (
    Problem, SlabMLP, exact_solution, handoff_target, periodic_losses,
    periodic_pairs, predict_piecewise, residual, sample_points, slab_index,
)
from pinn_model import MLP
from train_burgers2d_pinn import load_checkpoint, parse_args, run


class Constant(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, coords):
        return coords[:, :2] * 0 + self.value


class Burgers2DTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(123)
        self.problem = Problem()
        self.generator = torch.Generator().manual_seed(19)

    def points(self, n=96):
        return sample_points(n, 0, 1, "cpu", torch.float64, self.generator)

    def test_legacy_uniform_initialization_and_checkpoint(self):
        torch.manual_seed(9)
        old = MLP(in_dim=3, hidden_dim=12, hidden_layers=3, out_dim=2)
        torch.manual_seed(9)
        new = MLP(in_dim=3, out_dim=2, hidden_widths=[12, 12, 12])
        self.assertEqual(list(old.state_dict()), list(new.state_dict()))
        for key, value in old.state_dict().items():
            torch.testing.assert_close(value, new.state_dict()[key], rtol=0, atol=0)
        pts = self.points().float()
        torch.testing.assert_close(old(pts), new(pts), rtol=0, atol=0)

    def test_widths_and_parameter_count(self):
        model = MLP(in_dim=3, out_dim=2, hidden_widths=[256, 128, 64, 128, 256])
        self.assertEqual(sum(p.numel() for p in model.parameters()), 84034)
        self.assertEqual(model(torch.randn(7, 3)).shape, (7, 2))
        for widths in ([], [0], [-2], [1.5], [True]):
            with self.assertRaises(ValueError):
                MLP(hidden_widths=widths)

    def test_analytic_solution_satisfies_unforced_pde(self):
        for problem in [self.problem, Problem(nu=.05, amplitude=.5, speed_x=0, speed_y=-.2)]:
            r = residual(lambda x: exact_solution(x, problem), self.points(), problem)
            self.assertLess(r.abs().max().item(), 1e-10)

    def test_periodic_value_and_derivative(self):
        pairs = periodic_pairs(80, 0, 1, "cpu", torch.float64, self.generator)
        value, grad = periodic_losses(lambda x: exact_solution(x, self.problem), pairs)
        self.assertLess(value.item(), 1e-25)
        self.assertLess(grad.item(), 1e-24)

    def test_residual_matches_independent_polynomial(self):
        pts = self.points()
        def polynomial(z):
            t, x, y = z.split(1, 1)
            return torch.cat((t+x*x+y*y, t*t+x*y), 1)
        t, x, y = pts.split(1, 1)
        u, v = polynomial(pts).split(1, 1)
        expected = torch.cat((1+2*x*u+2*y*v-4*self.problem.nu, 2*t+y*u+x*v), 1)
        torch.testing.assert_close(residual(polynomial, pts, self.problem), expected)

    def test_slab_coordinate_gradient_and_pde_backprop(self):
        model = SlabMLP([8, 6], .4, .6).double()
        pts = self.points(8)
        pts[:, 0] = .45
        pts.requires_grad_(True)
        gradient = torch.autograd.grad(model(pts)[:, 0].sum(), pts)[0]
        h = 1e-6
        for axis in range(3):
            plus, minus = pts.detach().clone(), pts.detach().clone()
            plus[:, axis] += h
            minus[:, axis] -= h
            fd = (model(plus)[:, 0] - model(minus)[:, 0])/(2*h)
            torch.testing.assert_close(gradient[:, axis], fd, rtol=1e-5, atol=1e-7)
        residual(model, pts.detach(), self.problem).square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_handoff_never_uses_reference_after_first_slab(self):
        coords = self.points(4).requires_grad_(True)
        target = handoff_target(Constant(7), coords, self.problem)
        torch.testing.assert_close(target, torch.full_like(target, 7))
        self.assertFalse(target.requires_grad)
        torch.testing.assert_close(handoff_target(None, coords, self.problem), exact_solution(coords, self.problem))

    def test_piecewise_dispatch_endpoints(self):
        pts = torch.tensor([[0., .1, .2], [.49, .1, .2], [.5, .1, .2], [1., .1, .2]])
        values = predict_piecewise([Constant(2), Constant(3)], [0, .5, 1], pts)
        torch.testing.assert_close(values[:, 0], torch.tensor([2., 2., 3., 3.]))
        self.assertEqual(slab_index(.5, [0, .5, 1]), 1)
        self.assertEqual(slab_index(1, [0, .5, 1]), 1)
        with self.assertRaises(ValueError):
            slab_index(1.1, [0, .5, 1])

    def test_two_slab_training_output_and_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "run"
            args = parse_args(["--mode", "marching", "--windows", "2", "--epochs", "4",
                               "--hidden-widths", "8,8", "--n-interior", "8", "--n-initial", "8",
                               "--n-boundary", "4", "--eval-grid-size", "4", "--eval-times", "3",
                               "--eval-residual-points", "8", "--inference-repeats", "1", "--cpu-threads", "1",
                               "--runtime-sec", "30", "--device", "cpu", "--output-dir", str(out)])
            metrics = run(args)
            self.assertEqual(metrics["model_count"], 2)
            self.assertEqual(metrics["completed_epochs"], 4)
            import json
            slabs = json.loads((out / "slabs.json").read_text())
            self.assertEqual([s["budget_sec"] for s in slabs], [15, 15])
            self.assertEqual(slabs[1]["initial_target"], "previous_prediction")
            for name in ["fields.npz", "fields_u.png", "fields_v.png", "time_errors.csv", "interfaces.json", "metrics.json"]:
                self.assertTrue((out / name).exists())
            models = [load_checkpoint(out / f"slab_{i:02d}.pt") for i in range(2)]
            pts = self.points().float()
            self.assertTrue(torch.isfinite(predict_piecewise(models, [0, .5, 1], pts)).all())
            with self.assertRaises(FileExistsError):
                run(args)


if __name__ == "__main__":
    unittest.main()
