import tempfile
import unittest
from pathlib import Path

import torch

from burgers2d_benchmark import Problem, exact_solution
from run_field_bottleneck_comparison import architecture, errors, exact_fields, make_data, make_model, parse_args, run


class FieldBottleneckTests(unittest.TestCase):
    def test_shapes_and_depth(self):
        for grid in [8, 16, 64]:
            d = 2 * grid * grid
            bottom = 16 if grid == 8 else 128
            a = architecture(d, bottom, "abrupt")
            b = architecture(d, bottom, "gradual")
            self.assertEqual(len(a), len(b))
            self.assertEqual(min(a), min(b))
            self.assertEqual(b, list(reversed(b)))
        self.assertEqual(architecture(512, 128, "gradual"), [512, 256, 128, 256, 512])
        with self.assertRaises(ValueError):
            architecture(500, 128, "gradual")
        model = make_model(architecture(128, 16, "gradual"))
        output = model(torch.randn(3, 128))
        self.assertEqual(output.shape, (3, 128))
        output.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_family_matches_existing_reference(self):
        params = torch.tensor([[.8, .12, .34]], dtype=torch.float64)
        axis = torch.arange(8, dtype=torch.float64) / 8
        x, y = torch.meshgrid(axis, axis, indexing="ij")
        coords = torch.stack([torch.full_like(x, .25), x - .12, y - .34], -1).reshape(-1, 3)
        expected = exact_solution(coords, Problem()).T.flatten()
        torch.testing.assert_close(exact_fields(params, 8, .25)[0], expected)

    def test_split_reproducibility_and_errors(self):
        params, pairs, bg = make_data(8, [7, 3, 4], .25, 1)
        self.assertEqual(len(params.unique(dim=0)), 14)
        torch.testing.assert_close(params, make_data(8, [7, 3, 4], .25, 1)[0])
        self.assertEqual([len(x) for x, y in pairs], [7, 3, 4])
        self.assertEqual(errors(pairs[2][1], pairs[2][1], bg)["l2_relative_error"], 0)
        self.assertAlmostEqual(errors(torch.zeros_like(pairs[2][1]), pairs[2][1], bg)["fluctuation_l2_relative_error"], 1)

    def test_training_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "run"
            args = parse_args(["--grid", "8", "--bottleneck", "16", "--train-fields", "12",
                               "--validation-fields", "4", "--test-fields", "4", "--max-updates", "3",
                               "--device", "cpu", "--output-root", str(root)])
            rows = run(args)
            self.assertEqual([r["updates"] for r in rows], [3, 3])
            checkpoint = torch.load(root / "gradual/latest.pt", weights_only=False)
            model = make_model(checkpoint["widths"])
            model.load_state_dict(checkpoint["model"])
            self.assertTrue((root / "summary.csv").exists())
            with self.assertRaises(FileExistsError):
                run(args)


if __name__ == "__main__":
    unittest.main()
