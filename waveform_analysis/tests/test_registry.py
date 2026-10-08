import unittest

from waveform_analysis.models import get_model, model_names


class RegistryTests(unittest.TestCase):
    def test_active_registry(self):
        self.assertEqual(
            set(model_names()),
            {
                "antisymmetric_mlp",
                "locally_connected_mlp",
                "shared_cnn1d",
                "shared_linear_ridge",
                "shared_minirocket",
                "direct_linear_ridge",
                "direct_mlp",
                "independent_cnn1d",
                "onishi_cnn",
                "direct_minirocket",
            },
        )

    def test_architecture_specific_temporal_grid_policy(self):
        preserving = {"shared_cnn1d", "independent_cnn1d", "locally_connected_mlp"}
        for name in model_names():
            self.assertEqual(
                get_model(name).preserve_temporal_grid, name in preserving, name
            )


if __name__ == "__main__":
    unittest.main()
