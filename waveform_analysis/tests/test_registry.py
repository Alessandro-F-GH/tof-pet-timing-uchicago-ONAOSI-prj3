import unittest

from waveform_analysis.ml_pipeline.models import get_model, model_names


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

    def test_every_model_uses_training_derived_constant_sample_filter(self):
        for name in model_names():
            self.assertFalse(
                get_model(name).preserve_temporal_grid,
                f"{name} bypasses the constant-sample filter",
            )


if __name__ == "__main__":
    unittest.main()
