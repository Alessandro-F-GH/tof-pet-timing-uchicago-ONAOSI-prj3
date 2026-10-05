import unittest

from waveform_analysis.ml_pipeline.models import model_names


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


if __name__ == "__main__":
    unittest.main()
