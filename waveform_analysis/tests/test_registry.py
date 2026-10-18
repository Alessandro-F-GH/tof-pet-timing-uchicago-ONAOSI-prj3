import unittest

from waveform_analysis.models import model_names


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
                "direct_multirocket",
                "shared_multirocket",
                "direct_multirocket_rdst",
                "shared_multirocket_rdst",
                "direct_hydra",
                "shared_hydra",
                "direct_rdst",
                "shared_rdst",
            },
        )



if __name__ == "__main__":
    unittest.main()
