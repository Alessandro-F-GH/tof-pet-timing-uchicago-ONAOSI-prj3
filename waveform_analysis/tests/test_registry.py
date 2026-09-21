import unittest

from waveform_analysis.ml_pipeline.models import model_names


class RegistryTests(unittest.TestCase):
    def test_active_registry(self):
        self.assertEqual(set(model_names()), {"mlp", "onishi_cnn"})


if __name__ == "__main__":
    unittest.main()
