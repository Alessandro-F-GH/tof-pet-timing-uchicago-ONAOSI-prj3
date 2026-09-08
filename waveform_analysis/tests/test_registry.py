import unittest

from waveform_analysis.ml_pipeline.models import ModelSpec, model_names, register_model, unregister_model


class RegistryTests(unittest.TestCase):
    def test_active_registry(self):
        self.assertEqual(set(model_names()), {"linear_svr", "pca_ridge", "cnn"})

    def test_dummy_model_can_register_without_study_change(self):
        dummy = ModelSpec(
            name="dummy",
            candidates=lambda _cfg: [{}],
            fit=lambda *args, **kwargs: object(),
            predict=lambda _artifact, pair: pair[:, 0, 0] * 0,
            save=lambda _artifact, _path: None,
        )
        register_model(dummy)
        self.assertIn("dummy", model_names())
        unregister_model("dummy")
        self.assertNotIn("dummy", model_names())
