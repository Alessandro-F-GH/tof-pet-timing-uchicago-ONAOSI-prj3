import json
import tempfile
import unittest
from pathlib import Path

from waveform_analysis.ml_pipeline.latex_tables import (
    make_dataset_latex_table,
    selection_dataset_rows,
)


class DatasetLatexTableTests(unittest.TestCase):
    def test_reads_existing_selection_cache_without_preprocessing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            selection_dir = root / "selection"
            data_dir.mkdir()
            source = data_dir / "46V-440mV.root"
            source.write_bytes(b"")

            cache = selection_dir / source.stem
            cache.mkdir(parents=True)
            (cache / "manifest.json").write_text(
                json.dumps(
                    {
                        "n_raw": 1000,
                        "n_selected": 700,
                    }
                ),
                encoding="utf-8",
            )
            (cache / "selection_summary.csv").write_text(
                "criterion,split,remaining,rejected_from_previous\n"
                "finite_energy_amplitude,development,590,10\n"
                "finite_energy_amplitude,test,390,10\n"
                "photopeak,development,480,110\n"
                "photopeak,test,320,70\n"
                "main_hit,development,430,50\n"
                "main_hit,test,290,30\n"
                "baseline_noise,development,420,10\n"
                "baseline_noise,test,280,10\n",
                encoding="utf-8",
            )

            config = {
                "data": {
                    "root_folder": str(data_dir),
                    "root_glob": "*.root",
                    "recursive": False,
                },
                "preprocessing": {
                    "selection_store_dir": str(selection_dir),
                },
                "experiment": {},
            }

            rows = selection_dataset_rows(config)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["voltage_V"], 46.0)
            self.assertEqual(rows[0]["collected_events"], 1000)
            self.assertEqual(rows[0]["photopeak_events"], 800)
            self.assertEqual(rows[0]["selected_events"], 700)

            output = root / "dataset.tex"
            make_dataset_latex_table(
                config,
                output,
                caption="Dataset summary.",
                label="tab:dataset",
            )
            rendered = output.read_text(encoding="utf-8")
            self.assertIn("46.0 & 1000 & 800 & 700", rendered)
            self.assertIn("\\label{tab:dataset}", rendered)

    def test_missing_cache_does_not_trigger_rebuild(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data_dir = root / "data"
            data_dir.mkdir()
            (data_dir / "47V-470mV.root").write_bytes(b"")

            config = {
                "data": {
                    "root_folder": str(data_dir),
                    "root_glob": "*.root",
                    "recursive": False,
                },
                "preprocessing": {
                    "selection_store_dir": str(root / "selection"),
                },
                "experiment": {},
            }

            with self.assertRaisesRegex(
                FileNotFoundError,
                "never rebuilds preprocessing",
            ):
                selection_dataset_rows(config)


if __name__ == "__main__":
    unittest.main()
