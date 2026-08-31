from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "04_daily_refresh_job.py"
SPEC = importlib.util.spec_from_file_location("daily_refresh_job", SCRIPT)
assert SPEC and SPEC.loader
daily_refresh_job = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(daily_refresh_job)


class DailyRefreshValidationTests(unittest.TestCase):
    def write_valid_outputs(self, output_dir: Path) -> None:
        pd.DataFrame(
            {
                "date": ["2026-08-24", "2026-08-25"],
                "category": ["Shoes", "Shoes"],
                "actual_units": [10, 12],
                "revenue": [1000, 1200],
                "online_model_forecast": [9, 11],
                "production_forecast": [9, 11],
                "production_model": ["online_sgd", "seasonal_naive_7d"],
                "any_anomaly": [False, True],
            }
        ).to_csv(output_dir / "fact_forecast_anomaly.csv", index=False)
        pd.DataFrame({"date": ["2026-08-24", "2026-08-25"]}).to_csv(
            output_dir / "dim_date.csv", index=False
        )
        pd.DataFrame({"category": ["Shoes"]}).to_csv(
            output_dir / "kpi_summary_by_category.csv", index=False
        )
        pd.DataFrame(
            {"model": ["online_sgd", "seasonal_naive_7d", "production_selection"]}
        ).to_csv(output_dir / "model_performance.csv", index=False)

    def test_valid_outputs_return_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output_dir = Path(raw)
            self.write_valid_outputs(output_dir)
            with patch.object(daily_refresh_job, "OUTPUT_DIR", output_dir):
                metrics = daily_refresh_job.validate_published_outputs()
        self.assertEqual(metrics["fact_rows"], 2)
        self.assertEqual(metrics["anomaly_rows"], 1)
        self.assertEqual(metrics["max_data_date"], "2026-08-25")

    def test_duplicate_fact_key_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output_dir = Path(raw)
            self.write_valid_outputs(output_dir)
            path = output_dir / "fact_forecast_anomaly.csv"
            fact = pd.read_csv(path)
            pd.concat([fact, fact.iloc[[0]]], ignore_index=True).to_csv(path, index=False)
            with patch.object(daily_refresh_job, "OUTPUT_DIR", output_dir):
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    daily_refresh_job.validate_published_outputs()

    def test_noncontinuous_date_dimension_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            output_dir = Path(raw)
            self.write_valid_outputs(output_dir)
            pd.DataFrame({"date": ["2026-08-24"]}).to_csv(
                output_dir / "dim_date.csv", index=False
            )
            with patch.object(daily_refresh_job, "OUTPUT_DIR", output_dir):
                with self.assertRaisesRegex(ValueError, "continuously"):
                    daily_refresh_job.validate_published_outputs()


if __name__ == "__main__":
    unittest.main()
