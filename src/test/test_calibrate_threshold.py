import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from src.utils.calibrate_threshold import (
    calibrate_threshold,
    load_distances,
)


GENUINE_DISTANCES = [0.10, 0.20, 0.30]
IMPOSTOR_DISTANCES = [0.70, 0.80, 0.90]


class ThresholdCalibrationTests(unittest.TestCase):
    def test_target_far_finds_separating_threshold(self):
        result = calibrate_threshold(
            GENUINE_DISTANCES,
            IMPOSTOR_DISTANCES,
            method="far",
            target_far=0.0,
        )

        self.assertEqual(result["far"], 0.0)
        self.assertEqual(result["frr"], 0.0)
        self.assertGreater(result["threshold"], 0.30)
        self.assertLess(result["threshold"], 0.70)

    def test_eer_returns_threshold_and_error_rates(self):
        result = calibrate_threshold(
            GENUINE_DISTANCES,
            IMPOSTOR_DISTANCES,
            method="eer",
        )

        self.assertEqual(result["method"], "eer")
        self.assertAlmostEqual(result["far"], result["frr"])

    def test_youden_maximizes_separation(self):
        result = calibrate_threshold(
            GENUINE_DISTANCES,
            IMPOSTOR_DISTANCES,
            method="youden",
        )

        self.assertEqual(result["far"], 0.0)
        self.assertEqual(result["tpr"], 1.0)

    def test_cost_prioritizes_false_accepts(self):
        result = calibrate_threshold(
            genuine_distances=[0.20, 0.60],
            impostor_distances=[0.50, 0.80],
            method="cost",
            false_accept_cost=10.0,
            false_reject_cost=1.0,
        )

        self.assertEqual(result["far"], 0.0)

    def test_percentile_only_requires_genuine_distances(self):
        result = calibrate_threshold(
            genuine_distances=[0.10, 0.20, 0.30, 0.40],
            method="percentile",
            percentile=75.0,
        )

        expected = float(np.percentile([0.10, 0.20, 0.30, 0.40], 75))
        self.assertAlmostEqual(result["threshold"], expected)
        self.assertIsNone(result["far"])

    def test_method_with_impostors_rejects_missing_distances(self):
        with self.assertRaises(ValueError):
            calibrate_threshold(
                genuine_distances=GENUINE_DISTANCES,
                method="far",
            )

    def test_rejects_unknown_method(self):
        with self.assertRaises(ValueError):
            calibrate_threshold(
                GENUINE_DISTANCES,
                IMPOSTOR_DISTANCES,
                method="unknown",
            )

    def test_loads_distances_from_json(self):
        with TemporaryDirectory() as temporary_directory:
            input_path = Path(temporary_directory) / "distances.json"
            input_path.write_text(
                json.dumps(
                    {
                        "genuine_distances": GENUINE_DISTANCES,
                        "impostor_distances": IMPOSTOR_DISTANCES,
                    }
                ),
                encoding="utf-8",
            )

            genuine, impostor = load_distances(str(input_path))

        self.assertEqual(genuine, GENUINE_DISTANCES)
        self.assertEqual(impostor, IMPOSTOR_DISTANCES)


if __name__ == "__main__":
    unittest.main()
