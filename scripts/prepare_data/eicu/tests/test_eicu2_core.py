import unittest
from collections import Counter

from eicu2_core import (
    FORBIDDEN_STAY_FEATURES,
    assign_patient_stratified_splits,
    assert_feature_columns_safe,
    los_3class,
    max_split_ratio_error,
    within_feature_window,
)


class CoreRuleTests(unittest.TestCase):
    def test_los_three_class_boundaries(self):
        self.assertEqual(los_3class(0), 0)
        self.assertEqual(los_3class(3 * 1440 - 1), 0)
        self.assertEqual(los_3class(3 * 1440), 1)
        self.assertEqual(los_3class(7 * 1440 - 1), 1)
        self.assertEqual(los_3class(7 * 1440), 2)

    def test_feature_window_is_zero_to_24_hours_inclusive(self):
        self.assertFalse(within_feature_window(-1, 24))
        self.assertTrue(within_feature_window(0, 24))
        self.assertTrue(within_feature_window(1440, 24))
        self.assertFalse(within_feature_window(1441, 24))
        self.assertFalse(within_feature_window(None, 24))

    def test_all_discharge_fields_and_label_source_are_forbidden(self):
        expected = {
            "dischargeweight",
            "hospitaldischargeyear",
            "hospitaldischargetime24",
            "hospitaldischargeoffset",
            "hospitaldischargelocation",
            "hospitaldischargestatus",
            "unitdischargetime24",
            "unitdischargeoffset",
            "unitdischargelocation",
            "unitdischargestatus",
            "hospitaladmitoffset",
        }
        self.assertTrue(expected.issubset(FORBIDDEN_STAY_FEATURES))
        with self.assertRaisesRegex(ValueError, "forbidden"):
            assert_feature_columns_safe(["admissionweight", "unitdischargeoffset"])

    def test_patient_grouping_and_class_stratification(self):
        records = []
        for label in range(3):
            for patient_index in range(100):
                records.append(
                    {
                        "hospitalid": "73",
                        "uniquepid": f"p{label}_{patient_index}",
                        "patientunitstayid": f"s{label}_{patient_index}",
                        "icu_los_3class": label,
                    }
                )
        first = assign_patient_stratified_splits(records, 0.05, 0.15, 42)
        second = assign_patient_stratified_splits(records, 0.05, 0.15, 42)
        self.assertEqual(first, second)
        counts = {(split, label): 0 for split in ("train", "val", "test") for label in range(3)}
        for record in records:
            split = first[record["uniquepid"]]
            counts[(split, record["icu_los_3class"])] += 1
        for label in range(3):
            self.assertEqual(counts[("train", label)], 5)
            self.assertEqual(counts[("val", label)], 15)
            self.assertEqual(counts[("test", label)], 80)

    def test_repeated_stays_from_one_patient_never_cross_splits(self):
        records = [
            {
                "hospitalid": "73",
                "uniquepid": "same-patient",
                "patientunitstayid": "stay-a",
                "icu_los_3class": 0,
            },
            {
                "hospitalid": "73",
                "uniquepid": "same-patient",
                "patientunitstayid": "stay-b",
                "icu_los_3class": 2,
            },
        ]
        assignments = assign_patient_stratified_splits(records, 0.05, 0.15, 42)
        self.assertEqual(set(assignments), {"same-patient"})

    def test_mixed_label_patients_still_follow_stay_level_targets(self):
        records = []
        for patient_index in range(300):
            labels = [patient_index % 3]
            if patient_index % 10 == 0:
                labels.append((patient_index + 1) % 3)
            for stay_index, label in enumerate(labels):
                records.append(
                    {
                        "hospitalid": "73",
                        "uniquepid": f"p{patient_index}",
                        "patientunitstayid": f"s{patient_index}_{stay_index}",
                        "icu_los_3class": label,
                    }
                )
        assignments = assign_patient_stratified_splits(records, 0.05, 0.15, 42)
        counts = {(split, label): 0 for split in ("train", "val", "test") for label in range(3)}
        totals = Counter(record["icu_los_3class"] for record in records)
        for record in records:
            counts[(assignments[record["uniquepid"]], record["icu_los_3class"])] += 1
        targets = {"train": 0.05, "val": 0.15, "test": 0.80}
        for split, target in targets.items():
            for label in range(3):
                self.assertLessEqual(abs(counts[(split, label)] / totals[label] - target), 0.02)

    def test_split_ratio_error_uses_requested_targets(self):
        self.assertAlmostEqual(
            max_split_ratio_error({"train": 5, "val": 15, "test": 80}, 0.05, 0.15),
            0.0,
        )
        self.assertAlmostEqual(
            max_split_ratio_error({"train": 7, "val": 15, "test": 78}, 0.05, 0.15),
            0.02,
        )


if __name__ == "__main__":
    unittest.main()
