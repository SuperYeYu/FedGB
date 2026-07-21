import csv
import json
import tempfile
import unittest
from pathlib import Path

from build_intermediate import BuildConfig, build_intermediate


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class IntermediateBuildTests(unittest.TestCase):
    def test_build_removes_leakage_and_keeps_full_stay_relations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw = root / "raw"
            out = root / "out"
            raw.mkdir()
            patient_fields = [
                "patientunitstayid", "uniquepid", "hospitalid", "gender", "age",
                "ethnicity", "admissionheight", "admissionweight", "dischargeweight",
                "unitvisitnumber", "hospitaladmitoffset", "hospitaladmitsource", "unittype",
                "unitadmitsource", "unitstaytype", "unitdischargeoffset",
                "unitdischargestatus", "unitdischargelocation", "hospitaldischargestatus",
                "hospitaldischargeoffset",
            ]
            write_csv(
                raw / "patient.csv",
                patient_fields,
                [
                    {
                        "patientunitstayid": "10", "uniquepid": "p1", "hospitalid": "73",
                        "gender": "Female", "age": "70", "ethnicity": "Caucasian",
                        "admissionheight": "165", "admissionweight": "60", "dischargeweight": "999",
                        "unitvisitnumber": "1", "hospitaladmitoffset": "-120",
                        "hospitaladmitsource": "Emergency Department", "unittype": "MICU",
                        "unitadmitsource": "Emergency Department", "unitstaytype": "admit",
                        "unitdischargeoffset": str(4 * 1440), "unitdischargestatus": "Alive",
                        "unitdischargelocation": "Floor", "hospitaldischargestatus": "Alive",
                        "hospitaldischargeoffset": str(9 * 1440),
                    }
                ],
            )
            write_csv(raw / "apacheApsVar.csv", ["apacheapsvarid", "patientunitstayid", "glucose"], [{"apacheapsvarid": "1", "patientunitstayid": "10", "glucose": "100"}])
            write_csv(raw / "pastHistory.csv", ["patientunitstayid", "pasthistorypath", "pasthistoryvalue", "pasthistoryvaluetext"], [])
            write_csv(raw / "allergy.csv", ["patientunitstayid", "allergyname", "drugname", "allergytype"], [])
            write_csv(
                raw / "lab.csv",
                ["patientunitstayid", "labresultoffset", "labname", "labresult"],
                [
                    {"patientunitstayid": "10", "labresultoffset": "60", "labname": "glucose", "labresult": "111"},
                    {"patientunitstayid": "10", "labresultoffset": "1500", "labname": "glucose", "labresult": "9999"},
                ],
            )
            write_csv(
                raw / "vitalPeriodic.csv",
                ["patientunitstayid", "observationoffset", "temperature", "sao2", "heartrate", "respiration", "systemicsystolic", "systemicdiastolic", "systemicmean"],
                [
                    {"patientunitstayid": "10", "observationoffset": "30", "temperature": "36", "heartrate": "80"},
                    {"patientunitstayid": "10", "observationoffset": "1501", "temperature": "99", "heartrate": "999"},
                ],
            )
            write_csv(raw / "diagnosis.csv", ["patientunitstayid", "diagnosisoffset", "diagnosisstring", "icd9code", "diagnosispriority", "activeupondischarge"], [{"patientunitstayid": "10", "diagnosisoffset": "2000", "diagnosisstring": "late diagnosis", "icd9code": "123", "activeupondischarge": "true"}])
            write_csv(raw / "treatment.csv", ["patientunitstayid", "treatmentoffset", "treatmentstring", "activeupondischarge"], [{"patientunitstayid": "10", "treatmentoffset": "2000", "treatmentstring": "late treatment", "activeupondischarge": "true"}])
            write_csv(raw / "medication.csv", ["patientunitstayid", "drugstartoffset", "drugordercancelled", "drugname", "drughiclseqno"], [{"patientunitstayid": "10", "drugstartoffset": "2000", "drugordercancelled": "No", "drugname": "late medication", "drughiclseqno": "99"}])

            build_intermediate(
                raw,
                out,
                BuildConfig(retained_clients=("73",), min_client_patients=1, top_labs=1, top_history=0, top_allergy=0),
            )

            schema = json.loads((out / "schema.json").read_text(encoding="utf-8"))
            self.assertEqual(schema["tasks"], {"icu_los_3class": {"0": "ICU LOS < 3 days", "1": "3 <= ICU LOS < 7 days", "2": "ICU LOS >= 7 days"}})
            columns = schema["stay_feature_columns"]
            self.assertFalse(any("discharge" in name for name in columns))
            self.assertNotIn("hospitaladmitoffset", columns)

            with (out / "stay_features.csv").open(newline="", encoding="utf-8") as handle:
                stay = next(csv.DictReader(handle))
            self.assertEqual(stay["lab_glucose_mean"], "111")
            self.assertEqual(stay["vital_temperature_mean"], "36")

            with (out / "stays.csv").open(newline="", encoding="utf-8") as handle:
                label_row = next(csv.DictReader(handle))
            self.assertEqual(label_row["icu_los_3class"], "1")
            self.assertNotIn("icu_los_days", label_row)
            self.assertFalse(any("mortality" in name for name in label_row))

            for filename in ("edges_stay_diagnosis.csv", "edges_stay_treatment.csv", "edges_stay_medication.csv"):
                with (out / filename).open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), 1, filename)
                self.assertFalse(any("discharge" in name for name in rows[0]), filename)


if __name__ == "__main__":
    unittest.main()
