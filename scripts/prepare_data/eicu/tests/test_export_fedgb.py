import csv
import json
import tempfile
import unittest
from pathlib import Path

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from export_fedgb import export_fedgb
    from validate_dataset import validate_expected_contract


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


@unittest.skipIf(torch is None, "PyTorch/PyG export environment is unavailable")
class FedGBExportTests(unittest.TestCase):
    def test_exports_exact_release_contract_and_los_only_labels(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "intermediate"
            output = root / "output"
            source.mkdir()
            stays = [
                {"stay_nid": index, "patientunitstayid": 100 + index, "patient_nid": index,
                 "uniquepid": f"p{index}", "hospitalid": 73, "icu_los_3class": index % 3}
                for index in range(6)
            ]
            write_csv(source / "stays.csv", stays[0].keys(), stays)
            write_csv(source / "patients.csv", ["patient_nid", "uniquepid", "age"],
                      [{"patient_nid": index, "uniquepid": f"p{index}", "age": 50 + index} for index in range(6)])
            write_csv(source / "stay_features.csv", ["stay_nid", "patientunitstayid", "admissionweight"],
                      [{"stay_nid": index, "patientunitstayid": 100 + index, "admissionweight": 60 + index} for index in range(6)])
            write_csv(source / "clients.csv", ["client_id", "hospitalid", "n_stays", "n_patients", "los_class_0", "los_class_1", "los_class_2"],
                      [{"client_id": 0, "hospitalid": 73, "n_stays": 6, "n_patients": 6, "los_class_0": 2, "los_class_1": 2, "los_class_2": 2}])
            write_csv(source / "splits.csv", ["hospitalid", "stay_nid", "split"],
                      [{"hospitalid": 73, "stay_nid": index, "split": ("train", "val", "test", "test", "test", "test")[index]} for index in range(6)])
            write_csv(source / "edges_patient_stay.csv", ["patient_nid", "stay_nid"],
                      [{"patient_nid": index, "stay_nid": index} for index in range(6)])
            concepts = [{"treatment_nid": index, "concept_key": f"t:{index}", "treatmentstring": f"treatment {index}"} for index in range(4)]
            write_csv(source / "treatment_concepts.csv", concepts[0].keys(), concepts)
            write_csv(source / "medication_concepts.csv", ["medication_nid", "concept_key", "drughiclseqno", "drugname"], [])
            write_csv(source / "diagnosis_concepts.csv", ["diagnosis_nid", "concept_key", "icd9code", "diagnosisstring"], [])
            write_csv(source / "edges_stay_diagnosis.csv", ["stay_nid", "diagnosis_nid"], [])
            write_csv(source / "edges_stay_medication.csv", ["stay_nid", "medication_nid"], [])
            write_csv(source / "edges_stay_treatment.csv", ["stay_nid", "treatment_nid"],
                      [{"stay_nid": stay_id, "treatment_nid": concept_id} for stay_id in range(2) for concept_id in range(4)])
            (source / "schema.json").write_text(json.dumps({
                "schema_version": "eicu2-intermediate-1.0",
                "patient_feature_columns": ["age"],
                "stay_feature_columns": ["admissionweight"],
                "retained_clients": ["73"],
                "tasks": {"icu_los_3class": {"0": "a", "1": "b", "2": "c"}},
            }), encoding="utf-8")

            export_fedgb(source, output, hom_threshold=3)
            for variant, level in (("het", "hetero_subgraph"), ("hom", "homo_subgraph")):
                manifest = json.loads((output / variant / "fedgb_manifest.json").read_text())
                self.assertEqual(manifest["level"], level)
                self.assertEqual(manifest["num_clients"], 1)
                partition = output / variant / "distrib" / manifest["processed_partition"]
                data = torch.load(partition / "data_0.pt", map_location="cpu", weights_only=False)
                self.assertTrue(hasattr(data, "x"))
                self.assertTrue(hasattr(data, "edge_index"))
                self.assertTrue(hasattr(data, "y"))
                self.assertTrue(hasattr(data, "global_map"))
                self.assertFalse(hasattr(data, "mortality_3class"))
                split_root = partition / "node_cls" / "default_split"
                masks = [torch.load(split_root / f"{name}_0.pt", weights_only=False) for name in ("train", "val", "test")]
                self.assertFalse(torch.any(masks[0] & masks[1]))
                self.assertTrue(torch.all(masks[0] | masks[1] | masks[2] | (data.y < 0)))
            het = torch.load(output / "het" / "distrib" / "subgraph_fl_louvain_1_ACM_client_1" / "data_0.pt", weights_only=False)
            self.assertEqual(het.target_node_type, "stay")
            self.assertEqual(het.edge_type.numel(), het.edge_index.shape[1])
            hom = torch.load(output / "hom" / "distrib" / "subgraph_fl_louvain_1_ACM_client_1" / "data_0.pt", weights_only=False)
            self.assertEqual(hom.edge_index.shape[1], 2)
            self.assertEqual(set(hom.edge_index.flatten().tolist()), {0, 1})

    def test_expected_contract_rejects_drift(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            contract = Path(temp_dir) / "expected.json"
            contract.write_text(json.dumps({
                "num_clients": 40,
                "num_stays": 116383,
                "num_patients": 79294,
                "feature_dim": 315,
                "num_classes": 3,
                "num_tasks": 1,
                "class_counts": {"0": 86433, "1": 20628, "2": 9322},
                "split_counts": {"train": 5825, "val": 17455, "test": 93103},
                "heterogeneous": {
                    "num_unique_nodes": 202997,
                    "num_node_types": 5,
                    "num_directed_relation_types": 8,
                    "num_forward_semantic_edges": 6486909,
                },
                "homogeneous": {"num_nodes": 116383, "num_undirected_edges": 105210981},
            }), encoding="utf-8")
            report = {
                "aggregate": {
                    "num_clients": 39,
                    "num_stays": 116383,
                    "num_patients": 79294,
                    "feature_dim": 315,
                    "num_classes": 3,
                    "num_tasks": 1,
                    "class_counts": {"0": 86433, "1": 20628, "2": 9322},
                    "split_counts": {"train": 5825, "val": 17455, "test": 93103},
                    "heterogeneous": {
                        "num_unique_nodes": 202997,
                        "num_node_types": 5,
                        "num_directed_relation_types": 8,
                        "num_forward_semantic_edges": 6486909,
                    },
                    "homogeneous": {"num_nodes": 116383, "num_undirected_edges": 105210981},
                }
            }
            with self.assertRaisesRegex(ValueError, "num_clients"):
                validate_expected_contract(report, contract)


if __name__ == "__main__":
    unittest.main()
