#!/usr/bin/env python3
"""Build auditable eICU_2 CSV intermediates from licensed eICU-CRD tables."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from eicu2_core import (
    assert_feature_columns_safe,
    los_3class,
    parse_float,
    split_records_by_hospital,
    within_feature_window,
)


DEFAULT_RETAINED_CLIENTS = (
    "73", "79", "110", "122", "141", "148", "152", "157", "165", "167",
    "171", "176", "183", "188", "197", "199", "208", "227", "243", "248",
    "252", "264", "281", "283", "300", "307", "331", "338", "345", "394",
    "400", "411", "413", "416", "417", "420", "435", "443", "449", "458",
)
REQUIRED_TABLES = (
    "patient.csv", "apacheApsVar.csv", "pastHistory.csv", "allergy.csv", "lab.csv",
    "vitalPeriodic.csv", "diagnosis.csv", "treatment.csv", "medication.csv",
)
VITAL_COLUMNS = (
    "temperature", "sao2", "heartrate", "respiration", "systemicsystolic",
    "systemicdiastolic", "systemicmean",
)
APACHE_SKIP = {"apacheapsvarid", "patientunitstayid"}
MISSING = ""


@dataclass(frozen=True)
class BuildConfig:
    retained_clients: Tuple[str, ...] = DEFAULT_RETAINED_CLIENTS
    min_client_patients: int = 1000
    time_window_hours: float = 24.0
    top_labs: int = 50
    top_history: int = 100
    top_allergy: int = 50
    train_fraction: float = 0.05
    val_fraction: float = 0.15
    seed: int = 42


class Vocab:
    def __init__(self) -> None:
        self.values = ["<missing>"]
        self.lookup = {"<missing>": 0}

    def add(self, value: object) -> int:
        key = normalize_text(value) or "<missing>"
        if key not in self.lookup:
            self.lookup[key] = len(self.values)
            self.values.append(key)
        return self.lookup[key]


class NumericStats:
    __slots__ = ("count", "total", "minimum", "maximum", "last_offset", "last_value")

    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.minimum = None
        self.maximum = None
        self.last_offset = None
        self.last_value = None

    def update(self, value: float, offset: Optional[float] = None) -> None:
        self.count += 1
        self.total += value
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)
        if offset is None or self.last_offset is None or offset >= self.last_offset:
            self.last_offset = offset
            self.last_value = value

    def values(self) -> Tuple[str, str, str, str, str]:
        if not self.count:
            return "0", MISSING, MISSING, MISSING, MISSING
        return (
            str(self.count),
            format_number(self.total / self.count),
            format_number(self.minimum),
            format_number(self.maximum),
            format_number(self.last_value),
        )


def log(message: str) -> None:
    print(f"[eicu2] {message}", file=sys.stderr, flush=True)


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip()).lower()


def safe_name(value: object) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", normalize_text(value)).strip("_")
    return normalized[:80] or "missing"


def format_number(value: Optional[float]) -> str:
    if value is None or not math.isfinite(value):
        return MISSING
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.8g}"


def open_rows(path: Path) -> Iterable[dict]:
    with path.open(newline="", encoding="utf-8", errors="replace") as handle:
        yield from csv.DictReader(handle)


def write_rows(path: Path, fields: Sequence[str], rows: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def most_common(values: Counter) -> str:
    if not values:
        return ""
    return min(values.items(), key=lambda item: (-item[1], normalize_text(item[0])))[0]


def diagnosis_key(row: dict) -> Tuple[str, str, str]:
    code = normalize_text(row.get("icd9code"))
    label = normalize_text(row.get("diagnosisstring"))
    if code:
        codes = [part.strip() for part in re.split(r"[,;]", code) if part.strip()]
        return "icd9:" + "|".join(codes), row.get("icd9code", ""), row.get("diagnosisstring", "")
    return "diagnosis:" + label, row.get("icd9code", ""), row.get("diagnosisstring", "")


def treatment_key(row: dict) -> Tuple[str, str]:
    label = normalize_text(row.get("treatmentstring"))
    return "treatment:" + label, row.get("treatmentstring", "")


def medication_key(row: dict) -> Tuple[str, str, str]:
    hicl = normalize_text(row.get("drughiclseqno"))
    label = normalize_text(row.get("drugname"))
    if hicl and hicl not in {"0", "nan"}:
        return "hicl:" + hicl, row.get("drughiclseqno", ""), row.get("drugname", "")
    return "drug:" + label, row.get("drughiclseqno", ""), row.get("drugname", "")


def _validate_inputs(raw_root: Path) -> None:
    missing = [name for name in REQUIRED_TABLES if not (raw_root / name).is_file()]
    if missing:
        raise FileNotFoundError("missing required eICU tables: " + ", ".join(missing))


def _read_stays(raw_root: Path, config: BuildConfig, vocabs: Dict[str, Vocab]):
    requested = set(config.retained_clients)
    stays = []
    patient_aggs = defaultdict(
        lambda: {
            "age": NumericStats(), "height": NumericStats(), "gender": Counter(),
            "ethnicity": Counter(), "history": set(), "allergy": set(),
        }
    )
    client_patients = defaultdict(set)
    for source in open_rows(raw_root / "patient.csv"):
        hospital_id = str(source.get("hospitalid", ""))
        if hospital_id not in requested:
            continue
        try:
            label = los_3class(source.get("unitdischargeoffset"))
        except ValueError:
            continue
        stay_id = str(source["patientunitstayid"])
        patient_id = str(source.get("uniquepid") or f"missing_patient:{stay_id}")
        aggregate = patient_aggs[patient_id]
        age = parse_float(source.get("age"))
        height = parse_float(source.get("admissionheight"))
        if age is not None:
            aggregate["age"].update(age)
        if height is not None:
            aggregate["height"].update(height)
        aggregate["gender"][source.get("gender", "")] += 1
        aggregate["ethnicity"][source.get("ethnicity", "")] += 1
        stays.append(
            {
                "patientunitstayid": stay_id,
                "uniquepid": patient_id,
                "hospitalid": hospital_id,
                "icu_los_3class": label,
                "admissionweight": format_number(parse_float(source.get("admissionweight"))),
                "unitvisitnumber": format_number(parse_float(source.get("unitvisitnumber"))),
                "hospitaladmitsource_id": vocabs["hospitaladmitsource"].add(source.get("hospitaladmitsource")),
                "unittype_id": vocabs["unittype"].add(source.get("unittype")),
                "unitadmitsource_id": vocabs["unitadmitsource"].add(source.get("unitadmitsource")),
                "unitstaytype_id": vocabs["unitstaytype"].add(source.get("unitstaytype")),
            }
        )
        client_patients[hospital_id].add(patient_id)

    found = set(client_patients)
    missing_clients = sorted(requested - found, key=int)
    if missing_clients:
        raise ValueError("retained clients absent from patient.csv: " + ", ".join(missing_clients))
    too_small = {
        hospital_id: len(client_patients[hospital_id])
        for hospital_id in requested
        if len(client_patients[hospital_id]) < config.min_client_patients
    }
    if too_small:
        raise ValueError(f"retained clients below min_client_patients={config.min_client_patients}: {too_small}")

    stays.sort(key=lambda row: (int(row["hospitalid"]), int(row["patientunitstayid"])))
    for index, stay in enumerate(stays):
        stay["stay_nid"] = index
    patients = sorted({stay["uniquepid"] for stay in stays})
    patient_nids = {patient_id: index for index, patient_id in enumerate(patients)}
    for stay in stays:
        stay["patient_nid"] = patient_nids[stay["uniquepid"]]
    stay_nids = {stay["patientunitstayid"]: stay["stay_nid"] for stay in stays}
    return stays, patient_nids, stay_nids, patient_aggs, client_patients


def _ingest_patient_concepts(
    raw_root: Path,
    stay_to_patient: Dict[str, str],
    patient_aggs,
    time_window_hours: float,
) -> None:
    specs = (
        (
            "pastHistory.csv",
            "history",
            "pasthistoryoffset",
            ("pasthistorypath", "pasthistoryvalue", "pasthistoryvaluetext"),
        ),
        ("allergy.csv", "allergy", "allergyoffset", ("allergyname", "drugname", "allergytype")),
    )
    for filename, key, offset_field, fields in specs:
        for row in open_rows(raw_root / filename):
            patient_id = stay_to_patient.get(str(row.get("patientunitstayid", "")))
            if patient_id is None:
                continue
            if row.get(offset_field) not in (None, "") and not within_feature_window(
                row.get(offset_field), time_window_hours
            ):
                continue
            concept = next((normalize_text(row.get(field)) for field in fields if normalize_text(row.get(field))), "")
            if concept:
                patient_aggs[patient_id][key].add(concept)


def _top_patient_concepts(patient_aggs, key: str, limit: int) -> List[str]:
    counts = Counter()
    for aggregate in patient_aggs.values():
        counts.update(aggregate[key])
    return [value for value, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]]


def _ingest_apache(raw_root: Path, stay_nids: Dict[str, int]):
    features = {}
    columns = set()
    for row in open_rows(raw_root / "apacheApsVar.csv"):
        stay_id = str(row.get("patientunitstayid", ""))
        if stay_id not in stay_nids:
            continue
        values = {}
        for name, raw_value in row.items():
            if name.lower() in APACHE_SKIP:
                continue
            column = "apache_" + safe_name(name)
            columns.add(column)
            values[column] = format_number(parse_float(raw_value))
        features[stay_id] = values
    return features, sorted(columns)


def _top_labs(raw_root: Path, stay_nids: Dict[str, int], config: BuildConfig) -> List[str]:
    counts = Counter()
    for row in open_rows(raw_root / "lab.csv"):
        if str(row.get("patientunitstayid", "")) not in stay_nids:
            continue
        if not within_feature_window(row.get("labresultoffset"), config.time_window_hours):
            continue
        if parse_float(row.get("labresult")) is None:
            continue
        name = normalize_text(row.get("labname"))
        if name:
            counts[name] += 1
    return [name for name, _ in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[: config.top_labs]]


def _aggregate_labs(raw_root: Path, stay_nids: Dict[str, int], config: BuildConfig, selected_labs: Sequence[str]):
    selected = set(selected_labs)
    features = defaultdict(dict)
    for row in open_rows(raw_root / "lab.csv"):
        stay_id = str(row.get("patientunitstayid", ""))
        if stay_id not in stay_nids or not within_feature_window(row.get("labresultoffset"), config.time_window_hours):
            continue
        name = normalize_text(row.get("labname"))
        value = parse_float(row.get("labresult"))
        if name not in selected or value is None:
            continue
        stats = features[stay_id].setdefault(name, NumericStats())
        stats.update(value, parse_float(row.get("labresultoffset")))
    return features


def _aggregate_vitals(raw_root: Path, stay_nids: Dict[str, int], config: BuildConfig):
    features = defaultdict(dict)
    for row in open_rows(raw_root / "vitalPeriodic.csv"):
        stay_id = str(row.get("patientunitstayid", ""))
        if stay_id not in stay_nids or not within_feature_window(row.get("observationoffset"), config.time_window_hours):
            continue
        offset = parse_float(row.get("observationoffset"))
        for name in VITAL_COLUMNS:
            value = parse_float(row.get(name))
            if value is not None:
                features[stay_id].setdefault(name, NumericStats()).update(value, offset)
    return features


def _stat_columns(prefix: str, names: Sequence[str]) -> List[str]:
    result = []
    for name in names:
        base = f"{prefix}_{safe_name(name)}"
        result.extend([f"{base}_count", f"{base}_mean", f"{base}_min", f"{base}_max", f"{base}_last"])
    return result


def _add_stats(target: dict, prefix: str, values: Dict[str, NumericStats]) -> None:
    for name, stats in values.items():
        count, mean, minimum, maximum, last = stats.values()
        base = f"{prefix}_{safe_name(name)}"
        target.update(
            {
                f"{base}_count": count, f"{base}_mean": mean, f"{base}_min": minimum,
                f"{base}_max": maximum, f"{base}_last": last,
            }
        )


def _write_relations(raw_root: Path, output_root: Path, stay_nids: Dict[str, int]):
    specs = (
        ("diagnosis.csv", "edges_stay_diagnosis.csv", "diagnosis_concepts.csv", "diagnosis_nid", diagnosis_key,
         ("diagnosis_nid", "concept_key", "icd9code", "diagnosisstring")),
        ("treatment.csv", "edges_stay_treatment.csv", "treatment_concepts.csv", "treatment_nid", treatment_key,
         ("treatment_nid", "concept_key", "treatmentstring")),
        ("medication.csv", "edges_stay_medication.csv", "medication_concepts.csv", "medication_nid", medication_key,
         ("medication_nid", "concept_key", "drughiclseqno", "drugname")),
    )
    counts = {}
    for source_name, edge_name, concept_name, id_column, key_function, concept_fields in specs:
        concepts = {}
        edge_count = 0
        with (output_root / edge_name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=("stay_nid", id_column))
            writer.writeheader()
            for row in open_rows(raw_root / source_name):
                stay_id = str(row.get("patientunitstayid", ""))
                if stay_id not in stay_nids:
                    continue
                if source_name == "medication.csv" and normalize_text(row.get("drugordercancelled")) == "yes":
                    continue
                values = key_function(row)
                key = values[0]
                if not key or key.endswith(":"):
                    continue
                if key not in concepts:
                    concept_id = len(concepts)
                    concepts[key] = dict(zip(concept_fields, (concept_id,) + values))
                writer.writerow({"stay_nid": stay_nids[stay_id], id_column: concepts[key][id_column]})
                edge_count += 1
        write_rows(output_root / concept_name, concept_fields, sorted(concepts.values(), key=lambda row: int(row[id_column])))
        counts[edge_name] = edge_count
        counts[concept_name] = len(concepts)
    return counts


def build_intermediate(raw_root: Path, output_root: Path, config: BuildConfig = BuildConfig()) -> dict:
    raw_root = Path(raw_root)
    output_root = Path(output_root)
    _validate_inputs(raw_root)
    output_root.mkdir(parents=True, exist_ok=True)

    vocabs = {name: Vocab() for name in ("gender", "ethnicity", "hospitaladmitsource", "unittype", "unitadmitsource", "unitstaytype")}
    log("reading patient.csv and deriving LOS labels")
    stays, patient_nids, stay_nids, patient_aggs, client_patients = _read_stays(raw_root, config, vocabs)
    stay_to_patient = {stay["patientunitstayid"]: stay["uniquepid"] for stay in stays}
    _ingest_patient_concepts(raw_root, stay_to_patient, patient_aggs, config.time_window_hours)
    top_history = _top_patient_concepts(patient_aggs, "history", config.top_history)
    top_allergy = _top_patient_concepts(patient_aggs, "allergy", config.top_allergy)

    patient_columns = (
        ["age", "gender_id", "ethnicity_id", "height", "past_history_count", "allergy_count"]
        + [f"history_{index:03d}" for index in range(len(top_history))]
        + [f"allergy_{index:03d}" for index in range(len(top_allergy))]
    )
    patient_rows = []
    for patient_id, patient_nid in sorted(patient_nids.items(), key=lambda item: item[1]):
        aggregate = patient_aggs[patient_id]
        row = {
            "patient_nid": patient_nid,
            "uniquepid": patient_id,
            "age": aggregate["age"].values()[1],
            "gender_id": vocabs["gender"].add(most_common(aggregate["gender"])),
            "ethnicity_id": vocabs["ethnicity"].add(most_common(aggregate["ethnicity"])),
            "height": aggregate["height"].values()[1],
            "past_history_count": len(aggregate["history"]),
            "allergy_count": len(aggregate["allergy"]),
        }
        row.update({f"history_{index:03d}": int(value in aggregate["history"]) for index, value in enumerate(top_history)})
        row.update({f"allergy_{index:03d}": int(value in aggregate["allergy"]) for index, value in enumerate(top_allergy)})
        patient_rows.append(row)
    write_rows(output_root / "patients.csv", ["patient_nid", "uniquepid"] + patient_columns, patient_rows)

    log("building admission and first-24-hour stay features")
    apache, apache_columns = _ingest_apache(raw_root, stay_nids)
    top_labs = _top_labs(raw_root, stay_nids, config)
    labs = _aggregate_labs(raw_root, stay_nids, config, top_labs)
    vitals = _aggregate_vitals(raw_root, stay_nids, config)
    stay_columns = [
        "admissionweight", "unitvisitnumber", "hospitaladmitsource_id", "unittype_id",
        "unitadmitsource_id", "unitstaytype_id",
    ] + apache_columns + _stat_columns("lab", top_labs) + _stat_columns("vital", VITAL_COLUMNS)
    assert_feature_columns_safe(stay_columns)

    stay_feature_rows = []
    for stay in stays:
        stay_id = stay["patientunitstayid"]
        row = {"stay_nid": stay["stay_nid"], "patientunitstayid": stay_id}
        for column in stay_columns:
            row[column] = stay.get(column, MISSING)
        row.update(apache.get(stay_id, {}))
        _add_stats(row, "lab", labs.get(stay_id, {}))
        _add_stats(row, "vital", vitals.get(stay_id, {}))
        stay_feature_rows.append(row)
    write_rows(output_root / "stay_features.csv", ["stay_nid", "patientunitstayid"] + stay_columns, stay_feature_rows)

    stay_fields = ("stay_nid", "patientunitstayid", "patient_nid", "uniquepid", "hospitalid", "icu_los_3class")
    write_rows(output_root / "stays.csv", stay_fields, stays)
    write_rows(
        output_root / "edges_patient_stay.csv",
        ("patient_nid", "stay_nid"),
        ({"patient_nid": stay["patient_nid"], "stay_nid": stay["stay_nid"]} for stay in stays),
    )

    log("writing full-stay diagnosis, treatment, and medication topology")
    relation_counts = _write_relations(raw_root, output_root, stay_nids)

    assignments = split_records_by_hospital(stays, config.train_fraction, config.val_fraction, config.seed)
    split_rows = []
    split_counts = Counter()
    class_split_counts = defaultdict(Counter)
    for stay in stays:
        split = assignments[(stay["hospitalid"], stay["uniquepid"])]
        split_rows.append({"hospitalid": stay["hospitalid"], "stay_nid": stay["stay_nid"], "split": split})
        split_counts[split] += 1
        class_split_counts[str(stay["icu_los_3class"])][split] += 1
    write_rows(output_root / "splits.csv", ("hospitalid", "stay_nid", "split"), split_rows)

    client_rows = []
    for hospital_id in config.retained_clients:
        client_stays = [stay for stay in stays if stay["hospitalid"] == hospital_id]
        labels = Counter(stay["icu_los_3class"] for stay in client_stays)
        client_rows.append(
            {
                "hospitalid": hospital_id,
                "n_stays": len(client_stays),
                "n_patients": len(client_patients[hospital_id]),
                "los_class_0": labels[0], "los_class_1": labels[1], "los_class_2": labels[2],
            }
        )
    client_rows.sort(key=lambda row: (-int(row["n_stays"]), int(row["hospitalid"])))
    for index, row in enumerate(client_rows):
        row["client_id"] = index
    write_rows(
        output_root / "clients.csv",
        ("client_id", "hospitalid", "n_stays", "n_patients", "los_class_0", "los_class_1", "los_class_2"),
        client_rows,
    )

    schema = {
        "schema_version": "eicu2-intermediate-1.0",
        "source_database": "eICU Collaborative Research Database v2.0",
        "client_key": "hospitalid",
        "target_node": "stay",
        "tasks": {
            "icu_los_3class": {
                "0": "ICU LOS < 3 days", "1": "3 <= ICU LOS < 7 days", "2": "ICU LOS >= 7 days"
            }
        },
        "label_source_not_feature": "patient.csv:unitdischargeoffset",
        "feature_time_window_hours": config.time_window_hours,
        "relation_time_window": "full ICU stay",
        "patient_feature_columns": patient_columns,
        "stay_feature_columns": stay_columns,
        "forbidden_stay_features_checked": True,
        "retained_clients": [row["hospitalid"] for row in client_rows],
        "split": {
            "unit": "patient within hospital", "seed": config.seed,
            "train": config.train_fraction, "val": config.val_fraction,
            "test": 1.0 - config.train_fraction - config.val_fraction,
            "method": "deterministic grouped class-stratified allocation",
        },
        "node_types": ["patient", "stay", "diagnosis_concept", "treatment_concept", "medication_concept"],
        "edge_types": ["patient_has_stay", "stay_has_diagnosis", "stay_has_treatment", "stay_has_medication"],
        "categorical_vocabs": {name: vocab.values for name, vocab in vocabs.items()},
        "top_history_concepts": top_history,
        "top_allergy_concepts": top_allergy,
        "top_lab_concepts": top_labs,
        "counts": {
            "patients": len(patient_nids), "stays": len(stays), "clients": len(client_rows),
            "patient_stay_edges": len(stays), **relation_counts,
            "split_counts": dict(split_counts),
            "class_split_counts": {label: dict(counts) for label, counts in class_split_counts.items()},
        },
    }
    (output_root / "schema.json").write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"complete: {len(stays)} stays across {len(client_rows)} clients")
    return schema


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--retained-clients", default=",".join(DEFAULT_RETAINED_CLIENTS))
    parser.add_argument("--min-client-patients", type=int, default=1000)
    parser.add_argument("--time-window-hours", type=float, default=24.0)
    parser.add_argument("--top-labs", type=int, default=50)
    parser.add_argument("--top-history", type=int, default=100)
    parser.add_argument("--top-allergy", type=int, default=50)
    parser.add_argument("--train-fraction", type=float, default=0.05)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    retained = tuple(value.strip() for value in args.retained_clients.split(",") if value.strip())
    config = BuildConfig(
        retained_clients=retained,
        min_client_patients=args.min_client_patients,
        time_window_hours=args.time_window_hours,
        top_labs=args.top_labs,
        top_history=args.top_history,
        top_allergy=args.top_allergy,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    build_intermediate(args.raw_root, args.output_root, config)


if __name__ == "__main__":
    main()
