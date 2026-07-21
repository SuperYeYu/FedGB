"""Pure-Python rules shared by the eICU_2 construction stages."""

from __future__ import annotations

import hashlib
import math
import random
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence


SPLIT_NAMES = ("train", "val", "test")
FORBIDDEN_STAY_FEATURES = frozenset(
    {
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
        "hospital_los_days",
        "icu_los_days",
        "icu_los_3class",
    }
)


def parse_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    if text.startswith(">"):
        text = text[1:].strip()
        try:
            return float(text) + 1.0
        except ValueError:
            return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def los_3class(unit_discharge_offset_minutes: object) -> int:
    minutes = parse_float(unit_discharge_offset_minutes)
    if minutes is None or minutes < 0:
        raise ValueError("unitdischargeoffset must be a non-negative number")
    days = minutes / 1440.0
    if days < 3.0:
        return 0
    if days < 7.0:
        return 1
    return 2


def within_feature_window(offset_minutes: object, hours: float = 24.0) -> bool:
    offset = parse_float(offset_minutes)
    return offset is not None and 0.0 <= offset <= float(hours) * 60.0


def assert_feature_columns_safe(columns: Iterable[str]) -> None:
    normalized = {str(column).strip().lower() for column in columns}
    forbidden = sorted(normalized & FORBIDDEN_STAY_FEATURES)
    discharge_named = sorted(column for column in normalized if "discharge" in column)
    leaks = sorted(set(forbidden + discharge_named))
    if leaks:
        raise ValueError("forbidden stay feature columns: " + ", ".join(leaks))


def max_split_ratio_error(
    counts: Mapping[str, int], train_fraction: float = 0.05, val_fraction: float = 0.15
) -> float:
    total = sum(int(counts.get(split, 0)) for split in SPLIT_NAMES)
    if total <= 0:
        raise ValueError("split counts must contain at least one sample")
    targets = {
        "train": train_fraction,
        "val": val_fraction,
        "test": 1.0 - train_fraction - val_fraction,
    }
    return max(abs(int(counts.get(split, 0)) / total - targets[split]) for split in SPLIT_NAMES)


def _largest_remainder_counts(size: int, fractions: Sequence[float]) -> List[int]:
    raw = [size * value for value in fractions]
    counts = [int(math.floor(value)) for value in raw]
    remainder = size - sum(counts)
    order = sorted(range(len(raw)), key=lambda index: (-(raw[index] - counts[index]), index))
    for index in order[:remainder]:
        counts[index] += 1
    return counts


def _seed_for(seed: int, *parts: object) -> int:
    payload = "|".join([str(seed)] + [str(part) for part in parts])
    return int(hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16], 16)


def _single_label_assignments(
    patient_vectors: Mapping[str, Counter], fractions: Sequence[float], seed: int
) -> Dict[str, str]:
    by_label = defaultdict(list)
    for patient_id, vector in patient_vectors.items():
        label = next(iter(vector))
        by_label[label].append((patient_id, int(vector[label])))

    result: Dict[str, str] = {}
    for label, patient_groups in sorted(by_label.items()):
        ordered = sorted(patient_groups)
        random.Random(_seed_for(seed, label)).shuffle(ordered)
        ordered.sort(key=lambda item: -item[1])
        stay_targets = dict(zip(SPLIT_NAMES, _largest_remainder_counts(sum(weight for _, weight in ordered), fractions)))
        patient_targets = dict(zip(SPLIT_NAMES, _largest_remainder_counts(len(ordered), fractions)))
        assigned_stays = Counter()
        assigned_patients = Counter()
        for patient_id, weight in ordered:
            candidates = []
            for split_index, split in enumerate(SPLIT_NAMES):
                stay_scale = max(1, stay_targets[split])
                patient_scale = max(1, patient_targets[split])
                stay_cost = ((assigned_stays[split] + weight - stay_targets[split]) / stay_scale) ** 2
                patient_cost = ((assigned_patients[split] + 1 - patient_targets[split]) / patient_scale) ** 2
                overflow = max(0, assigned_stays[split] + weight - stay_targets[split])
                candidates.append((overflow, stay_cost + 0.1 * patient_cost, split_index, split))
            split = min(candidates)[-1]
            result[patient_id] = split
            assigned_stays[split] += weight
            assigned_patients[split] += 1
    return result


def _mixed_label_assignments(
    patient_vectors: Mapping[str, Counter], fractions: Sequence[float], seed: int
) -> Dict[str, str]:
    labels = sorted({label for vector in patient_vectors.values() for label in vector})
    totals = Counter()
    for vector in patient_vectors.values():
        totals.update(vector)
    targets = {
        split: {label: totals[label] * fraction for label in labels}
        for split, fraction in zip(SPLIT_NAMES, fractions)
    }
    assigned = {split: Counter() for split in SPLIT_NAMES}
    ordered = sorted(patient_vectors)
    random.Random(_seed_for(seed, "joint-greedy")).shuffle(ordered)
    ordered.sort(key=lambda patient_id: -sum(patient_vectors[patient_id].values()))

    result: Dict[str, str] = {}
    for patient_id in ordered:
        vector = patient_vectors[patient_id]
        candidates = []
        for split_index, candidate_split in enumerate(SPLIT_NAMES):
            cost = 0.0
            for split in SPLIT_NAMES:
                for label in labels:
                    count = assigned[split][label]
                    if split == candidate_split:
                        count += vector[label]
                    cost += ((count - targets[split][label]) / max(1, totals[label])) ** 2
            candidates.append((cost, split_index, candidate_split))
        split = min(candidates)[-1]
        result[patient_id] = split
        assigned[split].update(vector)
    return result


def assign_patient_stratified_splits(
    records: Iterable[Mapping[str, object]],
    train_fraction: float = 0.05,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> Dict[str, str]:
    if train_fraction <= 0 or val_fraction < 0 or train_fraction + val_fraction >= 1:
        raise ValueError("split fractions must satisfy train > 0, val >= 0, train + val < 1")

    patient_vectors: Dict[str, Counter] = defaultdict(Counter)
    hospitals = set()
    for record in records:
        patient_id = str(record["uniquepid"])
        label = int(record["icu_los_3class"])
        if label not in (0, 1, 2):
            raise ValueError(f"invalid LOS class {label}")
        hospitals.add(str(record["hospitalid"]))
        patient_vectors[patient_id][label] += 1
    if len(hospitals) > 1:
        raise ValueError("assign_patient_stratified_splits expects one hospital at a time")

    fractions = (train_fraction, val_fraction, 1.0 - train_fraction - val_fraction)
    return _mixed_label_assignments(patient_vectors, fractions, seed)


def split_records_by_hospital(
    records: Iterable[Mapping[str, object]],
    train_fraction: float = 0.05,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> Dict[tuple, str]:
    grouped = defaultdict(list)
    for record in records:
        grouped[str(record["hospitalid"])].append(record)
    assignments = {}
    for hospital_id, client_records in sorted(grouped.items(), key=lambda item: int(item[0])):
        client_assignments = assign_patient_stratified_splits(
            client_records, train_fraction, val_fraction, _seed_for(seed, hospital_id)
        )
        for patient_id, split in client_assignments.items():
            assignments[(hospital_id, patient_id)] = split
    return assignments
