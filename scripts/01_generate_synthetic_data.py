from __future__ import annotations

import hashlib
import io
import json
import math
import re
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np
import openpyxl
import pandas as pd


SEED = 42
BASE_SAMPLE_COUNT = 15_132
SPLIT_TARGETS = {"train": 12_106, "validation": 1_513, "test": 1_513}
SENSOR_COLUMNS = [
    "temperature_c",
    "humidity_pct",
    "vibration_ms2",
    "acceleration_g",
    "noise_db",
]
HEADER_MAP = {
    "timestamp": "timestamp",
    "temperature(°C)": "temperature_c",
    "humidity(%)": "humidity_pct",
    "vibration(m/s²)": "vibration_ms2",
    "acceleration(g)": "acceleration_g",
    "noise(dB)": "noise_db",
}
ROUND_DIGITS = {
    "temperature_c": 2,
    "humidity_pct": 2,
    "vibration_ms2": 4,
    "acceleration_g": 4,
    "noise_db": 2,
}
SPLIT_PREFIX = {"train": "TR", "validation": "VA", "test": "TE"}
FORBIDDEN_OPERATION_TERMS = [
    "정상",
    "불량",
    "미납",
    "냉납",
    "밀림",
    "쇼트",
    "오삽",
    "미삽",
    "역삽",
    "뒤집힘",
    "일어섬",
    "납볼",
    "납금감",
    "핀홀",
    "납고드름",
]


def progress(message: str) -> None:
    print(message, flush=True)


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def parse_zip_metadata(path: Path) -> tuple[str, str, str]:
    provided_split = "Training" if "Training" in path.parts else "Validation"
    match = re.search(r"_(불량|정상)_(\d+)\.(.+)\.zip$", path.name)
    if not match:
        raise ValueError(f"ZIP 이름을 해석할 수 없습니다: {path.name}")
    status_ko, _, defect_type = match.groups()
    status = "DEF" if status_ko == "불량" else "NOR"
    return provided_split, status, defect_type


def build_source_index(raw_dir: Path) -> pd.DataFrame:
    source_rows: list[dict] = []
    label_index: dict[tuple[str, str], tuple[str, str]] = {}
    zip_paths = sorted(raw_dir.rglob("*.zip"))
    if len(zip_paths) != 96:
        raise ValueError(f"ZIP 파일 수가 96개가 아닙니다: {len(zip_paths)}")

    for zip_path in zip_paths:
        provided_split, status, defect_type = parse_zip_metadata(zip_path)
        is_source = "01.원천데이터" in zip_path.parts
        expected_suffix = ".xlsx" if is_source else ".json"
        with zipfile.ZipFile(zip_path) as archive:
            members = [n for n in archive.namelist() if n.lower().endswith(expected_suffix)]
        for member in members:
            stem = PurePosixPath(member).stem
            key = (provided_split, stem)
            if is_source:
                tokens = stem.split("_")
                if len(tokens) < 5 or tokens[3] not in {"A", "B", "C", "D"}:
                    raise ValueError(f"원본 파일명을 해석할 수 없습니다: {stem}")
                source_rows.append(
                    {
                        "provided_split": provided_split,
                        "status": status,
                        "defect_type": defect_type,
                        "source_group": tokens[3],
                        "source_stem": stem,
                        "source_zip": str(zip_path),
                        "source_member": member,
                    }
                )
            else:
                if key in label_index:
                    raise ValueError(f"중복 라벨 stem: {key}")
                label_index[key] = (str(zip_path), member)

    index = pd.DataFrame(source_rows)
    if len(index) != 27_830:
        raise ValueError(f"원천 XLSX 수가 27,830개가 아닙니다: {len(index)}")
    if len(label_index) != 27_830:
        raise ValueError(f"라벨 JSON 수가 27,830개가 아닙니다: {len(label_index)}")
    if index.duplicated(["provided_split", "source_stem"]).any():
        raise ValueError("원천 stem 중복이 있습니다.")

    labels = []
    for row in index.itertuples(index=False):
        key = (row.provided_split, row.source_stem)
        if key not in label_index:
            raise ValueError(f"대응 라벨이 없습니다: {key}")
        labels.append(label_index[key])
    index[["label_zip", "label_member"]] = pd.DataFrame(labels, index=index.index)
    return index


def proportional_allocation(counts: pd.Series, target: int) -> pd.Series:
    raw = counts.astype(float) * target / counts.sum()
    allocation = np.floor(raw).astype(int)
    remainder = target - int(allocation.sum())
    order = sorted(
        counts.index,
        key=lambda key: (-(raw.loc[key] - allocation.loc[key]), str(key)),
    )
    for key in order:
        if remainder == 0:
            break
        if allocation.loc[key] < counts.loc[key]:
            allocation.loc[key] += 1
            remainder -= 1
    if remainder:
        raise ValueError("층화 표본 수를 배분하지 못했습니다.")
    return allocation


def stratified_take(
    frame: pd.DataFrame,
    target: int,
    strata: list[str],
    rng: np.random.Generator,
) -> pd.DataFrame:
    grouped = frame.groupby(strata, dropna=False, sort=True)
    counts = grouped.size()
    allocation = proportional_allocation(counts, target)
    selected_indices: list[int] = []
    for key, group in grouped:
        n = int(allocation.loc[key])
        if n:
            selected_indices.extend(rng.choice(group.index.to_numpy(), size=n, replace=False).tolist())
    result = frame.loc[selected_indices].copy()
    if len(result) != target:
        raise AssertionError((len(result), target))
    return result


def select_samples(index: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    strata = ["status", "defect_type", "source_group"]
    training_pool = index[index["provided_split"] == "Training"]
    validation_pool = index[index["provided_split"] == "Validation"]

    training_selected = stratified_take(training_pool, 13_619, strata, rng)
    validation_selected = stratified_take(training_selected, 1_513, strata, rng)
    train_selected = training_selected.drop(validation_selected.index)
    test_selected = stratified_take(validation_pool, 1_513, strata, rng)

    train_selected = train_selected.assign(split="train")
    validation_selected = validation_selected.assign(split="validation")
    test_selected = test_selected.assign(split="test")
    selected = pd.concat([train_selected, validation_selected, test_selected], ignore_index=True)

    for split, expected in SPLIT_TARGETS.items():
        actual = int((selected["split"] == split).sum())
        if actual != expected:
            raise AssertionError((split, actual, expected))
    if selected["source_stem"].duplicated().any():
        raise ValueError("분할 간 원본 stem이 겹칩니다.")

    selected["split_order"] = selected["split"].map({"train": 0, "validation": 1, "test": 2})
    selected = selected.sort_values(["split_order", "source_stem"]).reset_index(drop=True)
    selected["sample_id"] = ""
    for split, group in selected.groupby("split", sort=False):
        prefix = SPLIT_PREFIX[split]
        selected.loc[group.index, "sample_id"] = [
            f"SYN_{prefix}_{i:06d}" for i in range(1, len(group) + 1)
        ]
    return selected.drop(columns="split_order")


def read_xlsx_sequence(blob: bytes) -> list[dict]:
    workbook = openpyxl.load_workbook(io.BytesIO(blob), read_only=True, data_only=True)
    try:
        values = list(workbook.active.iter_rows(values_only=True))
    finally:
        workbook.close()
    if len(values) != 6:
        raise ValueError(f"XLSX 행 수가 6이 아닙니다: {len(values)}")
    header = list(values[0])
    if header != list(HEADER_MAP):
        raise ValueError(f"예상하지 못한 XLSX 헤더: {header}")
    rows = []
    for sequence_no, values_row in enumerate(values[1:], start=1):
        raw = dict(zip(header, values_row, strict=True))
        rows.append(
            {
                "sequence_no": sequence_no,
                "timestamp": pd.Timestamp(raw["timestamp"]),
                **{
                    normalized: float(raw[source])
                    for source, normalized in HEADER_MAP.items()
                    if source != "timestamp"
                },
            }
        )
    return rows


def load_selected_data(selected: pd.DataFrame) -> pd.DataFrame:
    sensor_rows: list[dict] = []
    progress("선정한 XLSX 센서 시퀀스를 읽는 중입니다.")
    for zip_name, group in selected.groupby("source_zip", sort=True):
        with zipfile.ZipFile(zip_name) as archive:
            for row in group.itertuples(index=False):
                sequence = read_xlsx_sequence(archive.read(row.source_member))
                for sensor_row in sequence:
                    sensor_rows.append(
                        {
                            "sample_id": row.sample_id,
                            "split": row.split,
                            "source_stem": row.source_stem,
                            "source_group": row.source_group,
                            "defect_detected": 1 if row.status == "DEF" else 0,
                            "defect_type": row.defect_type,
                            **sensor_row,
                        }
                    )

    progress("선정한 JSON 라벨을 대조하는 중입니다.")
    checked = 0
    for zip_name, group in selected.groupby("label_zip", sort=True):
        with zipfile.ZipFile(zip_name) as archive:
            for row in group.itertuples(index=False):
                payload = json.loads(archive.read(row.label_member).decode("utf-8-sig"))
                sequences = payload.get("sensor_data", [])
                if len(sequences) != 1 or len(sequences[0].get("sensor_sequence", [])) != 5:
                    raise ValueError(f"라벨 센서 시퀀스 구조 오류: {row.source_stem}")
                observed = {int(x["defect_detected"]) for x in sequences[0]["sensor_sequence"]}
                expected = {1 if row.status == "DEF" else 0}
                if observed != expected:
                    raise ValueError(f"라벨 불일치: {row.source_stem} {observed} {expected}")
                checked += 1
    if checked != BASE_SAMPLE_COUNT:
        raise AssertionError(checked)

    frame = pd.DataFrame(sensor_rows)
    if len(frame) != BASE_SAMPLE_COUNT * 5:
        raise AssertionError(len(frame))
    return frame


def make_sensor_semisynthetic(
    original: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.DataFrame, dict]:
    result = original.copy()
    train = original[original["split"] == "train"]
    stats: dict[str, dict[str, float]] = {}
    sample_ids = result["sample_id"].drop_duplicates().to_numpy()

    for sensor in SENSOR_COLUMNS:
        std = float(train[sensor].std(ddof=1))
        lower = float(train[sensor].min())
        upper = float(train[sensor].max())
        q1 = float(train[sensor].quantile(0.25))
        q3 = float(train[sensor].quantile(0.75))
        offsets = pd.Series(
            rng.normal(0.0, std * 0.03, len(sample_ids)), index=sample_ids
        )
        offset_values = result["sample_id"].map(offsets).to_numpy()
        row_noise = rng.normal(0.0, std * 0.01, len(result))
        values = result[sensor].to_numpy(dtype=float) + offset_values + row_noise
        values = np.clip(values, lower, upper)
        result[sensor] = np.round(values, ROUND_DIGITS[sensor])
        stats[sensor] = {
            "std": std,
            "min": lower,
            "max": upper,
            "q1": q1,
            "q3": q3,
            "iqr": q3 - q1,
        }
    return result, stats


def assign_operational_fields(
    sensor: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame]:
    first_rows = (
        sensor.sort_values(["sample_id", "sequence_no"])
        .groupby("sample_id", as_index=False)
        .first()
    )
    sample_meta = first_rows[
        [
            "sample_id",
            "split",
            "source_stem",
            "source_group",
            "timestamp",
            "defect_detected",
            "defect_type",
        ]
    ].rename(columns={"timestamp": "sample_start_at"})

    def equipment_for(row: pd.Series) -> str:
        group_offset = ord(row["source_group"]) - ord("A")
        machine_offset = stable_int(row["source_stem"]) % 2
        return f"EQP_SOLDER_{group_offset * 2 + machine_offset + 1:02d}"

    sample_meta["equipment_id"] = sample_meta.apply(equipment_for, axis=1)
    hours = sample_meta["sample_start_at"].dt.hour
    sample_meta["shift_code"] = np.where((hours >= 8) & (hours < 20), "DAY", "NIGHT")
    sample_meta["shift_date"] = sample_meta["sample_start_at"].dt.normalize()
    before_eight = hours < 8
    sample_meta.loc[before_eight, "shift_date"] -= pd.Timedelta(days=1)
    sample_meta["lot_id"] = ""

    counters = Counter()
    group_columns = ["split", "equipment_id", "shift_date", "shift_code"]
    for keys, group in sample_meta.groupby(group_columns, sort=True):
        split = keys[0]
        ordered = group.sort_values(["sample_start_at", "sample_id"]).index.to_list()
        position = 0
        while position < len(ordered):
            remaining = len(ordered) - position
            if remaining <= 40:
                chunk_size = remaining
            else:
                chunk_size = int(rng.integers(20, 41))
                if remaining - chunk_size < 10:
                    chunk_size = remaining
            counters[split] += 1
            lot_id = f"LOT_{SPLIT_PREFIX[split]}_{counters[split]:05d}"
            chosen = ordered[position : position + chunk_size]
            sample_meta.loc[chosen, "lot_id"] = lot_id
            position += chunk_size

    if (sample_meta["lot_id"] == "").any():
        raise AssertionError("lot_id가 비어 있습니다.")

    operational = sample_meta[
        ["sample_id", "equipment_id", "lot_id", "shift_date", "shift_code"]
    ]
    result = sensor.merge(operational, on="sample_id", how="left", validate="many_to_one")
    result["source_type"] = "semi_synthetic"
    return result, sample_meta


def allocate_by_split(total: int, split_sizes: dict[str, int]) -> dict[str, int]:
    counts = pd.Series(split_sizes, dtype=int)
    allocation = proportional_allocation(counts, total)
    return {key: int(value) for key, value in allocation.items()}


def make_log_row(
    log_id: str,
    split: str,
    created: pd.Timestamp,
    start: pd.Timestamp,
    end: pd.Timestamp,
    shift_date: pd.Timestamp,
    shift_code: str,
    equipment_id: str,
    lot_id: str | None,
    log_type: str,
    record_stage: str,
    note: str,
    action: str,
    downtime: float,
    writer_role: str,
    target_sample_id: str | None = None,
) -> dict:
    return {
        "log_id": log_id,
        "split": split,
        "log_created_at": created,
        "related_start_at": start,
        "related_end_at": end,
        "shift_date": shift_date,
        "shift_code": shift_code,
        "equipment_id": equipment_id,
        "lot_id": lot_id,
        "log_type": log_type,
        "record_stage": record_stage,
        "note_text": note,
        "action_text": action,
        "downtime_minutes": downtime,
        "writer_role": writer_role,
        "source_type": "synthetic",
        "_target_sample_id": target_sample_id,
    }


def build_worklogs(
    clean_sensor: pd.DataFrame,
    sample_meta: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict]:
    logs: list[dict] = []
    counters = Counter()

    def next_log_id(split: str) -> str:
        counters[split] += 1
        return f"LOG_{SPLIT_PREFIX[split]}_{counters[split]:06d}"

    sensor_ranges = (
        clean_sensor.groupby("sample_id", as_index=False)
        .agg(sample_end_at=("timestamp", "max"))
    )
    samples = sample_meta.merge(sensor_ranges, on="sample_id", how="left", validate="one_to_one")

    shift_cols = ["split", "equipment_id", "shift_date", "shift_code"]
    for keys, group in samples.groupby(shift_cols, sort=True):
        split, equipment, shift_date, shift_code = keys
        start = group["sample_start_at"].min()
        end = group["sample_end_at"].max()
        is_start = stable_int(f"{equipment}-{shift_date}-{shift_code}") % 2 == 0
        if is_start:
            log_type = "시작점검"
            created = start - pd.Timedelta(minutes=10)
            note = f"{equipment} 교대 시작 전 노즐, 피더, 컨베이어 상태 확인."
            action = "안전장치와 운전 조건 점검 완료."
            role = "작업자"
        else:
            log_type = "인수인계"
            created = end + pd.Timedelta(minutes=5)
            note = f"{equipment} 교대 운전 현황과 관찰사항 인계."
            action = "다음 교대조에 초기 운전 상태 재확인 요청."
            role = "조장"
        logs.append(
            make_log_row(
                next_log_id(split), split, created, start, end, shift_date,
                shift_code, equipment, None, log_type, "operation", note,
                action, 0.0, role,
            )
        )

    for (split, lot_id), group in samples.groupby(["split", "lot_id"], sort=True):
        start = group["sample_start_at"].min()
        end = group["sample_end_at"].max()
        equipment = group["equipment_id"].iloc[0]
        shift_date = group["shift_date"].iloc[0]
        shift_code = group["shift_code"].iloc[0]
        logs.append(
            make_log_row(
                next_log_id(split), split, end + pd.Timedelta(minutes=2), start, end,
                shift_date, shift_code, equipment, lot_id, "생산실적", "operation",
                f"{equipment} {lot_id} 생산 구간 기록 마감.",
                "생산 수량과 설비 운전 상태 확인 완료.", 0.0, "작업자",
            )
        )

    sample_features = clean_sensor.groupby("sample_id")[SENSOR_COLUMNS].mean()
    sample_features = samples.set_index("sample_id").join(sample_features)
    train_features = sample_features[sample_features["split"] == "train"]
    medians = train_features[SENSOR_COLUMNS].median()
    mads = (train_features[SENSOR_COLUMNS] - medians).abs().median().replace(0, np.nan)
    scales = (1.4826 * mads).fillna(train_features[SENSOR_COLUMNS].std()).replace(0, 1.0)
    robust_scores = ((sample_features[SENSOR_COLUMNS] - medians) / scales).abs()
    sample_features["event_score"] = robust_scores.max(axis=1)
    sample_features["event_sensor"] = robust_scores.idxmax(axis=1)

    event_total = round(BASE_SAMPLE_COUNT * 0.02)
    event_allocation = allocate_by_split(event_total, SPLIT_TARGETS)
    event_samples = []
    for split, n in event_allocation.items():
        event_samples.append(
            sample_features[sample_features["split"] == split]
            .sort_values(["event_score", "sample_start_at"], ascending=[False, True])
            .head(n)
        )
    events = pd.concat(event_samples)
    event_descriptions = {
        "temperature_c": "온도 변동",
        "humidity_pct": "습도 변동",
        "vibration_ms2": "진동 증가",
        "acceleration_g": "구동부 움직임 변화",
        "noise_db": "운전 소음 증가",
    }
    for sample_id, row in events.sort_values(["split", "sample_start_at"]).iterrows():
        note = f"{row.equipment_id} {event_descriptions[row.event_sensor]} 확인, 추이 관찰 필요."
        action = "연결부와 운전 조건 점검 후 재가동 완료."
        logs.append(
            make_log_row(
                next_log_id(row.split), row.split,
                row.sample_end_at + pd.Timedelta(minutes=1), row.sample_start_at,
                row.sample_end_at, row.shift_date, row.shift_code,
                row.equipment_id, row.lot_id, "이상점검", "operation",
                note, action, float(rng.integers(0, 16)), "보전",
                target_sample_id=sample_id,
            )
        )

    lot_ids = sorted(samples["lot_id"].unique())
    quality_lot_count = round(len(lot_ids) * 0.25)
    selected_lots = rng.choice(np.array(lot_ids), size=quality_lot_count, replace=False)
    for lot_id in sorted(selected_lots):
        group = samples[samples["lot_id"] == lot_id]
        row = group.iloc[int(rng.integers(0, len(group)))]
        if int(row.defect_detected) == 1:
            note = f"{row.equipment_id} {lot_id} 검사 결과 불량 판정. 결함 유형 {row.defect_type} 확인."
            action = "해당 LOT 보류 후 재검 요청."
        else:
            note = f"{row.equipment_id} {lot_id} 검사 결과 정상 판정."
            action = "검사 결과 등록 후 다음 공정 진행 승인."
        logs.append(
            make_log_row(
                next_log_id(row.split), row.split,
                row.sample_end_at + pd.Timedelta(minutes=30), row.sample_start_at,
                row.sample_end_at, row.shift_date, row.shift_code,
                row.equipment_id, lot_id, "품질검사", "post_inspection",
                note, action, 0.0, "품질",
            )
        )

    worklogs = pd.DataFrame(logs)
    operation_text = worklogs.loc[
        worklogs["record_stage"] == "operation", ["note_text", "action_text"]
    ].fillna("").agg(" ".join, axis=1)
    leaked = {
        term: int(operation_text.str.contains(term, regex=False).sum())
        for term in FORBIDDEN_OPERATION_TERMS
    }
    leaked = {key: value for key, value in leaked.items() if value}
    if leaked:
        raise ValueError(f"operation 일지에 사후 판정 표현이 있습니다: {leaked}")

    metrics = {
        "base_worklogs": len(worklogs),
        "shift_logs": int((worklogs["log_type"].isin(["시작점검", "인수인계"])).sum()),
        "production_logs": int((worklogs["log_type"] == "생산실적").sum()),
        "event_logs": int((worklogs["log_type"] == "이상점검").sum()),
        "post_inspection_logs": int((worklogs["record_stage"] == "post_inspection").sum()),
    }
    return worklogs, metrics


def choose_disjoint_by_split(
    candidates: pd.DataFrame,
    total: int,
    rng: np.random.Generator,
    excluded: set[str],
) -> set[str]:
    available = candidates[~candidates["sample_id"].isin(excluded)]
    split_sizes = {
        split: int((candidates["split"] == split).sum()) for split in SPLIT_TARGETS
    }
    allocation = allocate_by_split(total, split_sizes)
    chosen: set[str] = set()
    for split, n in allocation.items():
        ids = available.loc[available["split"] == split, "sample_id"].to_numpy()
        chosen.update(rng.choice(ids, size=n, replace=False).tolist())
    return chosen


def inject_sensor_issues(
    clean_sensor: pd.DataFrame,
    sensor_stats: dict,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict]:
    result = clean_sensor.copy()
    base_samples = result[["sample_id", "split"]].drop_duplicates()
    excluded: set[str] = set()

    missing_ids = choose_disjoint_by_split(base_samples, 454, rng, excluded)
    excluded |= missing_ids
    missing_cells = 0
    for sample_id in sorted(missing_ids):
        indices = result.index[result["sample_id"] == sample_id].to_numpy()
        cell_count = 2 if rng.random() < 0.2 else 1
        pairs = [(int(i), sensor) for i in indices for sensor in SENSOR_COLUMNS]
        for pair_index in rng.choice(len(pairs), size=cell_count, replace=False):
            row_index, sensor = pairs[int(pair_index)]
            result.at[row_index, sensor] = np.nan
            missing_cells += 1

    spike_ids = choose_disjoint_by_split(base_samples, 151, rng, excluded)
    excluded |= spike_ids
    broad_bounds = {
        "temperature_c": (0.0, 100.0),
        "humidity_pct": (0.0, 100.0),
        "vibration_ms2": (0.0, 5.0),
        "acceleration_g": (0.0, 2.0),
        "noise_db": (20.0, 130.0),
    }
    for sample_id in sorted(spike_ids):
        indices = result.index[result["sample_id"] == sample_id].to_numpy()
        row_index = int(rng.choice(indices))
        sensor = str(rng.choice(SENSOR_COLUMNS))
        stats = sensor_stats[sensor]
        multiplier = float(rng.uniform(3.0, 6.0))
        if rng.random() < 0.5:
            value = stats["q3"] + multiplier * stats["iqr"]
        else:
            value = stats["q1"] - multiplier * stats["iqr"]
        value = float(np.clip(value, *broad_bounds[sensor]))
        result.at[row_index, sensor] = round(value, ROUND_DIGITS[sensor])

    full_duplicate_ids = choose_disjoint_by_split(base_samples, 76, rng, excluded)
    excluded |= full_duplicate_ids
    duplicate_frames = []
    duplicate_counter = Counter()
    for sample_id in sorted(full_duplicate_ids):
        copy = result[result["sample_id"] == sample_id].copy()
        split = copy["split"].iloc[0]
        duplicate_counter[split] += 1
        copy["sample_id"] = (
            f"SYN_{SPLIT_PREFIX[split]}_DUP_{duplicate_counter[split]:05d}"
        )
        duplicate_frames.append(copy)

    row_duplicate_ids = choose_disjoint_by_split(base_samples, 151, rng, excluded)
    row_duplicates = []
    for sample_id in sorted(row_duplicate_ids):
        rows = result[result["sample_id"] == sample_id]
        row_duplicates.append(rows.iloc[[int(rng.integers(0, len(rows)))]].copy())

    result = pd.concat([result, *duplicate_frames, *row_duplicates], ignore_index=True)
    result = result.sort_values(["split", "sample_id", "timestamp", "sequence_no"]).reset_index(drop=True)
    metrics = {
        "sensor_missing_samples": len(missing_ids),
        "sensor_missing_cells": missing_cells,
        "sensor_spike_samples": len(spike_ids),
        "full_duplicate_samples": len(full_duplicate_ids),
        "duplicate_measurement_rows": len(row_duplicate_ids),
        "dirty_sensor_rows": len(result),
        "dirty_sample_ids": result["sample_id"].nunique(),
    }
    return result, metrics


def split_sample_indices(
    frame: pd.DataFrame,
    total: int,
    rng: np.random.Generator,
    eligible_mask: pd.Series | None = None,
) -> list[int]:
    if eligible_mask is None:
        eligible_mask = pd.Series(True, index=frame.index)
    eligible = frame[eligible_mask]
    split_sizes = eligible.groupby("split").size().to_dict()
    allocation = allocate_by_split(total, split_sizes)
    chosen: list[int] = []
    for split, n in allocation.items():
        indices = eligible.index[eligible["split"] == split].to_numpy()
        chosen.extend(rng.choice(indices, size=n, replace=False).tolist())
    return chosen


def replace_first(text: str, replacements: list[tuple[str, str]]) -> str:
    for old, new in replacements:
        if old in text:
            return text.replace(old, new, 1)
    return text


def inject_worklog_issues(
    clean_logs: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, dict]:
    result = clean_logs.copy().reset_index(drop=True)
    base_count = len(result)

    event_mask = result["_target_sample_id"].notna()
    mismatch_indices = split_sample_indices(result, 151, rng, event_mask)
    shuffled_mismatch = np.array(mismatch_indices)
    rng.shuffle(shuffled_mismatch)
    time_indices = shuffled_mismatch[:76].tolist()
    equipment_indices = shuffled_mismatch[76:].tolist()
    for index in time_indices:
        result.loc[index, ["log_created_at", "related_start_at", "related_end_at"]] += pd.Timedelta(days=1)
    equipment_values = sorted(result["equipment_id"].unique())
    for index in equipment_indices:
        current = result.at[index, "equipment_id"]
        alternatives = [value for value in equipment_values if value != current]
        result.at[index, "equipment_id"] = str(rng.choice(alternatives))

    protected_ids = set(result.loc[mismatch_indices, "log_id"])
    omit_count = round(base_count * 0.03)
    omit_mask = ~result["log_id"].isin(protected_ids)
    omit_indices = split_sample_indices(result, omit_count, rng, omit_mask)
    omitted_log_ids = set(result.loc[omit_indices, "log_id"])
    result = result.drop(index=omit_indices).reset_index(drop=True)

    blank_count = round(base_count * 0.02)
    blank_mask = ~result["log_id"].isin(protected_ids)
    blank_indices = split_sample_indices(result, blank_count, rng, blank_mask)
    result.loc[blank_indices, ["note_text", "action_text"]] = pd.NA

    nonblank_mask = result["note_text"].notna() & result["action_text"].notna()
    issue_counts = pd.Series(0, index=result.index, dtype=int)

    def apply_text_issue(name: str, rate: float) -> int:
        target = round(int(nonblank_mask.sum()) * rate)
        candidates = result.index[nonblank_mask & (issue_counts < 2)].to_numpy()
        chosen = rng.choice(candidates, size=target, replace=False).tolist()
        if name == "abbreviation":
            replacements = [
                ("컨베이어", "CV"), ("피더", "FDR"), ("노즐", "NZL"),
                ("점검", "CHK"), ("인수인계", "SHIFT 인계"), ("확인", "CHK"),
            ]
            for index in chosen:
                result.at[index, "note_text"] = replace_first(result.at[index, "note_text"], replacements)
        elif name == "spacing":
            for index in chosen:
                text = result.at[index, "action_text"]
                position = text.find(" ")
                if position >= 0:
                    result.at[index, "action_text"] = text[:position] + text[position + 1 :]
                else:
                    result.at[index, "note_text"] = result.at[index, "note_text"].replace(" ", "", 1)
        elif name == "typo":
            replacements = [
                ("확인", "확잉"), ("완료", "완로"), ("점검", "점겅"),
                ("교체", "교채"), ("관찰", "괸찰"), ("결과", "결괴"),
            ]
            for index in chosen:
                original = result.at[index, "note_text"]
                changed = replace_first(original, replacements)
                if changed == original:
                    result.at[index, "action_text"] = replace_first(result.at[index, "action_text"], replacements)
                else:
                    result.at[index, "note_text"] = changed
        elif name == "equipment_variant":
            for index in chosen:
                equipment = clean_logs.loc[
                    clean_logs["log_id"] == result.at[index, "log_id"], "equipment_id"
                ]
                if equipment.empty:
                    equipment_id = result.at[index, "equipment_id"]
                else:
                    equipment_id = equipment.iloc[0]
                number = re.search(r"(\d+)$", equipment_id).group(1)
                variants = [f"{int(number)}호기", f"SOLDER-{number}", f"설비{number}"]
                variant = str(rng.choice(variants))
                result.at[index, "note_text"] = result.at[index, "note_text"].replace(equipment_id, variant)
        else:
            raise ValueError(name)
        issue_counts.loc[chosen] += 1
        return len(chosen)

    abbreviation_count = apply_text_issue("abbreviation", 0.15)
    spacing_count = apply_text_issue("spacing", 0.07)
    typo_count = apply_text_issue("typo", 0.03)
    equipment_variant_count = apply_text_issue("equipment_variant", 0.05)
    if issue_counts.max() > 2:
        raise AssertionError("한 작업일지에 텍스트 문제를 3개 이상 주입했습니다.")

    duplicate_count = round(len(result) * 0.02)
    duplicate_source_indices = split_sample_indices(result, duplicate_count, rng, nonblank_mask.loc[result.index])
    exact_count = duplicate_count // 2
    duplicate_rows = []
    for position, index in enumerate(duplicate_source_indices, start=1):
        copy = result.loc[[index]].copy()
        if position > exact_count:
            copy["log_id"] = f"LOG_CONTENT_DUP_{position - exact_count:05d}"
        duplicate_rows.append(copy)
    result = pd.concat([result, *duplicate_rows], ignore_index=True)

    result = result.drop(columns="_target_sample_id")
    result = result.sort_values(["split", "log_created_at", "log_id"], na_position="last").reset_index(drop=True)
    target_text = result.loc[
        result["record_stage"] == "post_inspection", ["note_text", "action_text"]
    ].fillna("").agg(" ".join, axis=1)
    target_expression_logs = int(
        target_text.str.contains("정상|불량|미납|냉납|밀림|쇼트|오삽|미삽|역삽|뒤집힘|일어섬|납볼|납금감|핀홀|납고드름", regex=True).sum()
    )

    metrics = {
        "omitted_worklogs": len(omitted_log_ids),
        "blank_worklogs": blank_count,
        "time_mismatch_logs": len(time_indices),
        "equipment_mismatch_logs": len(equipment_indices),
        "abbreviation_logs": abbreviation_count,
        "spacing_logs": spacing_count,
        "typo_logs": typo_count,
        "equipment_variant_logs": equipment_variant_count,
        "exact_duplicate_logs": exact_count,
        "content_duplicate_logs": duplicate_count - exact_count,
        "dirty_worklogs": len(result),
        "target_expression_logs": target_expression_logs,
    }
    return result, metrics


def distribution_checks(original: pd.DataFrame, synthetic: pd.DataFrame) -> dict:
    checks: dict[str, dict[str, float]] = {}
    original_base = original[SENSOR_COLUMNS]
    synthetic_base = synthetic[SENSOR_COLUMNS]
    for sensor in SENSOR_COLUMNS:
        original_std = float(original_base[sensor].std(ddof=1))
        standardized_mean_difference = abs(
            float(synthetic_base[sensor].mean() - original_base[sensor].mean())
        ) / original_std
        std_ratio = float(synthetic_base[sensor].std(ddof=1) / original_std)
        if standardized_mean_difference > 0.1:
            raise AssertionError(f"{sensor} 표준화 평균 차이 초과: {standardized_mean_difference}")
        if not 0.9 <= std_ratio <= 1.1:
            raise AssertionError(f"{sensor} 표준편차 비율 초과: {std_ratio}")
        checks[sensor] = {
            "standardized_mean_difference": standardized_mean_difference,
            "std_ratio": std_ratio,
        }
    original_corr = original_base.corr()
    synthetic_corr = synthetic_base.corr()
    max_corr_difference = float((original_corr - synthetic_corr).abs().to_numpy().max())
    if max_corr_difference > 0.1:
        raise AssertionError(f"상관계수 차이 초과: {max_corr_difference}")
    return {"sensors": checks, "max_correlation_difference": max_corr_difference}


def validate_and_save(
    original: pd.DataFrame,
    clean_sensor: pd.DataFrame,
    dirty_sensor: pd.DataFrame,
    dirty_logs: pd.DataFrame,
    output_sensor: Path,
    output_logs: Path,
) -> dict:
    base_counts = clean_sensor[["sample_id", "split"]].drop_duplicates()["split"].value_counts().to_dict()
    if base_counts != SPLIT_TARGETS:
        raise AssertionError(base_counts)
    per_sample_rows = clean_sensor.groupby("sample_id").size()
    if not (per_sample_rows == 5).all():
        raise AssertionError("오류 주입 전 5행 시퀀스가 깨졌습니다.")
    ordered = clean_sensor.sort_values(["sample_id", "sequence_no"])
    timestamp_order = ordered.groupby("sample_id")["timestamp"].apply(lambda x: x.is_monotonic_increasing)
    if not timestamp_order.all():
        raise AssertionError("오류 주입 전 timestamp 순서가 깨졌습니다.")
    if len(dirty_sensor) != 76_191:
        raise AssertionError(f"센서 최종 행 수 오류: {len(dirty_sensor)}")
    if dirty_sensor["sample_id"].nunique() != 15_208:
        raise AssertionError(f"센서 최종 sample_id 수 오류: {dirty_sensor['sample_id'].nunique()}")
    if dirty_logs.empty:
        raise AssertionError("작업일지가 비어 있습니다.")
    if set(dirty_sensor["split"].unique()) != set(SPLIT_TARGETS):
        raise AssertionError("센서 split 오류")
    if set(dirty_logs["split"].unique()) != set(SPLIT_TARGETS):
        raise AssertionError("작업일지 split 오류")

    distribution = distribution_checks(original, clean_sensor)
    output_sensor.parent.mkdir(parents=True, exist_ok=True)
    dirty_sensor.to_parquet(output_sensor, index=False, compression="snappy")
    dirty_logs.to_parquet(output_logs, index=False, compression="snappy")

    sensor_check = pd.read_parquet(output_sensor)
    log_check = pd.read_parquet(output_logs)
    if len(sensor_check) != len(dirty_sensor) or len(log_check) != len(dirty_logs):
        raise AssertionError("저장 후 행 수가 달라졌습니다.")
    return {
        "base_split_counts": base_counts,
        "distribution": distribution,
        "sensor_file_bytes": output_sensor.stat().st_size,
        "worklog_file_bytes": output_logs.stat().st_size,
    }


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    raw_dir = project_root / "data" / "raw"
    output_sensor = project_root / "data" / "interim" / "synthetic_sensor_dirty.parquet"
    output_logs = project_root / "data" / "interim" / "synthetic_worklog_dirty.parquet"
    rng = np.random.default_rng(SEED)

    progress("원본 ZIP 인덱스를 구성합니다.")
    source_index = build_source_index(raw_dir)
    progress("15,132개 원본 시퀀스를 층화 추출하고 8:1:1로 분할합니다.")
    selected = select_samples(source_index, rng)
    original_sensor = load_selected_data(selected)
    progress("센서 분포를 보존하며 반합성 값을 생성합니다.")
    clean_sensor, sensor_stats = make_sensor_semisynthetic(original_sensor, rng)
    clean_sensor, sample_meta = assign_operational_fields(clean_sensor, rng)
    progress("운영일지와 검사 후 품질일지를 생성합니다.")
    clean_logs, log_build_metrics = build_worklogs(clean_sensor, sample_meta, rng)
    progress("센서 데이터에 계획된 품질 문제를 주입합니다.")
    dirty_sensor, sensor_issue_metrics = inject_sensor_issues(clean_sensor, sensor_stats, rng)
    progress("작업일지와 연결정보에 계획된 품질 문제를 주입합니다.")
    dirty_logs, log_issue_metrics = inject_worklog_issues(clean_logs, rng)

    sensor_output_columns = [
        "sample_id", "split", "sequence_no", "timestamp", *SENSOR_COLUMNS,
        "equipment_id", "lot_id", "shift_date", "shift_code",
        "defect_detected", "defect_type", "source_type",
    ]
    worklog_output_columns = [
        "log_id", "split", "log_created_at", "related_start_at", "related_end_at",
        "shift_date", "shift_code", "equipment_id", "lot_id", "log_type",
        "record_stage", "note_text", "action_text", "downtime_minutes",
        "writer_role", "source_type",
    ]
    dirty_sensor = dirty_sensor[sensor_output_columns]
    dirty_logs = dirty_logs[worklog_output_columns]

    progress("분포·형상·저장 결과를 검증합니다.")
    validation_metrics = validate_and_save(
        original_sensor, clean_sensor[sensor_output_columns], dirty_sensor,
        dirty_logs, output_sensor, output_logs,
    )
    summary = {
        "seed": SEED,
        "source_samples": len(source_index),
        "selected_base_samples": BASE_SAMPLE_COUNT,
        "selected_base_sensor_rows": BASE_SAMPLE_COUNT * 5,
        **log_build_metrics,
        **sensor_issue_metrics,
        **log_issue_metrics,
        **validation_metrics,
        "sensor_output": str(output_sensor),
        "worklog_output": str(output_logs),
    }
    progress("생성이 완료되었습니다.")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
