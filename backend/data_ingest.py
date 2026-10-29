"""Validate uploaded traffic rate rows and sample reproducible vehicle arrival schedules."""

from __future__ import annotations

import csv
import io
import math
import re
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

MAX_DATASET_ROWS = 50000
MAX_SIMULATION_SECONDS = 3600.0
MAX_ARRIVAL_RATE = 0.5
DIRECTIONS = ("N", "S", "E", "W")


def generate_synthetic_dataset(seed: int = 42) -> list[dict]:
    """Create seeded five-minute demand rows with a morning north-south peak."""
    rng = np.random.default_rng(seed)
    dataset = []
    for start_time in range(0, 3600, 300):
        morning_peak = start_time < 1800
        for direction in DIRECTIONS:
            if morning_peak and direction in ("N", "S"):
                rate = float(rng.uniform(0.08, 0.12))
            else:
                rate = float(rng.uniform(0.04, 0.07))
            dataset.append({"time": float(start_time), "direction": direction, "rate": round(rate, 4)})
    return dataset


def _parse_time_string(val: str) -> float | None:
    """Parse float seconds, HH:MM:SS, or HH:MM:SS.mmm into total seconds."""
    val = val.strip()
    if not val:
        return None
    time_match = re.search(r"(\d{1,2}):(\d{2}):(\d{2})(?:\.(\d+))?", val)
    if time_match:
        hours = int(time_match.group(1))
        minutes = int(time_match.group(2))
        seconds = int(time_match.group(3))
        frac = float("0." + time_match.group(4)) if time_match.group(4) else 0.0
        return hours * 3600.0 + minutes * 60.0 + seconds + frac
    try:
        return float(val)
    except ValueError:
        return None


def _map_direction_string(dir_str: str) -> str | None:
    """Map flexible direction names (North, NB, N-S, Direction 13, route_12, etc.) to N, S, E, or W."""
    cleaned = str(dir_str).strip().upper()
    if not cleaned:
        return None
    # Direct match
    if cleaned in DIRECTIONS:
        return cleaned

    # Check for Bremen numbered movement: Direction 1..14, Dir 1..14, route_12
    num_match = re.search(r"(?:DIRECTION|DIR|ROUTE_?)\s*(\d+)", cleaned)
    if num_match:
        raw_num = int(num_match.group(1))
        # SUMO route_0..route_13 (0-indexed) to 1-indexed direction
        num = raw_num + 1 if "ROUTE" in cleaned and raw_num < 15 else raw_num
        # Bremen 4-arm intersection mapping:
        # Approach 1 (North): Directions 1, 2, 3, 15, 16, 17
        # Approach 2 (East): Directions 11, 12, 13, 14, 18, 19
        # Approach 3 (West): Directions 4, 5, 6, 7
        # Approach 4 (South): Directions 8, 9, 10
        if num in (1, 2, 3, 15, 16, 17):
            return "N"
        if num in (4, 5, 6, 7):
            return "W"
        if num in (8, 9, 10):
            return "S"
        if num in (11, 12, 13, 14, 18, 19):
            return "E"
        return ("N", "S", "E", "W")[(num - 1) % 4]

    # Bare integer: "1" -> "N", "2" -> "S", "3" -> "E", "4" -> "W"
    if cleaned.isdigit():
        num = int(cleaned)
        if num in (1, 2, 3, 15, 16, 17):
            return "N"
        if num in (4, 5, 6, 7):
            return "W"
        if num in (8, 9, 10):
            return "S"
        if num in (11, 12, 13, 14, 18, 19):
            return "E"
        return ("N", "S", "E", "W")[(num - 1) % 4]

    # Common prefixes and turning movements
    if cleaned.startswith("N") or cleaned in ("NORTH", "NORTHBOUND", "NORTH-SOUTH", "N-S", "N->S"):
        return "N"
    if cleaned.startswith("S") or cleaned in ("SOUTH", "SOUTHBOUND", "SOUTH-NORTH", "S-N", "S->N"):
        return "S"
    if cleaned.startswith("E") or cleaned in ("EAST", "EASTBOUND", "EAST-WEST", "E-W", "E->W"):
        return "E"
    if cleaned.startswith("W") or cleaned in ("WEST", "WESTBOUND", "WEST-EAST", "W-E", "W->E"):
        return "W"
    return None


def _build_dataset_from_raw_records(raw_records: list[tuple[float, str, float]], is_prebinned_rate: bool = False) -> list[dict]:
    """Convert raw or binned traffic entries into standard piecewise 5-minute interval rows."""
    if not raw_records:
        raise ValueError("Dataset contains no recognizable traffic entries.")

    if is_prebinned_rate:
        dataset = [{"time": float(t), "direction": d, "rate": float(r)} for t, d, r in raw_records]
        return sorted(dataset, key=lambda row: (row["direction"], row["time"]))

    # Aggregate raw vehicle passages into standard 300-second windows across 3600 seconds
    min_time = min(t for t, _, _ in raw_records)
    window_counts: dict[tuple[int, str], float] = {}
    for t, d, count in raw_records:
        rel_time = t - min_time
        if rel_time < 0 or rel_time >= MAX_SIMULATION_SECONDS:
            continue
        window_idx = int(rel_time // 300.0)
        key = (window_idx, d)
        window_counts[key] = window_counts.get(key, 0.0) + (count if count > 0 else 1.0)

    dataset = []
    for window_idx in range(12):  # 12 x 300s = 3600s
        time_sec = float(window_idx * 300)
        for direction in DIRECTIONS:
            count = window_counts.get((window_idx, direction), 0.0)
            if count > 0:
                rate = round(min(MAX_ARRIVAL_RATE, max(0.01, count / 300.0)), 4)
            else:
                rate = 0.03  # default baseline flow
            dataset.append({"time": time_sec, "direction": direction, "rate": rate})

    return sorted(dataset, key=lambda row: (row["direction"], row["time"]))


def load_from_ods(filepath_or_bytes) -> list[dict]:
    """Read OpenDocument Spreadsheet (.ods) files, such as the official CN+ Bremen release."""
    if isinstance(filepath_or_bytes, (bytes, bytearray)):
        zfile = zipfile.ZipFile(io.BytesIO(filepath_or_bytes))
    else:
        zfile = zipfile.ZipFile(filepath_or_bytes)

    with zfile:
        try:
            content_xml = zfile.read("content.xml")
        except KeyError:
            raise ValueError("Invalid ODS file: content.xml not found inside archive.")

    root = ET.fromstring(content_xml)
    ns_table = "{urn:oasis:names:tc:opendocument:xmlns:table:1.0}"
    rows = root.findall(".//" + ns_table + "table-row")
    if not rows:
        raise ValueError("ODS spreadsheet contains no data rows.")

    header: list[str] | None = None
    time_idx = 1
    dir_idx = 2
    count_idx = 3

    raw_records: list[tuple[float, str, float]] = []
    for r in rows:
        cells: list[str] = []
        for c in r.findall(".//" + ns_table + "table-cell"):
            rep = int(c.attrib.get(ns_table + "number-columns-repeated", 1))
            txt = "".join(c.itertext()).strip()
            cells.extend([txt] * min(rep, 20))
        if not any(cells):
            continue

        if header is None:
            header = [c.lower() for c in cells]
            for idx, col in enumerate(header):
                if any(k in col for k in ("time", "crossing", "second")):
                    time_idx = idx
                elif any(k in col for k in ("direction", "dir", "approach")):
                    dir_idx = idx
                elif any(k in col for k in ("number", "count", "vehicle", "volume")):
                    count_idx = idx
            continue

        if len(cells) <= max(time_idx, dir_idx):
            continue

        t_val = _parse_time_string(cells[time_idx])
        if t_val is None:
            continue
        d_val = _map_direction_string(cells[dir_idx])
        if d_val is None:
            continue

        count_val = 1.0
        if len(cells) > count_idx and cells[count_idx]:
            try:
                count_val = float(cells[count_idx])
            except ValueError:
                count_val = 1.0

        raw_records.append((t_val, d_val, count_val))
        if len(raw_records) >= MAX_DATASET_ROWS:
            break

    return _build_dataset_from_raw_records(raw_records)


def load_from_sumo_xml(filepath_or_bytes) -> list[dict]:
    """Parse SUMO scenario route XML files (<routes><vehicle depart=... route=... /></routes>)."""
    if isinstance(filepath_or_bytes, (bytes, bytearray)):
        source = io.BytesIO(filepath_or_bytes)
    else:
        source = open(filepath_or_bytes, "rb")

    try:
        tree = ET.parse(source)
        root = tree.getroot()
        raw_records: list[tuple[float, str, float]] = []
        for v in root.findall(".//vehicle"):
            depart = v.attrib.get("depart")
            route = v.attrib.get("route", "")
            if depart is not None:
                try:
                    t_val = float(depart)
                    d_val = _map_direction_string(route) or "N"
                    raw_records.append((t_val, d_val, 1.0))
                except ValueError:
                    continue
        if not raw_records:
            raise ValueError("No vehicle departures found in SUMO route XML.")
        return _build_dataset_from_raw_records(raw_records)
    finally:
        if not isinstance(filepath_or_bytes, (bytes, bytearray)):
            source.close()


def load_from_zip(filepath_or_bytes) -> list[dict]:
    """Extract and parse traffic datasets from ZIP archives (supporting .ods, .csv, and SUMO .rou.xml)."""
    if isinstance(filepath_or_bytes, (bytes, bytearray)):
        zfile = zipfile.ZipFile(io.BytesIO(filepath_or_bytes))
    else:
        zfile = zipfile.ZipFile(filepath_or_bytes)

    with zfile:
        names = zfile.namelist()
        # 1. Prefer .ods spreadsheet (e.g. CN+ Dataset.ods)
        ods_candidates = [n for n in names if n.lower().endswith(".ods") and not n.startswith("__MACOSX")]
        if ods_candidates:
            return load_from_ods(zfile.read(ods_candidates[0]))

        # 2. Prefer .csv files
        csv_candidates = [n for n in names if n.lower().endswith(".csv") and not n.startswith("__MACOSX")]
        if csv_candidates:
            return load_from_csv(zfile.read(csv_candidates[0]))

        # 3. Prefer SUMO .rou.xml route files
        xml_candidates = [n for n in names if n.lower().endswith(".rou.xml") and not n.startswith("__MACOSX")]
        if xml_candidates:
            return load_from_sumo_xml(zfile.read(xml_candidates[0]))

    raise ValueError("ZIP archive contains no recognized traffic dataset (.ods, .csv, or .rou.xml).")


def load_traffic_dataset(filepath_or_bytes, filename: str | None = None) -> list[dict]:
    """Auto-detect format (ZIP, ODS, SUMO XML, or CSV) and parse into standard traffic dataset."""
    name = (filename or (filepath_or_bytes if isinstance(filepath_or_bytes, str) else "")).lower()

    if isinstance(filepath_or_bytes, (bytes, bytearray)):
        magic = bytes(filepath_or_bytes[:4])
        if magic.startswith(b"PK\x03\x04"):
            if name.endswith(".ods"):
                return load_from_ods(filepath_or_bytes)
            try:
                return load_from_ods(filepath_or_bytes)
            except Exception:
                return load_from_zip(filepath_or_bytes)
        if magic.startswith(b"<?xm") or magic.startswith(b"<rou") or name.endswith(".xml") or name.endswith(".rou.xml"):
            return load_from_sumo_xml(filepath_or_bytes)
        return load_from_csv(filepath_or_bytes)

    if name.endswith(".zip"):
        return load_from_zip(filepath_or_bytes)
    if name.endswith(".ods"):
        return load_from_ods(filepath_or_bytes)
    if name.endswith(".xml") or name.endswith(".rou.xml"):
        return load_from_sumo_xml(filepath_or_bytes)
    return load_from_csv(filepath_or_bytes)


def load_from_csv(filepath_or_bytes) -> list[dict]:
    """Read a CSV file supporting standard TrafficLearner schedules or datasets like CN+ Bremen."""
    if isinstance(filepath_or_bytes, (bytes, bytearray)):
        magic = bytes(filepath_or_bytes[:4])
        if magic.startswith(b"PK\x03\x04") or magic.startswith(b"<?xm"):
            return load_traffic_dataset(filepath_or_bytes)

        for enc in ("utf-8-sig", "utf-8", "latin-1", "cp1252"):
            try:
                text_content = filepath_or_bytes.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise ValueError("CSV encoding not supported. Please save the file as UTF-8.")
        source = io.StringIO(text_content)
        close_source = False
    else:
        path_str = str(filepath_or_bytes).lower()
        if path_str.endswith(".zip") or path_str.endswith(".ods") or path_str.endswith(".xml"):
            return load_traffic_dataset(filepath_or_bytes)

        source = open(filepath_or_bytes, "r", encoding="utf-8-sig", newline="")
        close_source = True

    try:
        sample_text = source.read(4096)
        source.seek(0)

        # Detect delimiter: comma, semicolon, tab, or default comma
        delimiter = ","
        for delim in (",", ";", "\t"):
            if delim in sample_text and sample_text.count(delim) > sample_text.count(delimiter):
                delimiter = delim

        reader = csv.DictReader(source, delimiter=delimiter)
        raw_headers = reader.fieldnames or []
        normalized_headers = {h.strip().lower(): h for h in raw_headers}

        # Match Time column
        time_col = None
        for cand in (
            "time_seconds", "time", "timestamp", "crossing timing", "crossing_timing",
            "crossing_time", "crossing timing (s)", "time(s)", "seconds", "datetime"
        ):
            if cand in normalized_headers:
                time_col = normalized_headers[cand]
                break

        # Match Direction column
        dir_col = None
        for cand in (
            "direction", "vehicle direction", "vehicle_direction", "dir", "approach",
            "movement", "leg", "arm"
        ):
            if cand in normalized_headers:
                dir_col = normalized_headers[cand]
                break

        # Match Rate / Count column
        rate_col = None
        for cand in (
            "arrival_rate", "rate", "vehicle_count", "vehicle count", "number of vehicles",
            "number_of_vehicles", "number", "count", "volume", "vehicles", "flow"
        ):
            if cand in normalized_headers:
                rate_col = normalized_headers[cand]
                break

        if not time_col or not dir_col:
            found = ", ".join(repr(h) for h in raw_headers)
            raise ValueError(
                f"CSV headers must include a time column and direction column. Found: [{found}]. "
                "Accepted time columns: time_seconds, time, crossing timing, timestamp. "
                "Accepted direction columns: direction, vehicle direction, approach."
            )

        raw_records = []
        for row_number, original_row in enumerate(reader, start=2):
            if row_number > MAX_DATASET_ROWS + 1:
                break

            time_raw = original_row.get(time_col)
            dir_raw = original_row.get(dir_col)
            if time_raw is None or dir_raw is None:
                continue

            t_val = _parse_time_string(str(time_raw))
            if t_val is None:
                continue

            d_val = _map_direction_string(str(dir_raw))
            if d_val is None:
                continue

            # Check rate or count
            rate_val = None
            if rate_col and original_row.get(rate_col) not in (None, ""):
                try:
                    parsed_val = float(original_row[rate_col])
                    if "rate" in rate_col.lower() and parsed_val <= 1.0:
                        rate_val = parsed_val
                    else:
                        rate_val = parsed_val
                except ValueError:
                    rate_val = 1.0
            else:
                rate_val = 1.0

            raw_records.append((t_val, d_val, rate_val))

    finally:
        if close_source:
            source.close()

    is_prebinned_rate = (
        len(raw_records) <= 120
        and rate_col is not None
        and "rate" in rate_col.lower()
        and all(0 <= r <= MAX_ARRIVAL_RATE for _, _, r in raw_records)
        and all(0 <= t <= MAX_SIMULATION_SECONDS for t, _, _ in raw_records)
    )

    return _build_dataset_from_raw_records(raw_records, is_prebinned_rate=is_prebinned_rate)


def dataset_to_arrival_rates(dataset: list[dict], window_start: float, window_end: float) -> dict[str, float]:
    """Average matching direction rates for rows starting inside the requested time window."""
    totals = {direction: 0.0 for direction in DIRECTIONS}
    counts = {direction: 0 for direction in DIRECTIONS}
    for row in dataset:
        if window_start <= row["time"] < window_end and row["direction"] in totals:
            totals[row["direction"]] += float(row["rate"])
            counts[row["direction"]] += 1
    return {direction: totals[direction] / counts[direction] if counts[direction] else 0.0 for direction in DIRECTIONS}


def dataset_to_full_arrivals(dataset: list[dict], seed: int = 42,
                             duration: float = MAX_SIMULATION_SECONDS) -> dict[str, list[float]]:
    """Sample each piecewise-constant demand segment into sorted seeded vehicle timestamps."""
    if not math.isfinite(float(duration)) or duration < 0 or duration > MAX_SIMULATION_SECONDS:
        raise ValueError("duration must be between zero and 3600 seconds")
    rng = np.random.default_rng(seed)
    arrivals = {direction: [] for direction in DIRECTIONS}
    grouped = {direction: [] for direction in DIRECTIONS}
    for row in dataset:
        direction = row["direction"]
        start_time = float(row["time"])
        rate = float(row["rate"])
        if direction not in grouped or not math.isfinite(start_time) or not 0 <= start_time <= duration:
            raise ValueError("traffic rows must use a valid direction and in-range time")
        if not math.isfinite(rate) or not 0 <= rate <= MAX_ARRIVAL_RATE:
            raise ValueError(f"arrival rate must be between zero and {MAX_ARRIVAL_RATE} vehicles per second")
        grouped[direction].append((start_time, rate))

    for direction, segments in grouped.items():
        ordered_segments = sorted(segments)
        for index, (start_time, rate) in enumerate(ordered_segments):
            segment_end = ordered_segments[index + 1][0] if index + 1 < len(ordered_segments) else duration
            if rate <= 0 or segment_end <= start_time:
                continue
            window = segment_end - start_time
            expected_count = int(rng.poisson(rate * window))
            if expected_count > 0:
                timestamps = start_time + rng.uniform(0.0, window, size=expected_count)
                arrivals[direction].extend(float(ts) for ts in timestamps)
        arrivals[direction].sort()
    return arrivals
