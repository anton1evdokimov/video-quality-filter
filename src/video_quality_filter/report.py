"""Write metadata to JSONL and Parquet. Source videos are not touched."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def write_reports(output_dir: Path, records: list[dict]) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "results.jsonl"
    parquet_path = output_dir / "results.parquet"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    if records:
        table = pa.Table.from_pylist(records)
    else:
        table = pa.table({"video_id": pa.array([], type=pa.string())})
    pq.write_table(table, parquet_path)
    return jsonl_path, parquet_path
