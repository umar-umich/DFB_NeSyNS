"""Certified Evidence Records — the per-image JSONL store.

This file is what DPO, assembly, and every table read (Implementation v2 Task 7).
Append-only JSONL, one record per image, keyed by (image, detector, proposer).
Resumable: `done_keys()` lets a run skip images already certified, so a crashed
audit resumes without recomputing.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Iterator, Set

REPO = Path(__file__).resolve().parents[2]
RECORDS_DIR = REPO / "cec" / "records" / "data"


def record_key(image: str, detector: str, proposer: str) -> str:
    return hashlib.sha256(f"{image}|{detector}|{proposer}".encode()).hexdigest()[:16]


class RecordStore:
    def __init__(self, name: str):
        RECORDS_DIR.mkdir(parents=True, exist_ok=True)
        self.path = RECORDS_DIR / f"{name}.jsonl"

    def append(self, record: Dict) -> None:
        key = record_key(record["image"], record["detector"],
                         record["provenance"].get("proposer", "?"))
        record["_key"] = key
        with self.path.open("a") as fh:
            fh.write(json.dumps(record) + "\n")

    def done_keys(self) -> Set[str]:
        if not self.path.exists():
            return set()
        keys = set()
        with self.path.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    keys.add(json.loads(line).get("_key"))
        return keys

    def read(self) -> Iterator[Dict]:
        if not self.path.exists():
            return
        with self.path.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)
