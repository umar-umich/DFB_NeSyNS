"""
interpretability/data_export.py
================================
Shared CSV / NPZ / JSON dump helpers used by every analyzer to export
the underlying numerical data for each PNG it produces. The paper authors
re-plot offline (matplotlib / R / Cytoscape), so these files must be
sufficient to fully reconstruct the figure without re-running inference.

Conventions (enforced by ``analyzer_metadata`` and the dump helpers):

* CSV — tabular data only. Every CSV gets a header row.
* NPZ — raw per-sample distributions, named arrays only (no anonymous
  ``arr_0``).
* JSON — scalar metadata. Always includes ``n_samples`` and
  ``dataset_name``. Any scalar shown on the plot (ECE, AURC, ρ, threshold,
  …) is added by the caller.

File naming: same basename as the PNG, different extension. Example::

    reliability_diagram.png
    reliability_diagram.csv          (per-bin table)
    reliability_diagram.json         (ECE, n_bins, ...)
    reliability_per_sample.csv       (per-sample re-binnable view)

Date: 2026-05-04 — added as part of the Goal-1 data-export pass.
"""
from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np


def _to_jsonable(v: Any) -> Any:
    """Coerce numpy scalars / arrays into JSON-serialisable types."""
    if v is None:
        return None
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, dict):
        return {str(k): _to_jsonable(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_to_jsonable(x) for x in v]
    return v


def analyzer_metadata(
    n_samples: int,
    dataset_name: Optional[str] = None,
    **extra: Any,
) -> Dict[str, Any]:
    """Build the standard JSON-metadata block.

    Always emits ``n_samples`` and ``dataset_name`` (the two universally
    required keys); ``extra`` is merged on top so per-analyzer scalars
    (ECE, AURC, ρ, …) sit alongside.
    """
    out: Dict[str, Any] = {
        'n_samples': int(n_samples),
        'dataset_name': str(dataset_name) if dataset_name else 'unknown',
    }
    out.update({k: _to_jsonable(v) for k, v in extra.items()})
    return out


def dump_json(meta: Mapping[str, Any], save_path: str) -> None:
    """Write ``meta`` to ``save_path`` as pretty-printed JSON."""
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump({k: _to_jsonable(v) for k, v in meta.items()}, f, indent=2)


def dump_csv(
    rows: Iterable[Mapping[str, Any]],
    columns: Sequence[str],
    save_path: str,
) -> None:
    """Write ``rows`` to ``save_path`` as a CSV with a header row.

    Every row must contain every column in ``columns`` (None for missing
    fields is OK and renders as the empty string).
    """
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    with open(save_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(columns))
        w.writeheader()
        for r in rows:
            w.writerow({c: _format_cell(r.get(c)) for c in columns})


def dump_csv_columns(
    columns_data: Mapping[str, Sequence[Any]],
    save_path: str,
) -> None:
    """Column-major CSV dump. Convenience wrapper for arrays of equal
    length keyed by column name. Header order is the dict insertion order.
    """
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    cols = list(columns_data.keys())
    if not cols:
        with open(save_path, 'w') as f:
            f.write('')
        return
    n = len(next(iter(columns_data.values())))
    with open(save_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols)
        for i in range(n):
            w.writerow([_format_cell(columns_data[c][i]) for c in cols])


def dump_npz(
    arrays: Mapping[str, np.ndarray],
    save_path: str,
) -> None:
    """Save named numpy arrays. Refuses to dump anonymous ``arr_0`` keys —
    the spec requires every array to be named.
    """
    if not arrays:
        return
    bad = [k for k in arrays if not k or k.startswith('arr_')]
    if bad:
        raise ValueError(
            f'dump_npz refuses anonymous keys (got {bad}); '
            f'every NPZ array must be explicitly named per the spec.')
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    np.savez_compressed(save_path, **{k: np.asarray(v) for k, v in arrays.items()})


def _format_cell(v: Any) -> Any:
    """Render numpy scalars as plain Python; pass everything else through."""
    if v is None:
        return ''
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def sibling_path(png_path: str, new_ext: str) -> str:
    """``foo/bar.png`` + ``.csv`` → ``foo/bar.csv``."""
    base, _ = os.path.splitext(png_path)
    if not new_ext.startswith('.'):
        new_ext = '.' + new_ext
    return base + new_ext


def with_basename(png_path: str, new_basename: str) -> str:
    """Build a sibling file with a different basename in the same dir.
    ``foo/reliability.png`` + ``reliability_per_sample.csv`` →
    ``foo/reliability_per_sample.csv``.
    """
    return os.path.join(os.path.dirname(png_path) or '.', new_basename)
