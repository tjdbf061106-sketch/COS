from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(clean_json(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path, frame):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def interpolate_viscosity(shear, viscosity, target=100.0):
    x = np.asarray(shear, dtype=float)
    y = np.asarray(viscosity, dtype=float)
    if len(x) != len(y) or len(x) < 2:
        raise ValueError("Flow curve must contain paired values")
    if not (np.isfinite(x).all() and np.isfinite(y).all() and (x > 0).all() and (y > 0).all()):
        raise ValueError("Log interpolation needs finite positive measurements")
    order = np.argsort(x)
    x, y = x[order], y[order]
    if (np.diff(x) <= 0).any():
        raise ValueError("Duplicate shear rates need explicit curation")
    if not x[0] <= target <= x[-1]:
        return np.nan
    return float(10 ** np.interp(np.log10(target), np.log10(x), np.log10(y)))


def ingredient_group(row, columns):
    return " | ".join(sorted(c for c in columns if float(row[c]) > 0)) or "NONE"


def load_inputs():
    columns = read_json(ROOT / "data/feature_names.json")
    data = pd.read_csv(ROOT / "data/formulations.csv")
    return data, columns
