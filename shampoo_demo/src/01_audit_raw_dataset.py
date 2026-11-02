"""B안 샴푸 공개 데이터셋: 원본 감사(audit)와 기초 시각화.

실행: python shampoo_demo/src/01_audit_raw_dataset.py

Jupyter/VS Code에서 `# %%` 셀 단위로도 실행할 수 있다. 원본 JSON은 읽기만 하며,
results/data_audit 아래에 집계표와 그림만 만든다. 이 단계는 모델 학습이 아니다.
"""

# %% Imports and paths
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # GUI 없이 재현 가능하게 PNG로 저장
import matplotlib.pyplot as plt

DEMO = Path(__file__).resolve().parent.parent
RAW = DEMO / "data" / "liquid_formulations" / "LiquidFormulationsDataset_2023.json"
INGREDIENT_REFERENCE = DEMO / "data" / "ingredient_reference" / "surfactants" / "BASF Surfactants Information.csv"
PROVENANCE = DEMO / "docs" / "sources" / "provenance.json"
OUT = DEMO / "results" / "data_audit"
FIG = OUT / "figures"
# These seven columns are source outcomes/metadata, not formulation inputs.
OUTCOME_COLUMNS = {"ID", "Stability_Test", "Turbidity_NTU", "Turbidity_Error", "Viscosity", "Rheology_Type", "Rheology_Data"}


# %% Strict, read-only input validation
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key!r}")
        result[key] = value
    return result


def reject_constant(token: str) -> None:
    raise ValueError(f"Non-standard JSON constant: {token}")


def load_json(path: Path) -> Any:
    """Reject duplicate keys and NaN/Infinity instead of silently accepting them."""
    with path.open(encoding="utf-8") as source:
        return json.load(source, object_pairs_hook=no_duplicates, parse_constant=reject_constant)


def expected_hash_from_provenance(provenance: dict[str, Any], relative_path: str) -> str:
    """Support the checked-in provenance format: a list of explicitly recorded sources."""
    sources = provenance.get("sources")
    if not isinstance(sources, list):
        raise KeyError("provenance.json must contain a 'sources' list")
    matches = [item for item in sources if isinstance(item, dict) and item.get("path") == relative_path]
    if len(matches) != 1 or not isinstance(matches[0].get("sha256"), str):
        raise KeyError(f"Expected exactly one SHA-256 record for {relative_path}")
    return matches[0]["sha256"].upper()


# %% Aggregate schema discovery; raw rows and IDs are never written to outputs
def type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    if isinstance(value, str):
        return "string"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "object"
    return type(value).__name__


def count_scalar_paths(value: Any, path: str, counts: Counter[str]) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            count_scalar_paths(child, f"{path}.{key}" if path else key, counts)
    elif isinstance(value, list):
        counts[f"{path}[]<{type_name(value[0]) if value else 'empty'}>"] += 1
    else:
        counts[path] += 1


def value_at(record: dict[str, Any], path: str) -> Any | None:
    current: Any = record
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current if not isinstance(current, (dict, list)) else None


def find_stability_path(paths: Counter[str], n_records: int) -> str | None:
    candidates = [p for p, n in paths.items() if "stabil" in p.lower() and n >= n_records // 4]
    return min(candidates, key=lambda p: (p.count("."), -paths[p], p)) if candidates else None


def find_ingredient_list(list_lengths: dict[str, list[int]]) -> str | None:
    candidates = [key for key in list_lengths if any(word in key.lower() for word in ("ingredient", "component", "composition", "formula"))]
    return min(candidates, key=lambda key: (-len(list_lengths[key]), key)) if candidates else None


def normalized_trade_name(value: str) -> str:
    """The reference uses ® while JSON column headers omit it; map names transparently."""
    return value.replace("®", "").strip()


def reference_ingredient_columns(record: dict[str, Any]) -> list[str]:
    """Read the surfactant-subset names documented in the reference CSV.

    The source JSON is a wide table: each ingredient is a numeric column rather
    than an embedded ingredient list. This explicit mapping is safer than
    guessing from outcome-field names.
    """
    # utf-8-sig also handles the BOM used by some exported reference CSV files.
    with INGREDIENT_REFERENCE.open(newline="", encoding="utf-8-sig") as source:
        reference_names = [normalized_trade_name(row["trade_name"]) for row in csv.DictReader(source)]
    missing = [name for name in reference_names if name not in record]
    if missing:
        raise KeyError(f"Ingredient reference names not found in JSON columns: {missing}")
    return reference_names


def composition_columns(record: dict[str, Any]) -> list[str]:
    """Map all numeric input columns after excluding documented outputs/metadata.

    This source supplies formulation inputs in wide format. The BASF reference
    CSV documents 12 surfactants; six additional commercial input columns are
    still retained as composition features rather than being silently dropped.
    """
    columns = sorted(key for key in record if key not in OUTCOME_COLUMNS)
    if len(columns) != 18:
        raise ValueError(f"Expected 18 formulation input columns after exclusions, found {len(columns)}")
    return columns


# %% Useful visualizations: stability-label balance and formulation complexity
def save_bar(counts: Counter[str], title: str, destination: Path) -> None:
    if not counts:
        return
    labels, values = zip(*sorted(counts.items(), key=lambda item: (-item[1], item[0])))
    fig, ax = plt.subplots(figsize=(max(6, min(12, len(labels) * .9)), 4.8))
    bars = ax.bar(range(len(labels)), values, color="#266DD3")
    ax.set(title=title, xlabel="Source label", ylabel="Formulation count")
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value, str(value), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def save_histogram(values: list[int], title: str, destination: Path) -> None:
    if not values:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    bins = range(min(values), max(values) + 2)
    ax.hist(values, bins=bins, color="#31A36A", edgecolor="white", align="left")
    ax.set(title=title, xlabel="Ingredients per formulation", ylabel="Formulation count")
    ax.set_xticks(list(bins))
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def save_frequency_chart(counts: Counter[str], destination: Path) -> None:
    """Show only the documented trade-name columns and their non-zero usage counts."""
    labels, values = zip(*sorted(counts.items(), key=lambda item: (-item[1], item[0])))
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(labels)), values, color="#7C4DFF")
    ax.set(yticks=range(len(labels)), yticklabels=labels, xlabel="Formulations with amount > 0", title="Ingredient usage across 812 formulations")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


# %% Reproducible audit report
def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    source_hash = sha256(RAW)
    provenance = load_json(PROVENANCE)
    expected_hash = expected_hash_from_provenance(
        provenance, "data/liquid_formulations/LiquidFormulationsDataset_2023.json"
    )
    if source_hash != expected_hash:
        raise ValueError("SHA-256 mismatch: stop before analysis and inspect the source/provenance record.")
    payload = load_json(RAW)
    if not isinstance(payload, list) or not all(isinstance(row, dict) for row in payload):
        raise TypeError("Expected a JSON list of formulation objects.")
    records: list[dict[str, Any]] = payload

    top_fields: Counter[str] = Counter()
    scalar_paths: Counter[str] = Counter()
    list_lengths: dict[str, list[int]] = defaultdict(list)
    for record in records:
        top_fields.update(record.keys())
        count_scalar_paths(record, "", scalar_paths)
        for key, value in record.items():
            if isinstance(value, list):
                list_lengths[key].append(len(value))  # zero length stays zero

    stability_path = find_stability_path(scalar_paths, len(records))
    stability_counts: Counter[str] = Counter()
    if stability_path:
        for record in records:
            value = value_at(record, stability_path)
            if value is not None and str(value).strip():
                stability_counts[str(value).strip()] += 1
        save_bar(stability_counts, f"Stability label distribution ({stability_path})", FIG / "stability_label_distribution.png")

    ingredient_key = find_ingredient_list(dict(list_lengths))
    ingredient_counts = list_lengths.get(ingredient_key, []) if ingredient_key else []
    wide_ingredient_columns = composition_columns(records[0])
    reference_columns = reference_ingredient_columns(records[0])
    # `0` means the documented ingredient was not dosed; None stays missing and
    # is counted separately, never converted to zero.
    nonzero_per_formula: list[int] = []
    # Initialize every documented input at zero so an unused ingredient is not
    # silently omitted from the frequency chart.
    ingredient_usage: Counter[str] = Counter({column: 0 for column in wide_ingredient_columns})
    ingredient_missing: Counter[str] = Counter()
    for record in records:
        used = 0
        for column in wide_ingredient_columns:
            value = record.get(column)
            if value is None:
                ingredient_missing[column] += 1
            elif isinstance(value, (int, float)) and value > 0:
                used += 1
                ingredient_usage[column] += 1
            elif not isinstance(value, (int, float)):
                raise TypeError(f"Ingredient column {column!r} contains a non-numeric value")
        nonzero_per_formula.append(used)
    save_histogram(nonzero_per_formula, "Formulation complexity (non-zero commercial ingredients)", FIG / "ingredients_per_formulation.png")
    save_frequency_chart(ingredient_usage, FIG / "ingredient_usage_frequency.png")

    schema = {
        "observational_unit": "one shampoo formulation in the source JSON",
        "record_count": len(records), "raw_sha256": source_hash, "provenance_sha256": expected_hash,
        "top_level_field_counts": dict(sorted(top_fields.items())),
        "nested_scalar_path_counts": dict(sorted(scalar_paths.items())),
        "top_level_list_length_summary": {key: {"records_with_list": len(values), "min": min(values), "max": max(values), "mean": sum(values) / len(values)} for key, values in sorted(list_lengths.items()) if values},
        "automatic_field_mapping": {"stability_candidate": stability_path, "ingredient_list_candidate": ingredient_key, "wide_ingredient_columns": wide_ingredient_columns, "reference_csv_surfactant_columns": reference_columns},
        "wide_ingredient_missing_counts": dict(ingredient_missing),
    }
    (OUT / "schema_summary.json").write_text(json.dumps(schema, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "field_completeness.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file); writer.writerow(["top_level_field", "present_records", "total_records", "completeness"])
        for key, count in sorted(top_fields.items(), key=lambda item: (-item[1], item[0])):
            writer.writerow([key, count, len(records), f"{count / len(records):.6f}"])

    report = ["# Shampoo raw-data audit", "", f"- Generated (UTC): {datetime.now(timezone.utc).isoformat(timespec='seconds')}", "- Observational unit: one shampoo formulation in the source JSON", f"- Parsed formulations: {len(records)}", f"- Raw SHA-256: `{source_hash}`", "- Provenance hash check: **PASS**", "", "## Safeguards", "", "- Raw input is read-only; no row deletion, imputation, or composition renormalization occurred.", "- Numeric zero and an absent field remain distinct.", "- No split, threshold selection, or predictive performance result is produced here.", "", "## Top-level field availability", "", "| Field | Present records | Completeness |", "|---|---:|---:|"]
    report += [f"| `{key}` | {count} | {count / len(records):.1%} |" for key, count in sorted(top_fields.items(), key=lambda item: (-item[1], item[0]))]
    report += ["", "## Automatic mapping candidates", "", f"- Stability: `{stability_path}`" if stability_path else "- Stability: not found; manually map before modelling.", "- Inputs: 18 wide numeric formulation columns after excluding seven documented outcomes/metadata columns.", f"- BASF reference CSV: {len(reference_columns)} of those columns are explicitly documented surfactants; the other six remain composition inputs and are retained.", f"- Nested ingredient list: `{ingredient_key}`" if ingredient_key else "- Nested ingredient list: absent (expected for this wide-format source).", "- Figures: `figures/stability_label_distribution.png`, `figures/ingredients_per_formulation.png`, `figures/ingredient_usage_frequency.png`.", "", "## Nested scalar field coverage (top 30)", "", "| Field path | Present records |", "|---|---:|"]
    report += [f"| `{path}` | {count} |" for path, count in scalar_paths.most_common(30)]
    report += ["", "## Next decision", "", "Confirm mappings for stability, turbidity, viscosity, ingredient identity, and input quantity. Then define formulation-combination groups before group-aware model evaluation.", ""]
    (OUT / "audit_summary.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Audit complete: {OUT}")
    print(f"Formulations: {len(records)} | SHA-256 verified: True")
    print(f"Stability candidate: {stability_path or 'not found'}")
    print(f"Wide formulation inputs: {len(wide_ingredient_columns)} (reference-mapped surfactants: {len(reference_columns)})")
    print(f"Nested ingredient-list candidate: {ingredient_key or 'not applicable'}")


if __name__ == "__main__":
    main()
