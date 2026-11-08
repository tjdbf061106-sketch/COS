"""Verify the published Eye archive and execute only the five report cells."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import nbformat
from nbclient import NotebookClient
import pandas as pd


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional NEW JSON verification record")
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError(f"Existing output preserved: {args.output}")
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "upload_manifest.json").read_text(encoding="utf-8"))
    for name, item in manifest["files"].items():
        assert sha(root / name) == item["sha256"], f"Changed file: {name}"
    source = root / "data/eye_vectorized_production_673.csv"
    roles = pd.read_csv(source, usecols=["formulation_id", "evaluation_role"])
    assert len(roles) == 673 and roles.formulation_id.is_unique
    assert roles.evaluation_role.value_counts().to_dict() == {
        "existing_evaluation_pool": 606, "augmentation_train_only": 67}
    config = json.loads((root / "claude/results/concentration_production/f91f3c88787bec6b/config.json").read_text(encoding="utf-8"))
    assert sha(source) == config["payload"]["input_sha256"]
    original = nbformat.read(root / "eye_all_experiments_v2.ipynb", as_version=4)
    nbformat.validate(original)
    selected = [cell for cell in original.cells
                if cell.cell_type == "code" and "production-results-only" in cell.metadata.get("tags", [])]
    assert len(selected) == 5, "Expected five production-only report cells"
    for cell in selected:
        cell.outputs = []
        cell.execution_count = None
    report = nbformat.v4.new_notebook(cells=selected, metadata=original.metadata)
    NotebookClient(report, timeout=120, kernel_name="python3",
                   resources={"metadata": {"path": str(root)}}).execute()
    nbformat.validate(report)
    output = "\n".join(str(o.get("text", "")) for c in report.cells for o in c.get("outputs", []))
    marker = "NOTEBOOK_VALIDATION_JSON="
    validation = json.loads(next(line[len(marker):] for line in output.splitlines() if line.startswith(marker)))
    assert validation["outer_records_verified"] == 360 and validation["oof_sets_verified"] == 72
    assert validation["new_model_fits"] == 0
    for name, item in manifest["files"].items():
        assert sha(root / name) == item["sha256"], f"Verification changed a file: {name}"
    result = {"status": "passed", "files_sha256_verified": len(manifest["files"]),
              "eye_input_sha256": sha(source), "report_cells_executed": 5, **validation}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    for stream in [sys.stdout, sys.stderr]:
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    main()
