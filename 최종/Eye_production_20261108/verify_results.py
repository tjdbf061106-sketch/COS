"""Verify the published Eye results and execute only Parts 5 and 6; never fit models."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import nbformat
from nbclient import NotebookClient
import pandas as pd


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional NEW verification JSON")
    args = parser.parse_args()
    if args.output and args.output.exists():
        raise FileExistsError(f"Existing output preserved: {args.output}")
    root = Path(__file__).resolve().parent
    manifest_path = root / "upload_manifest.json"
    manifest_sha = sha(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for name, item in manifest["files"].items():
        assert sha(root / name) == item["sha256"], f"Changed file: {name}"
    source = root / "data/eye_vectorized_production_673.csv"
    roles = pd.read_csv(source, usecols=["formulation_id", "evaluation_role"])
    assert len(roles) == 673 and roles.formulation_id.is_unique
    assert roles.evaluation_role.value_counts().to_dict() == {
        "existing_evaluation_pool": 606, "augmentation_train_only": 67}
    assert sha(source) == "811268d4813fcdd68e85458641bf5823239cae04e1fa871766c77e04f0221e7f"
    original = nbformat.read(root / "eye_all_experiments_v2.ipynb", as_version=4)
    nbformat.validate(original)
    counts = {}
    validation = None
    for tag, expected in [("production-results-only", 5), ("external-results-only", 10)]:
        selected = [deepcopy(c) for c in original.cells
                    if c.cell_type == "code" and tag in c.metadata.get("tags", [])]
        assert len(selected) == expected, (tag, len(selected))
        for cell in selected:
            cell.outputs = []
            cell.execution_count = None
        if tag == "external-results-only":
            # Assert the loader resolved this archive, not another source folder on the same PC.
            selected.append(nbformat.v4.new_code_cell(
                "assert EXT_ROOT.resolve() == (Path.cwd() / '외부학습/Eye_STopTox_20261108').resolve()\n"
                "assert EXT_EYE_INPUT.resolve() == (Path.cwd() / 'data/eye_vectorized_production_673.csv').resolve()\n"
                "assert len(ext_recomputed) == 36 and len(ext_paired) == 18\n"
                "assert ext_cosmetics['decision']['eligible_independent_EPA_external_samples'] == 0\n"))
        report = nbformat.v4.new_notebook(cells=selected, metadata=deepcopy(original.metadata))
        print(f"Executing {tag}: {expected} report cells", flush=True)
        NotebookClient(report, timeout=180, kernel_name="python3", allow_errors=False,
                       resources={"metadata": {"path": str(root)}}).execute()
        nbformat.validate(report)
        counts[tag] = expected
        if tag == "production-results-only":
            output = "\n".join(str(o.get("text", "")) for c in report.cells for o in c.get("outputs", []))
            marker = "NOTEBOOK_VALIDATION_JSON="
            validation = json.loads(next(line[len(marker):] for line in output.splitlines() if line.startswith(marker)))
            assert validation["outer_records_verified"] == 360 and validation["oof_sets_verified"] == 72
            assert validation["new_model_fits"] == 0
    external = root / "외부학습/Eye_STopTox_20261108"
    with zipfile.ZipFile(external / "검증체크포인트.zip") as archive:
        names = archive.namelist()
        assert len(names) == 1890 and archive.testzip() is None
        assert sum(n.endswith("/outer.json") for n in names) == 90
        assert sum("/candidates/" in n for n in names) == 1800
    for name, item in manifest["files"].items():
        assert sha(root / name) == item["sha256"], f"Verification changed a file: {name}"
    assert sha(manifest_path) == manifest_sha
    result = {
        "status": "passed", "files_sha256_verified": len(manifest["files"]),
        "eye_input_sha256": sha(source), "report_cells_executed": counts,
        "external_oof_sets_recomputed": 36, "external_paired_seed_deltas_checked": 18,
        "external_evaluation_rows_per_arm_seed": 606,
        "external_checkpoint_members": 1890,
        "independent_cosmetics_EPA_evaluation_eligible_rows": 0,
        "published_paths_resolved_inside_package": True, **validation,
    }
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
