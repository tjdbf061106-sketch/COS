"""Canonical eye production commands: evaluate, train, predict and inspect."""

import argparse
import json
from pathlib import Path
import sys

import production_pipeline_v2 as V


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    evaluate = sub.add_parser(
        "evaluate",
        help="Audited 606-row production CV; completed checkpoints are reused",
    )
    evaluate.add_argument("--workers", type=int, default=6)
    evaluate.add_argument(
        "--models", nargs="+", choices=V.MODELS, default=list(V.MODELS)
    )
    evaluate.add_argument(
        "--prepare-only",
        action="store_true",
        help="Validate and prepare configuration; do not train",
    )
    train = sub.add_parser(
        "train",
        help="Fit one explicit model/arm; calibration OOF is not performance evaluation",
    )
    train.add_argument("--label", choices=V.LABELS, required=True)
    train.add_argument("--model", choices=V.MODELS, required=True)
    train.add_argument("--arm", choices=["baseline", "concentration"], required=True)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument(
        "--output",
        type=Path,
        help="New directory or the same run to resume; existing different files are preserved",
    )
    predict = sub.add_parser(
        "predict", help="Predict from raw vectors/metadata using a trusted saved bundle"
    )
    predict.add_argument("--model-dir", type=Path, required=True)
    predict.add_argument("--input", type=Path, required=True)
    predict.add_argument("--output", type=Path, required=True)
    inspect = sub.add_parser(
        "inspect", help="Verify bundle checksums/environment and show its input schema"
    )
    inspect.add_argument("--model-dir", type=Path, required=True)
    return p


def predict_csv(model_dir, source, output):
    source, output = Path(source), Path(output)
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    for path in [output, manifest_path]:
        if path.exists():
            raise FileExistsError(f"Existing output preserved: {path}")
    before = V.file_hash(source)
    bundle = V.load_model(model_dir)
    predictions = bundle.predict(V.read_raw_csv(source))
    V.require(V.file_hash(source) == before, "Inference input changed while predicting")
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output, index=False, encoding="utf-8-sig", mode="x")
    V.write_json_new(
        manifest_path,
        {
            "version": V.VERSION,
            "input": str(source.resolve()),
            "input_sha256": before,
            "model_manifest_sha256": V.file_hash(
                Path(model_dir) / "model_manifest.json"
            ),
            "model_sha256": V.file_hash(Path(model_dir) / "model.joblib"),
            "output_sha256": V.file_hash(output),
            "rows": len(predictions),
            "label": bundle.label,
            "threshold": bundle.threshold,
            "score_kind": bundle.describe()["score_kind"],
            "preprocessing_refitted": False,
        },
    )
    print(f"Predictions: {output.resolve()}; rows={len(predictions)}", flush=True)


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "evaluate":
        with V.training_lock():
            V.evaluate(
                workers=args.workers, models=args.models, prepare_only=args.prepare_only
            )
    elif args.command == "train":
        with V.training_lock():
            output = V.train_final(
                label=args.label,
                model=args.model,
                arm=args.arm,
                seed=args.seed,
                output=args.output,
            )
        print(f"Candidate bundle (not deployed): {output.resolve()}", flush=True)
    elif args.command == "predict":
        predict_csv(args.model_dir, args.input, args.output)
    else:
        print(
            json.dumps(
                V.load_model(args.model_dir).describe(), ensure_ascii=False, indent=2
            )
        )


if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    main()
