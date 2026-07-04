"""
export_tensorrt.py
------------------
Export the project's custom 10-class YOLO detection weights (.pt) to a
TensorRT engine (.engine) for use with infer.py and benchmark_speed.py.

No pretrained-model fallback is used. If the configured project weights are
missing, provide a compatible custom model with --weights.

Examples
--------
# Recommended: FP16, static batch=1, 1280x1280
python export_tensorrt.py

# Explicit weights and image size
python export_tensorrt.py --weights runs/detect/best.pt --imgsz 1280

# FP32
python export_tensorrt.py --fp32

# INT8 (requires a calibration dataset YAML)
python export_tensorrt.py --int8 --data datasets/VisDrone-YOLO/dataset_mot.yaml

# Dynamic input shapes; --batch is the maximum optimized batch size
python export_tensorrt.py --dynamic --batch 8

# Select a GPU
python export_tensorrt.py --device 1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from ultralytics import YOLO

try:
    # Available in current and many earlier Ultralytics releases.
    from ultralytics.cfg import DEFAULT_CFG_DICT
except ImportError:  # pragma: no cover - compatibility with older releases
    DEFAULT_CFG_DICT = {}


_DEFAULT_WEIGHTS = "runs/detect/train/weights/best.pt"

# This order must match infer.py's YOLO_TO_VISDRONE mapping.
EXPECTED_CLASS_NAMES = [
    "pedestrian",
    "people",
    "bicycle",
    "car",
    "van",
    "truck",
    "tricycle",
    "awning-tricycle",
    "bus",
    "motor",
]


def resolve_weights(path: str | None) -> Path:
    """Resolve a custom .pt weights file without any generic-model fallback."""
    weights = Path(path).expanduser() if path else Path(_DEFAULT_WEIGHTS)

    if not weights.is_file():
        if path:
            sys.exit(f"[ERROR] Weights file not found: {weights}")
        sys.exit(
            f"[ERROR] Default project weights not found: '{_DEFAULT_WEIGHTS}'.\n"
            "        Pass --weights with a compatible custom 10-class .pt model."
        )

    if weights.suffix.lower() != ".pt":
        sys.exit(f"[ERROR] --weights must point to a PyTorch .pt file: {weights}")

    return weights.resolve()


def normalize_model_names(raw_names: object) -> list[str]:
    """Convert Ultralytics model.names into a deterministic ordered list."""
    if isinstance(raw_names, dict):
        try:
            keys = sorted(raw_names, key=lambda key: int(key))
        except (TypeError, ValueError):
            keys = sorted(raw_names, key=str)
        return [str(raw_names[key]).strip() for key in keys]

    if isinstance(raw_names, (list, tuple)):
        return [str(name).strip() for name in raw_names]

    return []


def validate_class_names(names: Sequence[str]) -> None:
    """Stop export if class indices do not match the tracking pipeline."""
    actual = list(names)
    if actual == EXPECTED_CLASS_NAMES:
        return

    expected_text = ", ".join(f"{i}:{name}" for i, name in enumerate(EXPECTED_CLASS_NAMES))
    actual_text = ", ".join(f"{i}:{name}" for i, name in enumerate(actual)) or "<missing>"
    sys.exit(
        "[ERROR] Model class names/order do not match the project's taxonomy.\n"
        f"        Expected: {expected_text}\n"
        f"        Actual  : {actual_text}\n"
        "        Export stopped to prevent incorrect class IDs during tracking."
    )


def save_names_sidecar(engine_path: Path, names: Sequence[str]) -> Path:
    """Write class names to <engine_stem>.names for infer.py."""
    names_path = engine_path.with_suffix(".names")
    names_path.write_text("\n".join(names) + "\n", encoding="utf-8")
    return names_path


def validate_export_args(args: argparse.Namespace) -> None:
    if args.imgsz <= 0:
        sys.exit("[ERROR] --imgsz must be greater than 0.")
    if args.batch <= 0:
        sys.exit("[ERROR] --batch must be greater than 0.")
    if args.workspace is not None and args.workspace <= 0:
        sys.exit("[ERROR] --workspace must be greater than 0, or omit it.")
    if args.int8:
        if not args.data:
            sys.exit(
                "[ERROR] INT8 export requires a calibration dataset YAML.\n"
                "        Pass --data datasets/VisDrone-YOLO/dataset_mot.yaml."
            )
        data_path = Path(args.data).expanduser()
        if not data_path.is_file():
            sys.exit(f"[ERROR] Calibration dataset YAML not found: {data_path}")
    elif args.data:
        print("[WARNING] --data is ignored unless --int8 is selected.")

    if args.dynamic and args.batch == 1:
        print(
            "[WARNING] Dynamic TensorRT export with batch=1 only supports a maximum "
            "batch size of 1. Use --batch > 1 only when variable/multi-image batches "
            "are required."
        )


def _precision_kwargs(quantize: int | None) -> dict[str, object]:
    """
    Build precision arguments for current and legacy Ultralytics releases.

    Current releases use quantize={16, 8, None}. Older releases used
    half=True and int8=True.
    """
    if "quantize" in DEFAULT_CFG_DICT:
        return {"quantize": quantize}

    print(
        "[WARNING] This Ultralytics version uses the legacy export precision API. "
        "Consider upgrading to a current release."
    )
    return {
        "half": quantize == 16,
        "int8": quantize == 8,
    }


def export_engine(
    weights: Path,
    imgsz: int,
    quantize: int | None,
    dynamic: bool,
    data: str | None,
    workspace: float | None,
    device: str,
    simplify: bool,
    batch: int,
) -> Path:
    precision = {None: "FP32", 16: "FP16", 8: "INT8"}[quantize]

    print(f"\n[export] Source weights : {weights}")
    print(f"[export] Image size     : {imgsz} px")
    print(f"[export] Precision      : {precision}")
    print(f"[export] Dynamic shapes : {dynamic}")
    print(f"[export] Max batch      : {batch}")
    print(f"[export] TRT workspace  : {workspace if workspace is not None else 'auto'} GiB")
    print(f"[export] Device         : {device}\n")

    model = YOLO(str(weights))
    names = normalize_model_names(getattr(model, "names", None))
    validate_class_names(names)

    export_kwargs: dict[str, object] = {
        "format": "engine",
        "imgsz": imgsz,
        "dynamic": dynamic,
        "simplify": simplify,
        "workspace": workspace,
        "device": device,
        "batch": batch,
        "verbose": False,
        **_precision_kwargs(quantize),
    }
    if quantize == 8:
        export_kwargs["data"] = str(Path(data).expanduser().resolve())  # type: ignore[arg-type]

    result = model.export(**export_kwargs)
    if not result:
        sys.exit("[ERROR] Ultralytics did not return an exported engine path.")

    if isinstance(result, (list, tuple)):
        result = result[0]
    engine_path = Path(result)

    if not engine_path.is_file():
        sys.exit(f"[ERROR] Export finished but engine was not found: {engine_path}")

    engine_path = engine_path.resolve()
    names_path = save_names_sidecar(engine_path, names)

    print("\n✓ Export complete!")
    print(f"  Engine : {engine_path}")
    print(f"  Names  : {names_path.resolve()}")
    print("\n" + "─" * 68)
    print("Use with infer.py:")
    print(f"  python infer.py <sequence_dir> --model {engine_path} --imgsz {imgsz}")
    if dynamic:
        print(
            "Dynamic shapes are enabled. Use only input shapes covered by the "
            "generated TensorRT optimization profile."
        )
    else:
        print(f"Static engine: use --imgsz {imgsz} and batch <= {batch}.")
    print("─" * 68)

    return engine_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the project's compatible 10-class YOLO .pt model to TensorRT. "
            "No pretrained fallback model is used."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--weights",
        default=None,
        help=f"Compatible custom .pt weights. Default: '{_DEFAULT_WEIGHTS}'.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="TensorRT optimization/input image size in pixels.",
    )

    precision_group = parser.add_mutually_exclusive_group()
    precision_group.add_argument(
        "--fp32",
        action="store_true",
        help="Export FP32 instead of the default FP16.",
    )
    precision_group.add_argument(
        "--int8",
        action="store_true",
        help="Export INT8; requires --data for calibration.",
    )

    parser.add_argument(
        "--data",
        default=None,
        help="Local calibration dataset YAML used for INT8 export.",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="Enable dynamic input shapes; --batch is the maximum batch size.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=1,
        help="Static batch size, or maximum batch size for dynamic export.",
    )
    parser.add_argument(
        "--workspace",
        type=float,
        default=4.0,
        help="Maximum TensorRT workspace in GiB.",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="CUDA device index, for example 0 or 1.",
    )
    parser.add_argument(
        "--no-simplify",
        action="store_true",
        help="Disable ONNX graph simplification before TensorRT conversion.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_export_args(args)

    weights = resolve_weights(args.weights)
    quantize = 8 if args.int8 else (None if args.fp32 else 16)

    export_engine(
        weights=weights,
        imgsz=args.imgsz,
        quantize=quantize,
        dynamic=args.dynamic,
        data=args.data,
        workspace=args.workspace,
        device=args.device,
        simplify=not args.no_simplify,
        batch=args.batch,
    )


if __name__ == "__main__":
    main()