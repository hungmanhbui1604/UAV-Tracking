"""
benchmark_speed.py
------------------
Benchmark end-to-end YOLO + BoT-SORT/ByteTrack tracking speed on one image
sequence or every sequence inside a parent directory.

The timed region includes the complete model.track() API call. With --preload,
disk reads are removed from the timed region. The reported residual time is the
wall-clock total minus Ultralytics' preprocess, inference and postprocess
profilers; it includes tracker work plus Python/API/source overhead and must not
be interpreted as isolated tracker time.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Ensure that the sibling infer.py is imported even when launched elsewhere.
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

try:
    from infer import (
        _DEFAULT_MODEL_PATH,
        _DEFAULT_TRACKER_YAML,
        _collect_frames,
        _load_model,
        _reset_tracker,
    )
except ImportError as exc:
    sys.exit(
        "[ERROR] Could not import the sibling infer.py. Keep infer.py and "
        f"benchmark_speed.py in the same directory. Details: {exc}"
    )


class BenchmarkResult:
    def __init__(self, sequence_name: str, num_frames: int) -> None:
        self.sequence_name = sequence_name
        self.num_frames = num_frames
        self.total_time_s = 0.0
        self.sum_preprocess_ms = 0.0
        self.sum_inference_ms = 0.0
        self.sum_postprocess_ms = 0.0
        self.sum_residual_ms = 0.0
        self.total_objects = 0
        self.frame_latencies_ms: list[float] = []

    @property
    def fps(self) -> float:
        return self.num_frames / self.total_time_s if self.total_time_s else 0.0

    @property
    def avg_total_ms(self) -> float:
        return (
            self.total_time_s * 1000.0 / self.num_frames
            if self.num_frames
            else 0.0
        )

    @property
    def avg_preprocess_ms(self) -> float:
        return (
            self.sum_preprocess_ms / self.num_frames
            if self.num_frames
            else 0.0
        )

    @property
    def avg_inference_ms(self) -> float:
        return (
            self.sum_inference_ms / self.num_frames
            if self.num_frames
            else 0.0
        )

    @property
    def avg_postprocess_ms(self) -> float:
        return (
            self.sum_postprocess_ms / self.num_frames
            if self.num_frames
            else 0.0
        )

    @property
    def avg_residual_ms(self) -> float:
        return (
            self.sum_residual_ms / self.num_frames
            if self.num_frames
            else 0.0
        )

    @property
    def avg_objects_per_frame(self) -> float:
        return self.total_objects / self.num_frames if self.num_frames else 0.0

    @property
    def p50_ms(self) -> float:
        return (
            float(np.percentile(self.frame_latencies_ms, 50))
            if self.frame_latencies_ms
            else 0.0
        )

    @property
    def p95_ms(self) -> float:
        return (
            float(np.percentile(self.frame_latencies_ms, 95))
            if self.frame_latencies_ms
            else 0.0
        )


def _predictor_device(model: YOLO) -> torch.device | None:
    predictor = getattr(model, "predictor", None)
    device = getattr(predictor, "device", None)
    if isinstance(device, torch.device):
        return device

    backend = getattr(predictor, "model", None)
    device = getattr(backend, "device", None)
    return device if isinstance(device, torch.device) else None


def _cuda_synchronize(model: YOLO) -> None:
    device = _predictor_device(model)
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)


def _device_description(model: YOLO) -> str:
    device = _predictor_device(model)
    if device is None:
        return "Unknown (predictor not initialized)"
    if device.type != "cuda":
        return str(device)

    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    return f"CUDA:{index} ({torch.cuda.get_device_name(index)})"


def _track_kwargs(
    tracker_cfg: str,
    imgsz: int,
    conf: float,
    iou: float,
    device: str | None,
) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "tracker": tracker_cfg,
        "imgsz": imgsz,
        "conf": conf,
        "iou": iou,
        "persist": True,
        "stream": False,
        "verbose": False,
        "save": False,
    }
    if device is not None:
        kwargs["device"] = device
    return kwargs


def benchmark_single_sequence(
    model: YOLO,
    sequence_dir: Path,
    tracker_cfg: str,
    imgsz: int,
    conf: float,
    iou: float,
    warmup_frames: int,
    preload_ram: bool,
    device: str | None,
) -> BenchmarkResult | None:
    frame_map = _collect_frames(sequence_dir)
    if not frame_map:
        return None

    frame_ids = sorted(frame_map)
    num_frames = len(frame_ids)
    print("\n" + "─" * 66)
    print(
        f"▶ Benchmarking sequence: {sequence_dir.name} "
        f"({num_frames} frames)"
    )

    sources: list[str | np.ndarray]
    if preload_ram:
        print(
            f"  [RAM] Preloading {num_frames} images...",
            end="",
            flush=True,
        )
        sources = []
        for frame_id in frame_ids:
            image = cv2.imread(str(frame_map[frame_id]))
            if image is None:
                raise IOError(f"Cannot read image: {frame_map[frame_id]}")
            sources.append(image)
        print(" done")
    else:
        sources = [str(frame_map[frame_id]) for frame_id in frame_ids]

    kwargs = _track_kwargs(tracker_cfg, imgsz, conf, iou, device)

    actual_warmup = min(warmup_frames, num_frames)
    print(f"  [Warmup] Running {actual_warmup} untimed frames...")
    _reset_tracker(model)
    for index in range(actual_warmup):
        model.track(source=sources[index], **kwargs)
    _cuda_synchronize(model)

    _reset_tracker(model)
    timing_mode = (
        "RAM-preloaded; no timed disk read"
        if preload_ram
        else "includes image read/decode from disk"
    )
    print(f"  [Eval] End-to-end model.track timing ({timing_mode})")

    result = BenchmarkResult(sequence_dir.name, num_frames)

    for index, source in enumerate(sources, start=1):
        _cuda_synchronize(model)
        start_time = time.perf_counter()

        outputs = model.track(source=source, **kwargs)

        _cuda_synchronize(model)
        frame_time_s = time.perf_counter() - start_time
        frame_time_ms = frame_time_s * 1000.0

        result.total_time_s += frame_time_s
        result.frame_latencies_ms.append(frame_time_ms)

        if outputs:
            output = outputs[0]
            speed = getattr(output, "speed", None) or {}
            preprocess_ms = float(speed.get("preprocess", 0.0) or 0.0)
            inference_ms = float(speed.get("inference", 0.0) or 0.0)
            postprocess_ms = float(speed.get("postprocess", 0.0) or 0.0)

            result.sum_preprocess_ms += preprocess_ms
            result.sum_inference_ms += inference_ms
            result.sum_postprocess_ms += postprocess_ms

            profiled_ms = preprocess_ms + inference_ms + postprocess_ms
            result.sum_residual_ms += max(0.0, frame_time_ms - profiled_ms)

            boxes = getattr(output, "boxes", None)
            track_ids = getattr(boxes, "id", None)
            if track_ids is not None:
                result.total_objects += len(track_ids)
        else:
            # No Results object means no internal speed breakdown is available.
            result.sum_residual_ms += frame_time_ms

        if index % 100 == 0 or index == num_frames:
            current_fps = index / result.total_time_s
            print(
                f"    Processed {index:>5}/{num_frames} frames | "
                f"Current FPS: {current_fps:.2f}"
            )

    print(f"\n  ✓ {sequence_dir.name}")
    print(
        f"    End-to-end : {result.fps:.2f} FPS | "
        f"{result.avg_total_ms:.2f} ms/frame"
    )
    print(
        f"    Latency    : P50 {result.p50_ms:.2f} ms | "
        f"P95 {result.p95_ms:.2f} ms"
    )
    print(
        f"    Breakdown  : preprocess {result.avg_preprocess_ms:.2f} | "
        f"inference {result.avg_inference_ms:.2f} | "
        f"postprocess {result.avg_postprocess_ms:.2f} | "
        f"residual {result.avg_residual_ms:.2f} ms"
    )
    print(
        f"    Predictions: {result.avg_objects_per_frame:.1f} "
        "tracked objects/frame"
    )
    return result


def print_summary_table(
    results: list[BenchmarkResult],
    model_path: Path,
    imgsz: int,
    preload: bool,
    model: YOLO,
) -> None:
    if not results:
        return

    total_frames = sum(item.num_frames for item in results)
    total_time_s = sum(item.total_time_s for item in results)
    total_objects = sum(item.total_objects for item in results)
    all_latencies = [
        latency
        for item in results
        for latency in item.frame_latencies_ms
    ]

    overall_fps = total_frames / total_time_s if total_time_s else 0.0
    overall_latency = (
        total_time_s * 1000.0 / total_frames if total_frames else 0.0
    )
    overall_density = total_objects / total_frames if total_frames else 0.0
    overall_p95 = (
        float(np.percentile(all_latencies, 95)) if all_latencies else 0.0
    )

    print("\n" + "=" * 106)
    print("FINAL TRACKING BENCHMARK")
    print("=" * 106)
    print(f"Model         : {model_path}")
    print(f"Image size    : {imgsz}")
    print(f"Device        : {_device_description(model)}")
    print(
        "Input mode    : "
        + ("RAM preload" if preload else "disk path per frame")
    )
    print(f"Frames        : {total_frames} across {len(results)} sequence(s)")
    print(
        f"Overall speed : {overall_fps:.2f} FPS "
        f"({overall_latency:.2f} ms/frame; P95 {overall_p95:.2f} ms)"
    )
    print(f"Avg prediction: {overall_density:.1f} tracked objects/frame")
    print("-" * 106)
    print(
        f"{'Sequence':<28} | {'Frames':>7} | {'Obj/f':>7} | "
        f"{'FPS':>8} | {'Mean ms':>9} | {'P95 ms':>9} | {'Residual':>9}"
    )
    print("-" * 106)

    for item in results:
        print(
            f"{item.sequence_name:<28} | "
            f"{item.num_frames:>7} | "
            f"{item.avg_objects_per_frame:>7.1f} | "
            f"{item.fps:>8.2f} | "
            f"{item.avg_total_ms:>9.2f} | "
            f"{item.p95_ms:>9.2f} | "
            f"{item.avg_residual_ms:>9.2f}"
        )

    print("=" * 106)
    print(
        "Residual = wall-clock model.track time minus Ultralytics preprocess, "
        "inference and postprocess timings. It includes tracker work and API/"
        "source overhead; with no --preload it can also include disk decoding."
    )


def resolve_model_path(model_arg: str | None) -> Path:
    """Resolve a compatible model without any generic fallback."""
    model_path = (
        Path(model_arg).expanduser()
        if model_arg
        else Path(_DEFAULT_MODEL_PATH)
    )

    if not model_path.is_file():
        if model_arg:
            sys.exit(f"[ERROR] Model not found: {model_path}")
        sys.exit(
            f"[ERROR] Default model not found: '{_DEFAULT_MODEL_PATH}'.\n"
            "        Pass --model with a compatible VisDrone .pt or .engine model."
        )

    if model_path.suffix.lower() not in {".pt", ".engine"}:
        sys.exit(f"[ERROR] Model must be a .pt or .engine file: {model_path}")

    return model_path.resolve()


def validate_args(args: argparse.Namespace) -> None:
    if args.imgsz <= 0:
        sys.exit("[ERROR] --imgsz must be greater than 0.")
    if args.warmup <= 0:
        sys.exit("[ERROR] --warmup must be greater than 0.")
    if not 0 <= args.conf <= 1:
        sys.exit("[ERROR] --conf must be between 0 and 1.")
    if not 0 <= args.iou <= 1:
        sys.exit("[ERROR] --iou must be between 0 and 1.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark end-to-end YOLO plus BoT-SORT/ByteTrack tracking speed "
            "without rendering or output encoding."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "target",
        type=Path,
        help="One sequence directory or a parent containing sequence directories.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            f"Compatible .pt or .engine model. Default: '{_DEFAULT_MODEL_PATH}'. "
            "No generic fallback is used."
        ),
    )
    parser.add_argument(
        "--tracker",
        default=_DEFAULT_TRACKER_YAML,
        help="Tracker YAML path or Ultralytics built-in YAML name.",
    )
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--iou", type=float, default=0.60)
    parser.add_argument(
        "--warmup",
        type=int,
        default=30,
        help="Untimed warm-up frames before each sequence.",
    )
    parser.add_argument(
        "--preload",
        action="store_true",
        help="Load all images into RAM before timing.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device such as 0, 1 or cpu; omit for default.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)

    target = args.target.expanduser().resolve()
    if not target.is_dir():
        sys.exit(f"[ERROR] Target must be an existing directory: {target}")

    model_path = resolve_model_path(args.model)
    print(f"[Benchmark] Initializing model: {model_path}")
    model = _load_model(model_path)

    sequences: list[Path] = []
    if _collect_frames(target):
        sequences.append(target)
    else:
        for child in sorted(target.iterdir()):
            if child.is_dir() and _collect_frames(child):
                sequences.append(child)

    if not sequences:
        sys.exit(
            f"[ERROR] No sequence directories with numeric JPEG/PNG frames "
            f"were found in: {target}"
        )

    print(f"[Benchmark] Found {len(sequences)} sequence(s).")

    results: list[BenchmarkResult] = []
    for sequence_dir in sequences:
        sequence_result = benchmark_single_sequence(
            model=model,
            sequence_dir=sequence_dir,
            tracker_cfg=args.tracker,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            warmup_frames=args.warmup,
            preload_ram=args.preload,
            device=args.device,
        )
        if sequence_result is not None:
            results.append(sequence_result)

    print_summary_table(
        results=results,
        model_path=model_path,
        imgsz=args.imgsz,
        preload=args.preload,
        model=model,
    )


if __name__ == "__main__":
    main()