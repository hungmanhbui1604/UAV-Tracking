"""
infer.py
--------
Run YOLO + BoT-SORT/ByteTrack tracking on one numeric image sequence and
produce an annotated video or annotated frames, optionally with MOT text output.

The script uses only the configured custom model. It never falls back to a
generic pretrained model.
"""

from __future__ import annotations

import argparse
import colorsys
import sys
from pathlib import Path
from typing import TextIO

import cv2
import numpy as np
from ultralytics import YOLO


_DEFAULT_MODEL_PATH = "runs/detect/train/weights/best.pt"
_DEFAULT_TRACKER_YAML = "visdrone_botsort.yaml"

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

# YOLO zero-based class index -> VisDrone one-based class ID.
YOLO_TO_VISDRONE: dict[int, int] = {
    0: 1,
    1: 2,
    2: 3,
    3: 4,
    4: 5,
    5: 6,
    6: 7,
    7: 8,
    8: 9,
    9: 10,
}

VISDRONE_CLASSES: dict[int, str] = {
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
}

TRACKED_CLASSES: dict[int, str] = {
    1: "pedestrian",
    4: "car",
    5: "van",
    6: "truck",
    9: "bus",
}


def _normalize_names(raw_names: object) -> list[str]:
    if isinstance(raw_names, dict):
        try:
            keys = sorted(raw_names, key=lambda key: int(key))
        except (TypeError, ValueError):
            keys = sorted(raw_names, key=str)
        return [str(raw_names[key]).strip() for key in keys]
    if isinstance(raw_names, (list, tuple)):
        return [str(name).strip() for name in raw_names]
    return []


def _validate_names(names: list[str], source: Path) -> None:
    if names == EXPECTED_CLASS_NAMES:
        return
    expected = ", ".join(f"{i}:{name}" for i, name in enumerate(EXPECTED_CLASS_NAMES))
    actual = ", ".join(f"{i}:{name}" for i, name in enumerate(names)) or "<missing>"
    raise ValueError(
        f"Model taxonomy mismatch for {source}.\n"
        f"Expected: {expected}\n"
        f"Actual  : {actual}"
    )


def _build_palette(class_dict: dict[int, str]) -> dict[int, tuple[int, int, int]]:
    """Assign one visually distinct BGR colour to every displayed class."""
    ids = sorted(class_dict)
    count = max(len(ids), 1)
    palette: dict[int, tuple[int, int, int]] = {}
    for index, class_id in enumerate(ids):
        red, green, blue = colorsys.hsv_to_rgb(index / count, 0.85, 0.95)
        palette[class_id] = (
            int(blue * 255),
            int(green * 255),
            int(red * 255),
        )
    return palette


def _draw_box(
    image: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    track_id: int,
    color: tuple[int, int, int],
    show_id: bool,
) -> None:
    image_height, image_width = image.shape[:2]
    font_scale = max(0.28, image_height / 3840)
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_thickness = 1

    x1 = max(0, min(x1, image_width - 1))
    y1 = max(0, min(y1, image_height - 1))
    x2 = max(0, min(x2, image_width - 1))
    y2 = max(0, min(y2, image_height - 1))

    cv2.rectangle(image, (x1, y1), (x2, y2), color, 1, cv2.LINE_AA)

    if not show_id:
        return

    label = str(track_id)
    (text_width, text_height), baseline = cv2.getTextSize(
        label, font, font_scale, text_thickness
    )
    label_x = x1
    label_y = y1 - 2
    if label_y - text_height < 0:
        label_y = y1 + text_height + 2

    cv2.rectangle(
        image,
        (label_x, label_y - text_height - baseline),
        (label_x + text_width, label_y + baseline),
        color,
        cv2.FILLED,
    )

    red, green, blue = color[2] / 255, color[1] / 255, color[0] / 255
    luminance = 0.299 * red + 0.587 * green + 0.114 * blue
    text_color = (0, 0, 0) if luminance > 0.5 else (255, 255, 255)

    cv2.putText(
        image,
        label,
        (label_x, label_y),
        font,
        font_scale,
        text_color,
        text_thickness,
        cv2.LINE_AA,
    )


def _draw_legend(
    image: np.ndarray,
    palette: dict[int, tuple[int, int, int]],
    class_names: dict[int, str],
) -> None:
    image_height, image_width = image.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.30, image_height / 4320)
    thickness = 1
    swatch_size = max(8, int(image_height / 100))
    padding = 4
    line_height = swatch_size + padding
    class_ids = sorted(class_names)

    if not class_ids:
        return

    legend_height = len(class_ids) * line_height + padding
    max_label_width = max(
        cv2.getTextSize(
            class_names.get(class_id, str(class_id)),
            font,
            font_scale,
            thickness,
        )[0][0]
        for class_id in class_ids
    )
    legend_width = swatch_size + padding + max_label_width + padding
    x0 = image_width - legend_width - padding
    y0 = padding

    overlay = image.copy()
    cv2.rectangle(
        overlay,
        (x0 - padding, y0 - padding),
        (image_width - padding, y0 + legend_height),
        (30, 30, 30),
        cv2.FILLED,
    )
    cv2.addWeighted(overlay, 0.55, image, 0.45, 0, image)

    for index, class_id in enumerate(class_ids):
        color = palette.get(class_id, (200, 200, 200))
        name = class_names.get(class_id, str(class_id))
        swatch_y = y0 + index * line_height
        cv2.rectangle(
            image,
            (x0, swatch_y),
            (x0 + swatch_size, swatch_y + swatch_size),
            color,
            cv2.FILLED,
        )
        cv2.putText(
            image,
            name,
            (x0 + swatch_size + padding, swatch_y + swatch_size - 1),
            font,
            font_scale,
            (220, 220, 220),
            thickness,
            cv2.LINE_AA,
        )


def _collect_frames(sequence_dir: Path) -> dict[int, Path]:
    """Return numeric frame ID -> image path."""
    if not sequence_dir.is_dir():
        return {}

    frame_map: dict[int, Path] = {}
    for path in sorted(sequence_dir.iterdir()):
        if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        try:
            frame_map[int(path.stem)] = path
        except ValueError:
            continue
    return frame_map


def _load_model(model_path: Path) -> YOLO:
    """
    Load and validate a compatible .pt or .engine model.

    TensorRT engines must have the .names sidecar generated by
    export_tensorrt.py. No built-in taxonomy fallback is used.
    """
    suffix = model_path.suffix.lower()
    if suffix not in {".pt", ".engine"}:
        raise ValueError(f"Unsupported model format: {model_path}")

    model = YOLO(str(model_path))

    if suffix == ".pt":
        _validate_names(_normalize_names(getattr(model, "names", None)), model_path)
        return model

    names_path = model_path.with_suffix(".names")
    if not names_path.is_file():
        raise FileNotFoundError(
            f"TensorRT class-name sidecar not found: {names_path}\n"
            "Re-export the engine with export_tensorrt.py or place the matching "
            ".names file beside the engine."
        )

    names = [
        line.strip()
        for line in names_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    _validate_names(names, names_path)
    desired_names = {index: name for index, name in enumerate(names)}
    injected = False

    def _inject_names(predictor: object) -> None:
        nonlocal injected
        if injected:
            return
        backend = getattr(predictor, "model", None)
        if backend is None or not hasattr(backend, "names"):
            raise RuntimeError("Could not access TensorRT backend class names.")
        backend.names = desired_names
        injected = True
        print(f"[infer] Applied {len(desired_names)} class names from {names_path.name}")

    model.add_callback("on_predict_start", _inject_names)
    return model


def _reset_tracker(model: YOLO) -> None:
    """Reset tracker state before a new independent sequence."""
    predictor = getattr(model, "predictor", None)
    trackers = getattr(predictor, "trackers", None)
    if trackers:
        for tracker in trackers:
            tracker.reset()


def _resolve_model(model_arg: str | None) -> Path:
    """Resolve the project model without any generic fallback."""
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


def _validate_runtime_args(
    fps: float,
    imgsz: int,
    conf: float,
    iou: float,
    conf_min: float,
    start: int,
    end: int | None,
) -> None:
    if fps <= 0:
        raise ValueError("fps must be greater than 0.")
    if imgsz <= 0:
        raise ValueError("imgsz must be greater than 0.")
    if not 0 <= conf <= 1:
        raise ValueError("conf must be between 0 and 1.")
    if not 0 <= iou <= 1:
        raise ValueError("iou must be between 0 and 1.")
    if not 0 <= conf_min <= 1:
        raise ValueError("conf_min must be between 0 and 1.")
    if start < 0:
        raise ValueError("start must be non-negative.")
    if end is not None and end < start:
        raise ValueError("end must be greater than or equal to start.")


def infer(
    seq_dir: Path,
    model_path: Path,
    tracker_cfg: str,
    out_path: Path,
    save_txt: Path | None = None,
    fps: float = 25.0,
    imgsz: int = 1280,
    conf: float = 0.25,
    iou: float = 0.45,
    save_frames: bool = False,
    show_id: bool = True,
    conf_min: float = 0.0,
    all_classes: bool = True,
    start: int = 1,
    end: int | None = None,
    device: str | None = None,
) -> None:
    _validate_runtime_args(fps, imgsz, conf, iou, conf_min, start, end)

    print(f"[infer] Loading model : {model_path}")
    model = _load_model(model_path)

    print(f"[infer] Scanning frames in : {seq_dir}")
    frame_map = _collect_frames(seq_dir)
    if not frame_map:
        raise FileNotFoundError(f"No numeric JPEG/PNG frames found in {seq_dir}")

    last_frame = end if end is not None else max(frame_map)
    frame_ids = sorted(
        frame_id for frame_id in frame_map if start <= frame_id <= last_frame
    )
    if not frame_ids:
        raise ValueError(
            f"No frames in range [{start}, {last_frame}]. "
            f"Available: [{min(frame_map)}, {max(frame_map)}]"
        )

    print(
        f"[infer] Processing frames {frame_ids[0]}–{frame_ids[-1]} "
        f"({len(frame_ids)} frames)"
    )

    shown_classes = VISDRONE_CLASSES if all_classes else TRACKED_CLASSES
    palette = _build_palette(shown_classes)
    shown_class_ids = set(shown_classes)

    first_image = cv2.imread(str(frame_map[frame_ids[0]]))
    if first_image is None:
        raise IOError(f"Cannot read {frame_map[frame_ids[0]]}")
    output_height, output_width = first_image.shape[:2]

    video_writer: cv2.VideoWriter | None = None
    text_file: TextIO | None = None

    if save_frames:
        out_path.mkdir(parents=True, exist_ok=True)
        print(f"[infer] Saving frames to : {out_path}")
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(
            str(out_path), fourcc, fps, (output_width, output_height)
        )
        if not video_writer.isOpened():
            video_writer.release()
            raise IOError(f"Could not open video writer: {out_path}")
        print(
            f"[infer] Writing video : {out_path} "
            f"({output_width}×{output_height} @ {fps:g} fps)"
        )

    track_kwargs: dict[str, object] = {
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
        track_kwargs["device"] = device

    try:
        if save_txt is not None:
            save_txt = save_txt.expanduser()
            save_txt.parent.mkdir(parents=True, exist_ok=True)
            text_file = save_txt.open("w", encoding="utf-8", newline="\n")
            print(f"[infer] Saving MOT .txt : {save_txt}")

        _reset_tracker(model)

        for offset, frame_id in enumerate(frame_ids, start=1):
            original = cv2.imread(str(frame_map[frame_id]))
            if original is None:
                raise IOError(f"Cannot read image: {frame_map[frame_id]}")
            if original.shape[:2] != (output_height, output_width):
                raise ValueError(
                    f"Frame size mismatch at {frame_map[frame_id]}: "
                    f"expected {output_width}x{output_height}, "
                    f"got {original.shape[1]}x{original.shape[0]}"
                )

            results = model.track(source=original, **track_kwargs)
            result = results[0] if results else None
            image = original.copy()

            if (
                result is not None
                and result.boxes is not None
                and result.boxes.id is not None
            ):
                boxes = result.boxes.xyxy.cpu().numpy()
                track_ids = result.boxes.id.int().cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_indices = result.boxes.cls.int().cpu().numpy()

                for box, track_id, box_conf, class_index in zip(
                    boxes, track_ids, confidences, class_indices
                ):
                    class_index_int = int(class_index)
                    if class_index_int not in YOLO_TO_VISDRONE:
                        raise ValueError(
                            f"Unexpected model class index: {class_index_int}"
                        )

                    x1, y1, x2, y2 = (float(value) for value in box)
                    width = x2 - x1
                    height = y2 - y1
                    visdrone_class = YOLO_TO_VISDRONE[class_index_int]

                    if text_file is not None:
                        text_file.write(
                            f"{frame_id},{int(track_id)},"
                            f"{x1:.2f},{y1:.2f},{width:.2f},{height:.2f},"
                            f"{float(box_conf):.4f},{visdrone_class},-1,-1\n"
                        )

                    if (
                        visdrone_class in shown_class_ids
                        and float(box_conf) >= conf_min
                    ):
                        color = palette.get(visdrone_class, (200, 200, 200))
                        _draw_box(
                            image,
                            int(round(x1)),
                            int(round(y1)),
                            int(round(x2)),
                            int(round(y2)),
                            int(track_id),
                            color,
                            show_id,
                        )

            _draw_legend(image, palette, shown_classes)

            if save_frames:
                output_file = out_path / f"{frame_id:07d}.jpg"
                success = cv2.imwrite(
                    str(output_file),
                    image,
                    [cv2.IMWRITE_JPEG_QUALITY, 92],
                )
                if not success:
                    raise IOError(f"Could not write image: {output_file}")
            else:
                assert video_writer is not None
                video_writer.write(image)

            if offset % 50 == 0 or offset == len(frame_ids):
                print(f"  [{offset:>5}/{len(frame_ids)}] frame {frame_id}")

    finally:
        if video_writer is not None:
            video_writer.release()
        if text_file is not None:
            text_file.close()

    print("\n✓ Done!")
    print(f"  {'Frames' if save_frames else 'Video '} : {out_path.resolve()}")
    if save_txt is not None:
        print(f"  MOT    : {save_txt.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run YOLO plus BoT-SORT/ByteTrack on a numeric image sequence and "
            "write annotated output."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "sequence",
        type=Path,
        help="Directory containing numeric JPEG/PNG frame filenames.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            f"Compatible .pt or .engine model. Default: '{_DEFAULT_MODEL_PATH}'. "
            "No generic model fallback is used."
        ),
    )
    parser.add_argument(
        "--tracker",
        default=_DEFAULT_TRACKER_YAML,
        help="Tracker YAML path or an Ultralytics built-in tracker YAML name.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory; defaults to the sequence parent.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Output filename or frame-directory name.",
    )
    parser.add_argument(
        "--save-txt",
        type=Path,
        default=None,
        help="Optional MOT-format output text path.",
    )
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--iou", type=float, default=0.60)
    parser.add_argument(
        "--frames",
        action="store_true",
        help="Save annotated JPEG frames instead of an MP4 video.",
    )
    parser.add_argument(
        "--no-id",
        action="store_true",
        help="Do not draw track IDs.",
    )
    parser.add_argument(
        "--conf-min",
        type=float,
        default=0.0,
        help="Minimum confidence for drawing; MOT output remains unchanged.",
    )
    parser.add_argument(
        "--tracked-only",
        action="store_true",
        help="Draw only pedestrian, car, van, truck and bus.",
    )
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device such as 0, 1 or cpu; omit for Ultralytics default.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sequence_dir = args.sequence.expanduser().resolve()
    if not sequence_dir.is_dir():
        sys.exit(f"[ERROR] Sequence directory not found: {sequence_dir}")

    output_dir = (
        args.out_dir.expanduser().resolve()
        if args.out_dir is not None
        else sequence_dir.parent
    )

    if args.name is not None:
        output_name = args.name
        if not args.frames and not Path(output_name).suffix:
            output_name += ".mp4"
    else:
        output_name = (
            f"{sequence_dir.name}_tracked"
            if args.frames
            else f"{sequence_dir.name}_tracked.mp4"
        )

    infer(
        seq_dir=sequence_dir,
        model_path=_resolve_model(args.model),
        tracker_cfg=args.tracker,
        out_path=output_dir / output_name,
        save_txt=args.save_txt,
        fps=args.fps,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        save_frames=args.frames,
        show_id=not args.no_id,
        conf_min=args.conf_min,
        all_classes=not args.tracked_only,
        start=args.start,
        end=args.end,
        device=args.device,
    )


if __name__ == "__main__":
    main()