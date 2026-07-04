"""
visualize.py
----------
Visualise MOT-challenge tracking results (or ground-truth annotations) by
drawing thin bounding boxes coloured by class, with a small ID label, on
every frame of a sequence and writing the output to an annotated video or
image sequence.

Supported input annotation formats
  (both produced by track_visdrone.py / track_uavdt.py and the GT files):
  <frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<class>,-1,-1

Usage examples
--------------
# Basic – auto-detects output path next to the sequence:
python visualize.py \
    --seq  datasets/VisDrone/VisDrone2019-MOT-train/sequences/uav0000013_00000_v \
    --ann  datasets/VisDrone/VisDrone2019-MOT-train/annotations/uav0000013_00000_v.txt

# Specify a custom output directory and filename:
python visualize.py \
    --seq  datasets/VisDrone/VisDrone2019-MOT-test-dev/sequences/uav0000268_05773_v \
    --ann  runs/tracking/visdrone/test-dev/annotations/uav0000268_05773_v.txt \
    --out-dir vis_output/ --name uav0000268_05773_v_pred.mp4

# Save individual annotated frames (PNG) instead of a video:
python visualize.py \
    --seq  datasets/VisDrone/VisDrone2019-MOT-train/sequences/uav0000013_00000_v \
    --ann  datasets/VisDrone/VisDrone2019-MOT-train/annotations/uav0000013_00000_v.txt \
    --out-dir vis_output/frames/ \
    --frames

# Visualise only a range of frames:
python visualize.py \
    --seq  datasets/VisDrone/VisDrone2019-MOT-train/sequences/uav0000013_00000_v \
    --ann  datasets/VisDrone/VisDrone2019-MOT-train/annotations/uav0000013_00000_v.txt \
    --out-dir vis_output/ --name uav0000013_00000_v.mp4 \
    --start 10 --end 60

Arguments
---------
--seq       Directory that contains the JPEG/PNG frames (1-indexed filenames).
--ann       MOT-challenge .txt annotation file.
--out-dir   Output directory. Default: the parent folder of the sequence.
--name      Output filename (with or without .mp4 extension). For videos the
            '.mp4' extension is added automatically if omitted.
            Default: <sequence_name>_vis.mp4.
--dataset   'visdrone' (default) or 'uavdt' – picks the correct class names.
--fps       Frames per second for the output video (default: 25).
--start     First frame to visualise (1-indexed, default: 1).
--end       Last  frame to visualise (inclusive, default: last frame).
--frames    Save individual frames instead of a video.
--no-id     Suppress track ID labels.
--conf-min  Minimum confidence to display (default: 0.0 – show everything).
"""

from __future__ import annotations

import argparse
import colorsys
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


# ── Class definitions ──────────────────────────────────────────────────────────

# Full VisDrone-MOT 12-class taxonomy (1-indexed) – kept for reference.
VISDRONE_CLASSES: dict[int, str] = {
    0:  "ignore",
    1:  "pedestrian",
    2:  "person",
    3:  "bicycle",
    4:  "car",
    5:  "van",
    6:  "truck",
    7:  "tricycle",
    8:  "awning-tricycle",
    9:  "bus",
    10: "motor",
    11: "others",
}

# The 5 classes that both track_visdrone.py and track_uavdt.py output.
# Both scripts now write VisDrone 1-indexed class IDs so the same dict is used
# for every dataset mode in visualize.py.
TRACKED_CLASSES: dict[int, str] = {
    1: "pedestrian",  # Red (RGB: 242, 36, 36)
    4: "car",         # Yellow-Green (RGB: 201, 242, 36)
    5: "van",         # Green (RGB: 36, 242, 118)
    6: "truck",       # Blue (RGB: 36, 118, 242)
    9: "bus",         # Purple (RGB: 201, 36, 242)
}

# Classes present in the full VisDrone taxonomy that are NOT tracked.
_NON_TRACKED = frozenset(VISDRONE_CLASSES) - frozenset(TRACKED_CLASSES)


# ── Colour palette ─────────────────────────────────────────────────────────────

def _build_class_palette(class_dict: dict[int, str]) -> dict[int, tuple[int, int, int]]:
    """
    Assign a visually distinct BGR colour to each class.
    Uses evenly-spaced hues around the HSV wheel with high saturation and
    brightness so they stand out against typical aerial imagery.
    """
    ids = sorted(class_dict.keys())
    palette: dict[int, tuple[int, int, int]] = {}
    n = max(len(ids), 1)
    for i, cls_id in enumerate(ids):
        hue = i / n                          # 0.0 – <1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
        # OpenCV uses BGR
        palette[cls_id] = (int(b * 255), int(g * 255), int(r * 255))
    return palette


# Single palette built from the 5 tracked classes (same IDs for both datasets).
TRACKED_PALETTE = _build_class_palette(TRACKED_CLASSES)

# All active classes (excluding class 0: ignore region)
ALL_CLASSES = {k: v for k, v in VISDRONE_CLASSES.items() if k != 0}
ALL_PALETTE = _build_class_palette(ALL_CLASSES)


# ── Annotation loading ─────────────────────────────────────────────────────────

def load_annotations(
    ann_path: Path,
    conf_min: float = 0.0,
    ignore_classes: set[int] | None = None,
) -> dict[int, list[dict]]:
    """
    Parse a MOT-challenge annotation file.

    Format: <frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<class>,-1,-1

    Returns a dict mapping frame_index (int, 1-indexed) → list of detection dicts:
        { 'id': int, 'x': float, 'y': float, 'w': float, 'h': float,
          'conf': float, 'cls': int }

    ignore_classes: set of class IDs to discard entirely (default: {0} which is
        the VisDrone 'ignore region' – large crowd/occlusion marker boxes that
        are NOT real objects and would appear as huge boxes over many people).
    """
    if ignore_classes is None:
        # Skip VisDrone ignore region (class 0) by default
        ignore_classes = {0}

    detections: dict[int, list[dict]] = defaultdict(list)

    with open(ann_path, newline="") as fh:
        for row in csv.reader(fh):
            if len(row) < 8:
                continue  # skip malformed lines
            try:
                frame   = int(row[0])
                obj_id  = int(row[1])
                x       = float(row[2])
                y       = float(row[3])
                w       = float(row[4])
                h       = float(row[5])
                conf    = float(row[6])
                cls     = int(row[7])
            except ValueError:
                continue  # skip header / comment lines

            if cls in ignore_classes:
                continue  # skip ignore regions / unwanted classes
            if conf < conf_min:
                continue
            if w <= 0 or h <= 0:
                continue

            detections[frame].append(
                {"id": obj_id, "x": x, "y": y, "w": w, "h": h,
                 "conf": conf, "cls": cls}
            )

    return detections


# ── Drawing ────────────────────────────────────────────────────────────────────

def draw_detections(
    frame_img: np.ndarray,
    dets: list[dict],
    palette: dict[int, tuple[int, int, int]],
    class_names: dict[int, str],
    show_id: bool = True,
) -> np.ndarray:
    """
    Draw thin bounding boxes coloured by class with a small ID label.

    Design choices for dense small-object aerial imagery:
    - Box line thickness = 1 px (thin, non-occluding).
    - Font scale is adaptive: scales with image height so text is readable but
      does not dominate the box even on tiny detections.
    - Label background is a translucent filled rectangle so the text is legible
      over any background colour.
    - ID number only (no verbose class string in the label) to keep clutter low;
      the colour already encodes the class.
    """
    img = frame_img.copy()
    h_img, w_img = img.shape[:2]

    # Adaptive font size: 0.28 at 1080 p, scales linearly.
    font_scale   = max(0.28, h_img / 3840)
    font         = cv2.FONT_HERSHEY_SIMPLEX
    thickness_box  = 1                           # thin box
    thickness_text = 1

    default_color = (200, 200, 200)              # light grey for unknown classes

    for det in dets:
        cls   = det["cls"]
        color = palette.get(cls, default_color)

        x1 = int(round(det["x"]))
        y1 = int(round(det["y"]))
        x2 = int(round(det["x"] + det["w"]))
        y2 = int(round(det["y"] + det["h"]))

        # Clamp to frame bounds
        x1 = max(0, min(x1, w_img - 1))
        y1 = max(0, min(y1, h_img - 1))
        x2 = max(0, min(x2, w_img - 1))
        y2 = max(0, min(y2, h_img - 1))

        # ── Bounding box ──────────────────────────────────────────────────────
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness_box, cv2.LINE_AA)

        if not show_id:
            continue

        # ── ID label ─────────────────────────────────────────────────────────
        label = str(det["id"])
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, thickness_text)

        # Prefer label above the box; if too close to the top edge, put it inside.
        lx = x1
        ly = y1 - 2
        if ly - th < 0:
            ly = y1 + th + 2

        # Tiny filled background rectangle for readability
        cv2.rectangle(
            img,
            (lx, ly - th - baseline),
            (lx + tw, ly + baseline),
            color,
            cv2.FILLED,
        )

        # Choose black or white text based on colour luminance
        r_c, g_c, b_c = color[2] / 255, color[1] / 255, color[0] / 255
        lum = 0.299 * r_c + 0.587 * g_c + 0.114 * b_c
        text_color = (0, 0, 0) if lum > 0.5 else (255, 255, 255)

        cv2.putText(
            img, label,
            (lx, ly),
            font, font_scale, text_color, thickness_text, cv2.LINE_AA,
        )

    return img


# ── Legend ────────────────────────────────────────────────────────────────────

def draw_legend(
    frame_img: np.ndarray,
    palette: dict[int, tuple[int, int, int]],
    class_names: dict[int, str],
) -> np.ndarray:
    """
    Draw a compact class-colour legend in the top-right corner of the frame.
    Only classes that appear in the palette are shown.
    """
    img = frame_img.copy()
    h_img, w_img = img.shape[:2]

    font        = cv2.FONT_HERSHEY_SIMPLEX
    font_scale  = max(0.30, h_img / 4320)
    thickness   = 1
    swatch_size = max(8, int(h_img / 100))     # colour swatch square
    pad         = 4
    line_h      = swatch_size + pad

    classes = sorted(class_names.keys())
    n_cls   = len(classes)
    legend_h = n_cls * line_h + pad
    max_label_w = max(
        cv2.getTextSize(class_names.get(c, str(c)), font, font_scale, thickness)[0][0]
        for c in classes
    ) if classes else 80
    legend_w = swatch_size + pad + max_label_w + pad

    # Semi-transparent background
    x0 = w_img - legend_w - pad
    y0 = pad
    overlay = img.copy()
    cv2.rectangle(overlay, (x0 - pad, y0 - pad),
                  (w_img - pad, y0 + legend_h),
                  (30, 30, 30), cv2.FILLED)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)

    for i, cls_id in enumerate(classes):
        color = palette.get(cls_id, (200, 200, 200))
        name  = class_names.get(cls_id, str(cls_id))
        sy    = y0 + i * line_h
        # Colour swatch
        cv2.rectangle(img, (x0, sy), (x0 + swatch_size, sy + swatch_size),
                      color, cv2.FILLED)
        # Class name
        cv2.putText(img, name,
                    (x0 + swatch_size + pad, sy + swatch_size - 1),
                    font, font_scale, (220, 220, 220), thickness, cv2.LINE_AA)

    return img


# ── Frame helpers ─────────────────────────────────────────────────────────────

def collect_frame_paths(seq_dir: Path) -> dict[int, Path]:
    """
    Return a 1-indexed mapping of frame number → image path.
    Supports JPEG and PNG files with numeric stems (e.g. 0000001.jpg).
    """
    frame_map: dict[int, Path] = {}
    for p in sorted(seq_dir.iterdir()):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        try:
            idx = int(p.stem)
            frame_map[idx] = p
        except ValueError:
            pass
    return frame_map


# ── Main ──────────────────────────────────────────────────────────────────────

def visualise(
    seq_dir:    Path,
    ann_path:   Path,
    out_path:   Path,
    dataset:    str   = "visdrone",
    fps:        float = 25.0,
    start:      int   = 1,
    end:        int | None = None,
    save_frames: bool = False,
    show_id:    bool  = True,
    conf_min:      float = 0.0,
    ignore_classes: set[int] | None = None,
    all_classes:    bool  = True,
) -> None:
    class_names = ALL_CLASSES if all_classes else TRACKED_CLASSES
    palette     = ALL_PALETTE if all_classes else TRACKED_PALETTE

    if not all_classes and (ignore_classes is None or ignore_classes == {0}):
        ignore_classes = set(_NON_TRACKED)

    print(f"Loading annotations: {ann_path}")
    detections = load_annotations(ann_path, conf_min=conf_min,
                                  ignore_classes=ignore_classes)

    print(f"Collecting frames from: {seq_dir}")
    frame_map = collect_frame_paths(seq_dir)

    if not frame_map:
        raise FileNotFoundError(f"No image files found in {seq_dir}")

    first_frame = start
    last_frame  = end if end is not None else max(frame_map.keys())

    frame_indices = sorted(
        i for i in frame_map if first_frame <= i <= last_frame
    )
    if not frame_indices:
        raise ValueError(
            f"No frames found in range [{first_frame}, {last_frame}]. "
            f"Available: [{min(frame_map)}, {max(frame_map)}]"
        )

    print(f"Rendering frames {frame_indices[0]} – {frame_indices[-1]} "
          f"({len(frame_indices)} frames)")

    # Determine output mode
    video_writer: cv2.VideoWriter | None = None
    if save_frames:
        out_path.mkdir(parents=True, exist_ok=True)
        print(f"Saving frames to: {out_path}")
    else:
        # Probe first frame for video dimensions
        probe = cv2.imread(str(frame_map[frame_indices[0]]))
        if probe is None:
            raise IOError(f"Cannot read image: {frame_map[frame_indices[0]]}")
        h0, w0 = probe.shape[:2]
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w0, h0))
        print(f"Writing video: {out_path}  ({w0}×{h0} @ {fps} fps)")

    for frame_offset, frame_idx in enumerate(frame_indices):
        img_path = frame_map[frame_idx]
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  [WARN] Cannot read {img_path}, skipping.")
            continue

        dets = detections.get(frame_idx, [])
        img  = draw_detections(img, dets, palette, class_names, show_id=show_id)
        img  = draw_legend(img, palette, class_names)

        if save_frames:
            out_file = out_path / f"{frame_idx:07d}.jpg"
            cv2.imwrite(str(out_file), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        else:
            video_writer.write(img)  # type: ignore[union-attr]

        # Progress feedback every 50 frames
        if (frame_offset + 1) % 50 == 0 or (frame_offset + 1) == len(frame_indices):
            print(f"  [{frame_offset + 1:>5}/{len(frame_indices)}] frame {frame_idx}")

    if video_writer is not None:
        video_writer.release()

    print("\n✓ Done!")
    if not save_frames:
        print(f"  Video : {out_path.resolve()}")
    else:
        print(f"  Frames: {out_path.resolve()}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visualise MOT-challenge tracking results on video frames.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--seq",      required=True, type=Path,
                   help="Directory containing the JPEG/PNG frames.")
    p.add_argument("--ann",      required=True, type=Path,
                   help="MOT-challenge annotation .txt file.")
    p.add_argument("--out-dir",  type=Path, default=None,
                   help="Directory where the output file is written.  "
                        "Default: the parent folder of the sequence.")
    p.add_argument("--name",     type=str, default=None,
                   help="Output filename (with or without .mp4 extension).  "
                        "Default: <sequence_name>_vis.mp4.")
    p.add_argument("--dataset",  default="visdrone",
                   choices=["visdrone", "uavdt"],
                   help="Dataset type (determines class names).")
    p.add_argument("--fps",      type=float, default=25.0,
                   help="Output video frame rate.")
    p.add_argument("--start",    type=int, default=1,
                   help="First frame to render (1-indexed).")
    p.add_argument("--end",      type=int, default=None,
                   help="Last frame to render (inclusive). Defaults to last frame.")
    p.add_argument("--frames",   action="store_true",
                   help="Save individual frames instead of a video.")
    p.add_argument("--no-id",    action="store_true",
                   help="Suppress track ID labels.")
    p.add_argument("--conf-min", type=float, default=0.0,
                   help="Minimum confidence threshold (0.0 = show all).")
    p.add_argument("--ignore-cls", type=int, nargs="*", default=[0],
                   help="Class IDs to skip entirely (default: 0 = VisDrone ignore region). "
                        "Pass --ignore-cls with no values to show all classes.")
    p.add_argument("--all-classes", action="store_true", default=True,
                   help="Show all active classes (default: True). Kept for compatibility.")
    p.add_argument("--tracked-only", action="store_true",
                   help="Show only the 5 VisDrone tracked classes (pedestrian, car, van, truck, bus).")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    seq_dir = args.seq.resolve()
    if not seq_dir.is_dir():
        sys.exit(f"[ERROR] Sequence directory not found: {seq_dir}")

    # Resolve output directory
    out_dir = args.out_dir.resolve() if args.out_dir is not None else seq_dir.parent

    # Resolve output filename
    if args.name is not None:
        name = args.name
        # Append .mp4 if the user didn't provide an extension and we're writing a video
        if not args.frames and not Path(name).suffix:
            name = name + ".mp4"
    else:
        name = f"{seq_dir.name}_vis" + (".mp4" if not args.frames else "")

    out_path = out_dir / name

    visualise(
        seq_dir    = seq_dir,
        ann_path   = args.ann,
        out_path   = out_path,
        dataset    = args.dataset,
        fps        = args.fps,
        start      = args.start,
        end        = args.end,
        save_frames = args.frames,
        show_id        = not args.no_id,
        conf_min       = args.conf_min,
        ignore_classes = set(args.ignore_cls) if args.ignore_cls else set(),
        all_classes    = not args.tracked_only,
    )
