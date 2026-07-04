"""
track_visdrone.py
-----------------
Run finetuned YOLO + BoT-SORT tracking on the VisDrone-MOT test-dev set and
save per-sequence predictions in MOT-challenge format.

VisDrone-MOT layout:
    datasets/VisDrone/VisDrone2019-MOT-test-dev/sequences/{seq_name}/*.jpg

Output (one .txt per sequence):
    runs/tracking/visdrone/test-dev/annotations/{seq_name}.txt
    Format: <frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<class>,-1,-1

VisDrone class mapping (finetuned model → VisDrone 1-indexed submission):
    YOLO 0 pedestrian       → VisDrone class 1
    YOLO 1 people           → VisDrone class 2
    YOLO 2 bicycle          → VisDrone class 3
    YOLO 3 car              → VisDrone class 4
    YOLO 4 van              → VisDrone class 5
    YOLO 5 truck            → VisDrone class 6
    YOLO 6 tricycle         → VisDrone class 7
    YOLO 7 awning-tricycle  → VisDrone class 8
    YOLO 8 bus              → VisDrone class 9
    YOLO 9 motor            → VisDrone class 10
    (Mapped to the original VisDrone 12-class taxonomy used by the benchmark)
"""

from pathlib import Path

from ultralytics import YOLO

# ── Configuration ──────────────────────────────────────────────────────────────
# Point this to your finetuned best.pt; falls back to yolo11s.pt if not found.
MODEL_PATH = "runs/detect/train/weights/best.pt"
FALLBACK_MODEL_PATH = ""

SEQUENCES_DIR = "datasets/VisDrone/VisDrone2019-MOT-test-dev/sequences"
TRACKER_CONFIG = "visdrone_botsort.yaml"  # swap to visdrone_bytetrack.yaml to use ByteTrack

OUTPUT_DIR = Path("runs/tracking/visdrone/test-dev/annotations")

# Detection confidence threshold. Matches track_high_thresh in the YAML so the
# tracker and detector are aligned on what counts as a valid detection.
CONF_THRESH = 0.10
# NMS IoU threshold. Lowered to 0.45 (vs 0.5 default) so densely-packed objects
# in VisDrone (crowds, parked cars) are less likely to suppress each other.
IOU_THRESH = 0.60
# 1280 px input is essential for resolving the small objects common in aerial footage.
IMG_SIZE = 1280

# Map from YOLO class index → VisDrone benchmark class (1-indexed, 12-class taxonomy)
# Ref: https://github.com/VisDrone/VisDrone-Dataset
YOLO_TO_VISDRONE_CLASS = {
    0:  1,  # pedestrian
    1:  2,  # people
    2:  3,  # bicycle
    3:  4,  # car
    4:  5,  # van
    5:  6,  # truck
    6:  7,  # tricycle
    7:  8,  # awning-tricycle
    8:  9,  # bus
    9: 10,  # motor
}
# ──────────────────────────────────────────────────────────────────────────────


def reset_tracker(model: YOLO) -> None:
    """Reset BoT-SORT internal state between sequences."""
    if hasattr(model, "predictor") and model.predictor is not None:
        if hasattr(model.predictor, "trackers"):
            for tracker in model.predictor.trackers:
                tracker.reset()


def track_visdrone() -> None:
    model_path = Path(MODEL_PATH)
    if not model_path.exists():
        print(f"[WARNING] Model not found at '{MODEL_PATH}', falling back to '{FALLBACK_MODEL_PATH}'")
        model_path = Path(FALLBACK_MODEL_PATH)

    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))

    seq_root = Path(SEQUENCES_DIR)
    if not seq_root.exists():
        raise FileNotFoundError(f"Sequences directory not found: {seq_root.resolve()}")

    sequence_paths = sorted([p for p in seq_root.iterdir() if p.is_dir()])
    print(f"Found {len(sequence_paths)} sequences")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for seq_path in sequence_paths:
        seq_name = seq_path.name
        output_file = OUTPUT_DIR / f"{seq_name}.txt"
        print(f"\n[{seq_name}] Tracking → {output_file}")

        # Reset tracker state from the previous sequence
        reset_tracker(model)

        results = model.track(
            source=str(seq_path),
            tracker=TRACKER_CONFIG,
            imgsz=IMG_SIZE,
            conf=CONF_THRESH,
            iou=IOU_THRESH,
            persist=True,
            stream=True,
            verbose=False,
            save=False,  # Set to True to save annotated frames
        )

        with open(output_file, "w") as f:
            for frame_idx, r in enumerate(results, start=1):
                if r.boxes is None or r.boxes.id is None:
                    continue

                boxes = r.boxes.xyxy.cpu().numpy()      # [N, 4] x1 y1 x2 y2
                track_ids = r.boxes.id.int().cpu().numpy()
                confs = r.boxes.conf.cpu().numpy()
                classes = r.boxes.cls.int().cpu().numpy()

                for box, track_id, conf, cls in zip(boxes, track_ids, confs, classes):
                    x1, y1, x2, y2 = box
                    w, h = x2 - x1, y2 - y1
                    # Remap to VisDrone 1-indexed class taxonomy
                    visdrone_class = YOLO_TO_VISDRONE_CLASS.get(int(cls), int(cls) + 1)
                    # Format: <frame>,<id>,<bb_left>,<bb_top>,<bb_width>,<bb_height>,<conf>,<class>,-1,-1
                    f.write(
                        f"{frame_idx},{track_id},{x1:.2f},{y1:.2f},{w:.2f},{h:.2f}"
                        f",{conf:.4f},{visdrone_class},-1,-1\n"
                    )

        print(f"  ✓ Saved: {output_file}")

    print(f"\nDone! Results in: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    track_visdrone()