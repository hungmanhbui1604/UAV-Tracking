#!/usr/bin/env python3
"""
eval_detect.py — Simple YOLO Evaluation Script for UAV/Drone Object Detection
"""

import argparse
import csv
from pathlib import Path
import yaml
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Ultralytics YOLO models on UAV/Drone Detection Datasets")
    parser.add_argument("--model", "-m", type=str, default="yolov8s-p2.yaml", help="Model checkpoint weights")
    parser.add_argument("--data", "-d", type=str, default="datasets/VisDrone-YOLO/dataset_mot.yaml", help="Dataset YAML config file")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test", "train"], help="Dataset split to evaluate on")
    parser.add_argument("--batch", "-b", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=1280, help="Target image resolution")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--max-det", "--max_det", dest="max_det", type=int, default=1000, help="Maximum detections per image")
    parser.add_argument("--device", default="0", help="Computing device (e.g., '0' or 'cpu')")
    parser.add_argument("--workers", type=int, default=8, help="Number of dataloader workers")
    parser.add_argument("--project", type=str, default=None, help="Save project directory")
    parser.add_argument("--name", type=str, default=None, help="Experiment run name")
    return parser.parse_args()


def print_and_save_results(metrics, class_names=None):
    """
    Print and save evaluation results (mAP@50, mAP@50-95, P, R) both for average and per class.
    """
    header = f"{'Class':<25} {'mAP@50':<12} {'mAP@50-95':<12} {'P':<12} {'R':<12}"
    divider = "-" * len(header)
    eq_divider = "=" * len(header)
    lines = [
        eq_divider,
        "Evaluation Results (AP, AP@50-95, P, R)",
        eq_divider,
        header,
        divider,
    ]

    # Average metrics across all classes
    map50 = float(metrics.box.map50)
    map_val = float(metrics.box.map)
    mp = float(metrics.box.mp)
    mr = float(metrics.box.mr)

    avg_line = f"{'all':<25} {map50:<12.5f} {map_val:<12.5f} {mp:<12.5f} {mr:<12.5f}"
    lines.append(avg_line)
    lines.append(divider)

    # Per class metrics
    csv_rows = [
        ["Class", "AP", "AP@50-95", "P", "R"],
        ["all", f"{map50:.5f}", f"{map_val:.5f}", f"{mp:.5f}", f"{mr:.5f}"],
    ]

    for i, c_id in enumerate(metrics.ap_class_index):
        c_name = None
        if class_names is not None:
            if isinstance(class_names, dict):
                c_name = class_names.get(c_id) or class_names.get(int(c_id)) or class_names.get(str(c_id))
            elif isinstance(class_names, (list, tuple)) and 0 <= int(c_id) < len(class_names):
                c_name = class_names[int(c_id)]
        if c_name is None and hasattr(metrics, "names") and metrics.names:
            val = metrics.names.get(c_id) or metrics.names.get(int(c_id)) or metrics.names.get(str(c_id))
            if val is not None and str(val) != str(c_id):
                c_name = val
        if c_name is None:
            c_name = str(c_id)
        else:
            c_name = str(c_name)

        p = float(metrics.box.p[i])
        r = float(metrics.box.r[i])
        ap50 = float(metrics.box.ap50[i])
        ap = float(metrics.box.ap[i])

        class_line = f"{c_name:<25} {ap50:<12.5f} {ap:<12.5f} {p:<12.5f} {r:<12.5f}"
        lines.append(class_line)
        csv_rows.append([c_name, f"{ap50:.5f}", f"{ap:.5f}", f"{p:.5f}", f"{r:.5f}"])

    lines.append(eq_divider)

    # Print to console
    print("\n" + "\n".join(lines))

    # Save to file
    save_dir = getattr(metrics, "save_dir", None)
    if save_dir is None:
        save_dir = Path("runs/detect/val")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    txt_path = save_dir / "eval_results.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    csv_path = save_dir / "eval_results.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerows(csv_rows)

    print(f"\n[INFO] Evaluation results saved to:")
    print(f"       TXT: {txt_path}")
    print(f"       CSV: {csv_path}")


def main():
    args = parse_args()

    # Verify dataset configuration exists
    data_path = Path(args.data).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML configuration not found at: {data_path}")

    # Load class names from dataset configuration if available
    class_names = None
    try:
        with open(data_path, "r", encoding="utf-8") as f:
            data_cfg = yaml.safe_load(f)
            if isinstance(data_cfg, dict) and "names" in data_cfg:
                class_names = data_cfg["names"]
    except Exception as e:
        print(f"[WARNING] Could not load class names from {data_path}: {e}")

    # Build dictionary of evaluation arguments
    val_kwargs = {
        "data": str(data_path),
        "split": args.split,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "workers": args.workers,
        "conf": args.conf,
        "iou": args.iou,
        "max_det": args.max_det,
    }
    if args.device:
        val_kwargs["device"] = args.device
    if args.project:
        val_kwargs["project"] = args.project
    if args.name:
        val_kwargs["name"] = args.name

    print(f"[INFO] Initializing model: {args.model}")
    model = YOLO(args.model)

    if class_names is not None:
        try:
            if hasattr(model, "model") and model.model is not None and hasattr(model.model, "names"):
                model.model.names = class_names
        except Exception as e:
            print(f"[WARNING] Could not set model.model.names: {e}")

    print(f"[INFO] Starting evaluation on split '{args.split}' of dataset: {data_path}")
    metrics = model.val(**val_kwargs)
    print_and_save_results(metrics, class_names=class_names)


if __name__ == "__main__":
    main()


