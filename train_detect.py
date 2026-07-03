#!/usr/bin/env python3
"""
train.py — Simple YOLO Training Script for UAV/Drone Object Detection
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Train Ultralytics YOLO models on UAV/Drone Detection Datasets")
    parser.add_argument("--model", "-m", type=str, default="yolov8s-p2.yaml", help="Model checkpoint or architecture")
    parser.add_argument("--data", "-d", type=str, default="datasets/VisDrone-YOLO/dataset_combined.yaml", help="Dataset YAML config file")
    parser.add_argument("--epochs", "-e", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch", "-b", type=int, default=-1, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=1280, help="Target image resolution")
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final learning rate factor")
    parser.add_argument("--warmup-epochs", "--warmup_epochs", dest="warmup_epochs", type=float, default=3.0, help="Warmup epochs")
    parser.add_argument("--cos-lr", "--cos_lr", dest="cos_lr", action="store_true", help="Use cosine learning rate scheduler")
    parser.add_argument("--mosaic", type=float, default=1.0, help="Image mosaic probability")
    parser.add_argument("--close-mosaic", "--close_mosaic", dest="close_mosaic", type=int, default=10, help="Disable mosaic augmentation for final N epochs")
    parser.add_argument("--scale", type=float, default=0.5, help="Image scale gain")
    parser.add_argument("--translate", type=float, default=0.1, help="Image translation fraction")
    parser.add_argument("--degrees", type=float, default=0.0, help="Image rotation degrees")
    parser.add_argument("--hsv-h", "--hsv_h", dest="hsv_h", type=float, default=0.015, help="Image HSV-Hue augmentation fraction")
    parser.add_argument("--hsv-s", "--hsv_s", dest="hsv_s", type=float, default=0.7, help="Image HSV-Saturation augmentation fraction")
    parser.add_argument("--hsv-v", "--hsv_v", dest="hsv_v", type=float, default=0.4, help="Image HSV-Value augmentation fraction")
    parser.add_argument("--max-det", "--max_det", dest="max_det", type=int, default=300, help="Maximum detections per image")
    parser.add_argument("--device", default="0", help="Computing device (e.g., '0' for GPU 0, or 'cpu')")
    parser.add_argument("--workers", type=int, default=8, help="Number of dataloader workers")
    parser.add_argument("--project", type=str, default=None, help="Save project directory")
    parser.add_argument("--name", type=str, default=None, help="Experiment run name")
    return parser.parse_args()


def main():
    args = parse_args()

    # Verify dataset configuration exists
    data_path = Path(args.data).resolve()
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset YAML configuration not found at: {data_path}")

    # Build dictionary of training arguments
    train_kwargs = {
        "data": str(data_path),
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "workers": args.workers,
        "warmup_epochs": args.warmup_epochs,
        "cos_lr": args.cos_lr,
        "mosaic": args.mosaic,
        "close_mosaic": args.close_mosaic,
        "scale": args.scale,
        "translate": args.translate,
        "degrees": args.degrees,
        "hsv_h": args.hsv_h,
        "hsv_s": args.hsv_s,
        "hsv_v": args.hsv_v,
        "max_det": args.max_det,
    }
    if args.device:
        train_kwargs["device"] = args.device
    if args.project:
        train_kwargs["project"] = args.project
    if args.name:
        train_kwargs["name"] = args.name

    print(f"[INFO] Initializing model: {args.model}")
    model = YOLO(args.model)

    # Transfer pretrained backbone weights when training custom YAML architectures
    model_name = Path(args.model).name
    if  model_name.endswith(".yaml") and "-p2" in model_name:
        base_weight = model_name.replace("-p2.yaml", ".pt")
        print(f"[INFO] Loading pretrained backbone weights from {base_weight}...")
        try:
            model.load(base_weight)
        except Exception as e:
            print(f"[WARNING] Could not automatically load {base_weight}: {e}")

    print(f"[INFO] Starting training on dataset: {data_path}")
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
