#!/usr/bin/env python3
"""
convert_to_yolo.py — Unified VisDrone & DroneVehicle to YOLO Conversion Script

Supports converting:
  1. VisDrone2019-DET (images & .txt annotations)
  2. VisDrone2019-MOT (video sequences to flat images & .txt annotations with stride sampling)
  3. VisDrone-DroneVehicle (RGB images, 100px white border removal, polygon to HBB conversion,
     class merging into truck, and class-priority training sampling)

Usage:
  python convert_to_yolo.py --dataset det --src datasets/VisDrone --out datasets/VisDrone-YOLO
  python convert_to_yolo.py --dataset mot --src datasets/VisDrone --out datasets/VisDrone-YOLO --stride 5
  python convert_to_yolo.py --dataset dronevehicle --src datasets/VisDrone-DroneVehicle --out datasets/VisDrone-YOLO --dv-sample-ratio 0.5
  python convert_to_yolo.py --dataset all --src datasets/ --out datasets/VisDrone-YOLO --stride 5
"""

import os
import sys
import argparse
import shutil
import glob
import random
import math
from pathlib import Path
from collections import defaultdict, Counter
import xml.etree.ElementTree as ET
from PIL import Image

# Global Class Taxonomy (10-Class Standard)
CLASS_NAMES = {
    0: "pedestrian",
    1: "people",
    2: "bicycle",
    3: "car",
    4: "van",
    5: "truck",
    6: "tricycle",
    7: "awning-tricycle",
    8: "bus",
    9: "motor"
}

# DroneVehicle class name mapping to YOLO class IDs
DV_CLASS_MAPPING = {
    "car": 3,
    "van": 4,
    "truck": 5,
    "truvk": 5,          # typo handling
    "feright_car": 5,    # freight car merged into truck
    "feright car": 5,    # freight car variation
    "feright": 5,        # freight car typo
    "freight_car": 5,    # standard spelling
    "freight car": 5,    # standard spelling variation
    "freight": 5,        # standard spelling short
    "bus": 8
}

SKIP_CATS = {0, 11}  # VisDrone ignored regions (0) and others (11)


def convert_bbox(x, y, w, h, img_W, img_H):
    """Convert absolute bbox (left, top, width, height) to YOLO normalized (cx, cy, wn, hn)."""
    x1 = max(float(x), 0.0)
    y1 = max(float(y), 0.0)
    x2 = min(float(x + w), float(img_W))
    y2 = min(float(y + h), float(img_H))
    bw = x2 - x1
    bh = y2 - y1
    if bw <= 0 or bh <= 0:
        return None
    cx = (x1 + bw / 2.0) / float(img_W)
    cy = (y1 + bh / 2.0) / float(img_H)
    return cx, cy, bw / float(img_W), bh / float(img_H)


def to_yolo_line(x, y, w, h, cat, img_W, img_H, min_wh=0):
    """Convert one VisDrone annotation row to a YOLO label line string."""
    if cat in SKIP_CATS:
        return None, "cat_skip"
    result = convert_bbox(x, y, w, h, img_W, img_H)
    if result is None:
        return None, "clip_zero"
    cx, cy, wn, hn = result
    if wn * img_W < min_wh or hn * img_H < min_wh:
        return None, "min_wh_skip"
    yolo_cls = cat - 1
    if yolo_cls < 0 or yolo_cls >= len(CLASS_NAMES):
        return None, "invalid_class"
    return f"{yolo_cls} {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}", "valid"


def parse_row_det(line):
    parts = line.strip().split(',')
    x, y, w, h, score, cat = map(int, parts[:6])
    return x, y, w, h, score, cat


def parse_row_mot(line):
    parts = line.strip().split(',')
    fid, tid, x, y, w, h, score, cat = map(int, parts[:8])
    return fid, x, y, w, h, score, cat


def find_dataset_dir(src_root, candidates):
    """Find dataset directory under src_root matching candidate names or pattern."""
    src_path = Path(src_root)
    for c in candidates:
        p = src_path / c
        if p.exists() and p.is_dir():
            return p
    # Try pattern matching
    for c in candidates:
        matches = list(src_path.glob(c))
        if matches and matches[0].is_dir():
            return matches[0]
    return src_path


# ==============================================================================
# 1. VisDrone DET Converter
# ==============================================================================
def convert_det(src_root, out_root, splits, min_wh, report_stats):
    print("\n--- Starting VisDrone-DET Conversion ---")
    det_src = find_dataset_dir(src_root, ["VisDrone2019-DET-*", "*DET*", "VisDrone"])
    
    for split in splits:
        # Resolve split directory name
        split_dir_candidates = [
            det_src / f"VisDrone2019-DET-{split}",
            det_src / f"VisDrone2019-DET-{split}-dev",
            det_src / split
        ]
        split_dir = None
        for cand in split_dir_candidates:
            if cand.exists() and cand.is_dir():
                split_dir = cand
                break
        if not split_dir:
            print(f"[DET {split}] Warning: Source directory not found under {det_src}, skipping split.")
            continue

        src_img_dir = split_dir / "images"
        src_ann_dir = split_dir / "annotations"
        out_img_dir = out_root / "DET" / split / "images"
        out_lbl_dir = out_root / "DET" / split / "labels"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        img_paths = sorted(list(src_img_dir.glob("*.jpg")) + list(src_img_dir.glob("*.png")))
        print(f"[DET {split}] Processing {len(img_paths)} images from {src_img_dir}...")

        total_labels = 0
        for img_path in img_paths:
            stem = img_path.stem
            ann_path = src_ann_dir / f"{stem}.txt"
            dst_img = out_img_dir / img_path.name
            dst_lbl = out_lbl_dir / f"{stem}.txt"

            shutil.copy2(img_path, dst_img)
            report_stats["DET"][split]["frames_total"] += 1
            report_stats["DET"][split]["frames_sampled"] += 1

            if split == "test-dev" or split == "test" or not ann_path.exists():
                dst_lbl.write_text("")
                continue

            with Image.open(img_path) as img:
                img_W, img_H = img.size

            yolo_lines = []
            for line in ann_path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    x, y, w, h, score, cat = parse_row_det(line)
                except ValueError:
                    continue
                if score == 0:
                    report_stats["DET"][split]["skip_reasons"]["score_zero"] += 1
                    continue

                res_str, status = to_yolo_line(x, y, w, h, cat, img_W, img_H, min_wh)
                if status == "valid":
                    yolo_lines.append(res_str)
                    report_stats["DET"][split]["class_counts"][cat - 1] += 1
                    total_labels += 1
                else:
                    report_stats["DET"][split]["skip_reasons"][status] += 1

            dst_lbl.write_text("\n".join(yolo_lines))
        print(f"[DET {split}] Done. Wrote {total_labels} valid YOLO labels.")


# ==============================================================================
# 2. VisDrone MOT Converter
# ==============================================================================
def convert_mot(src_root, out_root, splits, train_stride, val_stride, min_wh, report_stats):
    print("\n--- Starting VisDrone-MOT Conversion ---")
    mot_src = find_dataset_dir(src_root, ["VisDrone2019-MOT-*", "*MOT*", "VisDrone"])

    for split in splits:
        split_dir_candidates = [
            mot_src / f"VisDrone2019-MOT-{split}",
            mot_src / f"VisDrone2019-MOT-{split}-dev",
            mot_src / split
        ]
        split_dir = None
        for cand in split_dir_candidates:
            if cand.exists() and cand.is_dir():
                split_dir = cand
                break
        if not split_dir:
            print(f"[MOT {split}] Warning: Source directory not found under {mot_src}, skipping split.")
            continue

        seq_root = split_dir / "sequences"
        ann_root = split_dir / "annotations"
        out_img_dir = out_root / "MOT" / split / "images"
        out_lbl_dir = out_root / "MOT" / split / "labels"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        stride = train_stride if split == "train" else (val_stride if split == "val" else 1)
        seq_dirs = sorted([d for d in seq_root.iterdir() if d.is_dir()])
        print(f"[MOT {split}] Processing {len(seq_dirs)} sequences with stride={stride}...")

        total_labels = 0
        for seq_dir in seq_dirs:
            seq_name = seq_dir.name
            ann_path = ann_root / f"{seq_name}.txt"
            report_stats["MOT"][split]["sequences"] += 1

            frame_anns = defaultdict(list)
            has_labels = (split != "test") and ann_path.exists()

            all_frames = sorted(list(seq_dir.glob("*.jpg")) + list(seq_dir.glob("*.png")))
            if not all_frames:
                continue

            with Image.open(all_frames[0]) as img:
                img_W, img_H = img.size

            if has_labels:
                for line in ann_path.read_text().splitlines():
                    if not line.strip():
                        continue
                    try:
                        fid, x, y, w, h, score, cat = parse_row_mot(line)
                    except ValueError:
                        continue
                    if score == 0:
                        report_stats["MOT"][split]["skip_reasons"]["score_zero"] += 1
                        continue
                    res_str, status = to_yolo_line(x, y, w, h, cat, img_W, img_H, min_wh)
                    if status == "valid":
                        frame_anns[fid].append((res_str, cat - 1))
                    else:
                        report_stats["MOT"][split]["skip_reasons"][status] += 1

            for frame_idx, frame_path in enumerate(all_frames):
                report_stats["MOT"][split]["frames_total"] += 1
                if frame_idx % stride != 0:
                    report_stats["MOT"][split]["skip_reasons"]["stride_skip"] += 1
                    continue

                report_stats["MOT"][split]["frames_sampled"] += 1
                try:
                    frame_id = int(frame_path.stem)
                except ValueError:
                    frame_id = frame_idx + 1

                out_stem = f"{seq_name}_{frame_path.stem}"
                dst_img = out_img_dir / f"{out_stem}.jpg"
                dst_lbl = out_lbl_dir / f"{out_stem}.txt"

                shutil.copy2(frame_path, dst_img)

                lines_to_write = []
                if has_labels and frame_id in frame_anns:
                    for res_str, cls_id in frame_anns[frame_id]:
                        lines_to_write.append(res_str)
                        report_stats["MOT"][split]["class_counts"][cls_id] += 1
                        total_labels += 1

                dst_lbl.write_text("\n".join(lines_to_write))

        print(f"[MOT {split}] Done. Wrote {total_labels} valid YOLO labels.")


# ==============================================================================
# 3. DroneVehicle Converter (100px border removal & Class-Priority Sampling)
# ==============================================================================
def convert_dronevehicle(src_root, out_root, splits, min_wh, sample_ratio, priority_order, seed, report_stats):
    print("\n--- Starting DroneVehicle Conversion ---")
    dv_src = find_dataset_dir(src_root, ["*DroneVehicle*", "VisDrone-DroneVehicle", "DroneVehicle"])

    for split in splits:
        # Map test-dev to test for DroneVehicle
        dv_split = "test" if split == "test-dev" else split
        split_dir = dv_src / dv_split
        if not split_dir.exists() or not split_dir.is_dir():
            print(f"[DroneVehicle {split}] Warning: Source directory {split_dir} not found under {dv_src}, skipping.")
            continue

        src_img_dir = split_dir / f"{dv_split}img"
        src_ann_dir = split_dir / f"{dv_split}label"
        if not src_img_dir.exists():
            src_img_dir = split_dir / "images"
        if not src_ann_dir.exists():
            src_ann_dir = split_dir / "annotations"

        # Target split folder in output (use test for test-dev)
        out_split = "test" if split == "test-dev" else split
        out_img_dir = out_root / "DroneVehicle" / out_split / "images"
        out_lbl_dir = out_root / "DroneVehicle" / out_split / "labels"
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)

        all_imgs = sorted(list(src_img_dir.glob("*.jpg")) + list(src_img_dir.glob("*.png")))
        print(f"[DroneVehicle {split}] Found {len(all_imgs)} images in {src_img_dir}...")

        # Class-Priority Sampling for train split
        selected_imgs = all_imgs
        if split == "train" and sample_ratio is not None:
            target_count = int(len(all_imgs) * sample_ratio)
            target_count = min(target_count, len(all_imgs))
            print(f"[DroneVehicle train] Applying Class-Priority Sampling: target = {target_count} images (out of {len(all_imgs)})...")

            # Map class priorities: higher score = rarer class
            p_map = {cls_name.strip(): len(priority_order) - idx for idx, cls_name in enumerate(priority_order)}
            
            scored_images = []
            tier_counts = Counter()
            for img_path in all_imgs:
                stem = img_path.stem
                ann_path = src_ann_dir / f"{stem}.xml"
                img_score = 0
                rarest_cls = "car/empty"
                if ann_path.exists():
                    try:
                        tree = ET.parse(ann_path)
                        for obj in tree.findall("object"):
                            name_el = obj.find("name")
                            if name_el is not None and name_el.text:
                                raw_name = name_el.text.strip().lower()
                                # Check if valid vehicle class
                                if raw_name in DV_CLASS_MAPPING:
                                    # Normalize name for scoring (e.g. feright_car -> truck)
                                    norm_name = "truck" if "feright" in raw_name or "freight" in raw_name or raw_name == "truvk" else raw_name
                                    score = p_map.get(norm_name, 0)
                                    if score > img_score:
                                        img_score = score
                                        rarest_cls = norm_name
                    except Exception:
                        pass
                scored_images.append((img_score, rarest_cls, img_path))
                tier_counts[rarest_cls] += 1

            print(f"[DroneVehicle train] Initial dataset composition by rarest class: {dict(tier_counts)}")

            # Group images by priority score descending
            by_score = defaultdict(list)
            for score, r_cls, path in scored_images:
                by_score[score].append((r_cls, path))

            rng = random.Random(seed)
            selected_imgs = []
            selected_tier_counts = Counter()
            for score in sorted(by_score.keys(), reverse=True):
                items = by_score[score]
                remaining = target_count - len(selected_imgs)
                if remaining <= 0:
                    break
                if len(items) <= remaining:
                    for r_cls, path in items:
                        selected_imgs.append(path)
                        selected_tier_counts[r_cls] += 1
                else:
                    # Sub-sample uniformly within this tier
                    sampled = rng.sample(items, remaining)
                    for r_cls, path in sampled:
                        selected_imgs.append(path)
                        selected_tier_counts[r_cls] += 1

            print(f"[DroneVehicle train] Selected dataset composition: {dict(selected_tier_counts)}")

        total_labels = 0
        for img_path in selected_imgs:
            stem = img_path.stem
            ann_path = src_ann_dir / f"{stem}.xml"
            dst_img = out_img_dir / img_path.name
            dst_lbl = out_lbl_dir / f"{stem}.txt"

            report_stats["DroneVehicle"][split]["frames_total"] += 1
            report_stats["DroneVehicle"][split]["frames_sampled"] += 1

            # 1. Open image and crop 100px white border (840x712 -> 640x512)
            try:
                with Image.open(img_path) as img:
                    orig_W, orig_H = img.size
                    # Crop border: left=100, top=100, right=orig_W-100, bottom=orig_H-100
                    left, top = 100, 100
                    right, bottom = max(100, orig_W - 100), max(100, orig_H - 100)
                    cropped_img = img.crop((left, top, right, bottom))
                    cropped_W, cropped_H = cropped_img.size
                    cropped_img.save(dst_img, quality=95)
            except Exception as e:
                print(f"[DroneVehicle {split}] Error cropping {img_path}: {e}")
                continue

            # 2. Convert XML annotations
            yolo_lines = []
            if ann_path.exists():
                try:
                    tree = ET.parse(ann_path)
                    for obj in tree.findall("object"):
                        name_el = obj.find("name")
                        if name_el is None or not name_el.text:
                            continue
                        raw_name = name_el.text.strip().lower()
                        if raw_name not in DV_CLASS_MAPPING:
                            report_stats["DroneVehicle"][split]["skip_reasons"]["unknown_class"] += 1
                            continue
                        yolo_cls = DV_CLASS_MAPPING[raw_name]

                        # Extract coordinates and shift by (-100, -100)
                        xs, ys = [], []
                        poly = obj.find("polygon")
                        if poly is not None:
                            for i in range(1, 5):
                                ex = poly.find(f"x{i}")
                                ey = poly.find(f"y{i}")
                                if ex is not None and ey is not None:
                                    xs.append(float(ex.text) - left)
                                    ys.append(float(ey.text) - top)
                        else:
                            bb = obj.find("bndbox")
                            if bb is not None:
                                xs = [float(bb.find("xmin").text) - left, float(bb.find("xmax").text) - left]
                                ys = [float(bb.find("ymin").text) - top, float(bb.find("ymax").text) - top]

                        if not xs or not ys:
                            continue

                        # Convert polygon/bbox to HBB enclosing box
                        xmin, xmax = min(xs), max(xs)
                        ymin, ymax = min(ys), max(ys)

                        # Clip to new image boundaries [0, cropped_W] x [0, cropped_H]
                        x1 = max(xmin, 0.0)
                        y1 = max(ymin, 0.0)
                        x2 = min(xmax, float(cropped_W))
                        y2 = min(ymax, float(cropped_H))
                        bw = x2 - x1
                        bh = y2 - y1

                        if bw <= 0 or bh <= 0:
                            report_stats["DroneVehicle"][split]["skip_reasons"]["clip_zero"] += 1
                            continue
                        if bw < min_wh or bh < min_wh:
                            report_stats["DroneVehicle"][split]["skip_reasons"]["min_wh_skip"] += 1
                            continue

                        cx = (x1 + bw / 2.0) / float(cropped_W)
                        cy = (y1 + bh / 2.0) / float(cropped_H)
                        wn = bw / float(cropped_W)
                        hn = bh / float(cropped_H)

                        yolo_lines.append(f"{yolo_cls} {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}")
                        report_stats["DroneVehicle"][split]["class_counts"][yolo_cls] += 1
                        total_labels += 1
                except Exception as e:
                    print(f"[DroneVehicle {split}] Error parsing XML {ann_path}: {e}")

            dst_lbl.write_text("\n".join(yolo_lines))

        print(f"[DroneVehicle {split}] Done. Wrote {total_labels} valid YOLO labels.")


# ==============================================================================
# 4. Dataset YAML Configuration Generator
# ==============================================================================
def generate_yaml_configs(out_root, converted_datasets):
    print("\n--- Generating Dataset YAML Configurations ---")
    out_path = Path(out_root).resolve()

    names_block = "\n".join([f"  {k}: {v}" for k, v in sorted(CLASS_NAMES.items())])

    if "det" in converted_datasets or "all" in converted_datasets:
        det_yaml = out_path / "dataset_det.yaml"
        content = f"""path: {out_path / 'DET'}
train: train/images
val: val/images
test: test-dev/images

nc: 10
names:
{names_block}
"""
        det_yaml.write_text(content)
        print(f"Generated: {det_yaml}")

    if "mot" in converted_datasets or "all" in converted_datasets:
        mot_yaml = out_path / "dataset_mot.yaml"
        content = f"""path: {out_path / 'MOT'}
train: train/images
val: val/images
test: test-dev/images

nc: 10
names:
{names_block}
"""
        mot_yaml.write_text(content)
        print(f"Generated: {mot_yaml}")

    if "dronevehicle" in converted_datasets or "all" in converted_datasets:
        dv_yaml = out_path / "dataset_dronevehicle.yaml"
        content = f"""path: {out_path / 'DroneVehicle'}
train: train/images
val: val/images
test: test/images

nc: 10
names:
{names_block}
"""
        dv_yaml.write_text(content)
        print(f"Generated: {dv_yaml}")

    # Combined master configuration
    train_paths = []
    val_paths = []
    test_paths = []
    if "det" in converted_datasets or "all" in converted_datasets:
        train_paths.append("  - DET/train/images")
    if "mot" in converted_datasets or "all" in converted_datasets:
        train_paths.append("  - MOT/train/images")
        val_paths.append("  - MOT/val/images")
        test_paths.append("  - MOT/test-dev/images")
    if "dronevehicle" in converted_datasets or "all" in converted_datasets:
        train_paths.append("  - DroneVehicle/train/images")

    combined_yaml = out_path / "dataset_combined.yaml"
    content = f"""path: {out_path}
train:
{chr(10).join(train_paths)}
val:
{chr(10).join(val_paths)}
test:
{chr(10).join(test_paths)}

nc: 10
names:
{names_block}
"""
    combined_yaml.write_text(content)
    print(f"Generated: {combined_yaml}")


# ==============================================================================
# 5. Summary Reporter
# ==============================================================================
def print_and_save_report(out_root, report_stats):
    lines = []
    lines.append("\n================================================================================")
    lines.append("                     VisDrone & DroneVehicle YOLO Conversion Report")
    lines.append("================================================================================\n")

    for ds_name, ds_stats in report_stats.items():
        if not any(ds_stats[s]["frames_sampled"] > 0 for s in ds_stats):
            continue
        lines.append(f"=== Dataset: {ds_name} ===")
        lines.append(f"{'Split':<10} | {'Sequences':<10} | {'Frames (Total)':<15} | {'Frames (Sampled)':<18} | {'Valid Labels':<14}")
        lines.append("-" * 75)
        for split, s_data in ds_stats.items():
            total_lbls = sum(s_data["class_counts"].values())
            lines.append(f"{split:<10} | {s_data.get('sequences', 0):<10} | {s_data['frames_total']:<15} | {s_data['frames_sampled']:<18} | {total_lbls:<14}")
        
        lines.append("\nSkip Reasons across splits:")
        for split, s_data in ds_stats.items():
            skips = {k: v for k, v in s_data["skip_reasons"].items() if v > 0}
            if skips:
                lines.append(f"  [{split}]: {dict(skips)}")

        lines.append("\nClass Distribution (Sampled Labels):")
        total_cls_counts = Counter()
        for split, s_data in ds_stats.items():
            for cid, cnt in s_data["class_counts"].items():
                total_cls_counts[cid] += cnt
        for cid in sorted(CLASS_NAMES.keys()):
            cnt = total_cls_counts[cid]
            if cnt > 0:
                lines.append(f"  {cid:2d} {CLASS_NAMES[cid]:<18}: {cnt:,}")
        lines.append("-" * 75 + "\n")

    report_str = "\n".join(lines)
    print(report_str)
    report_file = Path(out_root) / "conversion_report.txt"
    report_file.write_text(report_str)
    print(f"Full conversion report saved to: {report_file}")


# ==============================================================================
# Main Entry Point
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Unified VisDrone & DroneVehicle to YOLO Converter")
    parser.add_argument("--dataset", "-d", nargs="+", choices=["det", "mot", "dronevehicle", "all"], default=["all"],
                        help="Which dataset(s) to convert (default: all)")
    parser.add_argument("--src", "-s", type=str, default="datasets", help="Root source directory containing datasets")
    parser.add_argument("--out", "-o", type=str, default="datasets/VisDrone-YOLO", help="Output root directory")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test-dev"],
                        help="Splits to process (default: train val test-dev). Note: test-dev maps to test for DroneVehicle.")
    parser.add_argument("--stride", type=int, default=3, help="Frame sampling stride for MOT train sequences (default: 3)")
    parser.add_argument("--val-stride", type=int, default=1, help="Frame sampling stride for MOT val sequences (default: 1)")
    parser.add_argument("--min-wh", type=float, default=2.0, help="Minimum box width/height in pixels after clipping (default: 2.0)")
    
    # DroneVehicle class-priority sampling arguments
    parser.add_argument("--dv-sample-ratio", type=float, default=0.5, help="Ratio of training images to sample from DroneVehicle (default: 0.5)")
    parser.add_argument("--dv-priority-order", type=str, default="bus,van,truck,car",
                        help="Comma-separated class priority ranking for DroneVehicle sampling (default: bus,van,truck,car)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling (default: 42)")

    args = parser.parse_args()

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    # Initialize stats data structure
    report_stats = defaultdict(lambda: defaultdict(lambda: {
        "sequences": 0,
        "frames_total": 0,
        "frames_sampled": 0,
        "class_counts": Counter(),
        "skip_reasons": Counter()
    }))

    datasets_to_run = set(args.dataset)
    if "all" in datasets_to_run:
        datasets_to_run = {"det", "mot", "dronevehicle"}

    priority_order = [x.strip() for x in args.dv_priority_order.split(",") if x.strip()]

    if "det" in datasets_to_run:
        convert_det(args.src, out_root, args.splits, args.min_wh, report_stats)
    if "mot" in datasets_to_run:
        convert_mot(args.src, out_root, args.splits, args.stride, args.val_stride, args.min_wh, report_stats)
    if "dronevehicle" in datasets_to_run:
        convert_dronevehicle(args.src, out_root, args.splits, args.min_wh,
                             args.dv_sample_ratio, priority_order, args.seed, report_stats)

    generate_yaml_configs(out_root, datasets_to_run)
    print_and_save_report(out_root, report_stats)


if __name__ == "__main__":
    main()
