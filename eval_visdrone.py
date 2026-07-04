"""
evaluate_visdrone.py
--------------------
Category-aware MOT evaluation on the VisDrone test-dev split.

Metrics  : HOTA, DetA, AssA, LocA, DetPr, DetRe, AssPr, AssRe (α ∈ [0.05,0.95]),
             MOTA, MOTP, MT, PT, ML, FP, FN, IDS, Frag (CLEAR),
             IDF1, IDP, IDR (Identity)
Classes  : pedestrian (1), car (4), van (5), truck (6), bus (9)
Aggregations:
  - class-averaged   : arithmetic mean across classes (equal weight per class)
  - detection-averaged: weighted mean, weight = number of GT detections per class

Usage:
    conda run -n yolo python evaluate_visdrone.py [options]

Options:
    --gt-dir    PATH  Path to GT annotations dir
                      (default: datasets/VisDrone/VisDrone2019-MOT-test-dev/annotations)
    --pred-dir  PATH  Path to predicted annotations dir
                      (default: runs/tracking/visdrone/test-dev/annotations)
    --iou       FLOAT IoU threshold for MOTA/IDF1 matching  (default: 0.5)
    --out       PATH  Path for JSON results file             (default: evaluate_results.json)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_GT_DIR   = "datasets/VisDrone/VisDrone2019-MOT-test-dev/annotations"
DEFAULT_PRED_DIR = "botsort-tensorrt"
DEFAULT_IOU      = 0.5
DEFAULT_OUT      = "botsort_tensorrt_tracking_eval_result.json"

# VisDrone annotations use class 0 for ignored regions and class 11 for
# objects outside the evaluated categories ("others"). Predictions whose
# area lies at least 50% inside one of these regions are excluded from scoring.
IGNORED_CLASSES = frozenset({0, 11})
IGNORE_IOA_THRESHOLD = 0.5

# VisDrone class taxonomy used in this experiment
CLASSES: Dict[int, str] = {
    1: "pedestrian",
    4: "car",
    5: "van",
    6: "truck",
    9: "bus",
}

# ──────────────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────────────

def load_annotations(txt_path: Path) -> Tuple[
    Dict[int, np.ndarray],   # frame -> boxes  [N,4] (x,y,w,h)
    Dict[int, np.ndarray],   # frame -> ids    [N]
    Dict[int, np.ndarray],   # frame -> classes[N]
    Dict[int, np.ndarray],   # frame -> confs  [N]
]:
    """
    Parse a VisDrone/MOT-Challenge annotation file.

    CSV format per row:
        frame_id, object_id, bb_left, bb_top, bb_width, bb_height,
        conf, class, ?, ?

    Returns four dicts keyed by integer frame_id.
    """
    boxes   = defaultdict(list)
    ids     = defaultdict(list)
    classes = defaultdict(list)
    confs   = defaultdict(list)

    with open(txt_path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 8:
                continue
            frame_id  = int(parts[0])
            obj_id    = int(parts[1])
            bb_left   = float(parts[2])
            bb_top    = float(parts[3])
            bb_width  = float(parts[4])
            bb_height = float(parts[5])
            conf      = float(parts[6])
            cls       = int(parts[7])

            boxes[frame_id].append([bb_left, bb_top, bb_width, bb_height])
            ids[frame_id].append(obj_id)
            classes[frame_id].append(cls)
            confs[frame_id].append(conf)

    return (
        {f: np.asarray(v, dtype=np.float32) for f, v in boxes.items()},
        {f: np.asarray(v, dtype=np.int32)   for f, v in ids.items()},
        {f: np.asarray(v, dtype=np.int32)   for f, v in classes.items()},
        {f: np.asarray(v, dtype=np.float32) for f, v in confs.items()},
    )


def _calculate_box_ious(
    bboxes1: np.ndarray,
    bboxes2: np.ndarray,
    do_ioa: bool = False,
) -> np.ndarray:
    """
    Vectorised IoU (or IoA) between two sets of xywh boxes.
    Mirrors TrackEval's _BaseDataset._calculate_box_ious for consistency.
    """
    b1 = deepcopy(bboxes1).astype(np.float64)
    b2 = deepcopy(bboxes2).astype(np.float64)
    # Convert xywh -> xyxy
    b1[:, 2] += b1[:, 0]
    b1[:, 3] += b1[:, 1]
    b2[:, 2] += b2[:, 0]
    b2[:, 3] += b2[:, 1]

    # Intersections
    mins = np.minimum(b1[:, np.newaxis, :], b2[np.newaxis, :, :])
    maxs = np.maximum(b1[:, np.newaxis, :], b2[np.newaxis, :, :])
    inter = (
        np.maximum(mins[..., 2] - maxs[..., 0], 0)
        * np.maximum(mins[..., 3] - maxs[..., 1], 0)
    )

    area1 = (b1[:, 2] - b1[:, 0]) * (b1[:, 3] - b1[:, 1])
    if do_ioa:
        ioas = np.zeros_like(inter)
        valid = area1 > 0 + np.finfo("float").eps
        ioas[valid, :] = inter[valid, :] / area1[valid, np.newaxis]
        return ioas

    area2 = (b2[:, 2] - b2[:, 0]) * (b2[:, 3] - b2[:, 1])
    union = area1[:, np.newaxis] + area2[np.newaxis, :] - inter
    inter[area1 <= 0 + np.finfo("float").eps, :] = 0
    inter[:, area2 <= 0 + np.finfo("float").eps] = 0
    inter[union <= 0 + np.finfo("float").eps] = 0
    union[union <= 0 + np.finfo("float").eps] = 1
    return inter / union


# ──────────────────────────────────────────────────────────────────────────────
# Build per-class TrackEval data dict for a single sequence
# ──────────────────────────────────────────────────────────────────────────────

def build_sequence_data(
    gt_boxes:   Dict[int, np.ndarray],
    gt_ids:     Dict[int, np.ndarray],
    gt_classes: Dict[int, np.ndarray],
    gt_confs:   Dict[int, np.ndarray],
    pr_boxes:   Dict[int, np.ndarray],
    pr_ids:     Dict[int, np.ndarray],
    pr_classes: Dict[int, np.ndarray],
    target_cls: int,
) -> Dict:
    """
    Build the data dict expected by TrackEval metric.eval_sequence() for one
    sequence and one class.

    Active GT consists only of rows from ``target_cls`` with conf > 0.
    Ignored regions consist of rows with conf == 0, class 0 (ignored region),
    or class 11 (others). A prediction is excluded when at least 50% of the
    *prediction area* lies inside any ignored region.

    Return keys required by TrackEval:
        num_timesteps, num_gt_ids, num_tracker_ids,
        num_gt_dets, num_tracker_dets,
        gt_ids         : list[np.ndarray(int)]   length = num_timesteps
        tracker_ids    : list[np.ndarray(int)]   length = num_timesteps
        similarity_scores: list[np.ndarray(float)] shape [|gt_t|, |pred_t|]
    """
    all_frames = sorted(
        set(gt_boxes.keys()) | set(pr_boxes.keys())
    )
    if not all_frames:
        # Empty sequence
        return {
            "num_timesteps": 0,
            "num_gt_ids": 0,
            "num_tracker_ids": 0,
            "num_gt_dets": 0,
            "num_tracker_dets": 0,
            "gt_ids": [],
            "tracker_ids": [],
            "similarity_scores": [],
        }

    num_frames = max(all_frames)

    # Remap raw object IDs -> 0-indexed contiguous
    gt_id_map:  Dict[int, int] = {}
    pr_id_map:  Dict[int, int] = {}

    out_gt_ids: List[np.ndarray] = []
    out_pr_ids: List[np.ndarray] = []
    out_sims:   List[np.ndarray] = []
    num_gt_dets = 0
    num_pr_dets = 0

    for frame in range(1, num_frames + 1):
        # ── Ground-truth for this class ──────────────────────────────────────
        gt_b_active     = np.zeros((0, 4), dtype=np.float32)
        gt_b_ignored    = np.zeros((0, 4), dtype=np.float32)
        raw_gt_ids_active: np.ndarray = np.zeros(0, dtype=np.int32)

        if frame in gt_boxes:
            gc = gt_classes[frame]
            gf = gt_confs[frame]
            active_mask = (gc == target_cls) & (gf > 0)

            # Ignored regions are class-independent. In particular, class 0
            # marks ignored regions and class 11 marks "others" in VisDrone.
            # Rows explicitly marked with conf == 0 are ignored as well.
            ignored_mask = (gf == 0) | np.isin(gc, tuple(IGNORED_CLASSES))

            gt_b_active = gt_boxes[frame][active_mask]
            gt_b_ignored = gt_boxes[frame][ignored_mask]
            raw_gt_ids_active = gt_ids[frame][active_mask]

        # Remap GT IDs
        mapped_gt = np.array(
            [
                gt_id_map.setdefault(rid, len(gt_id_map))
                for rid in raw_gt_ids_active
            ],
            dtype=np.int32,
        )

        # ── Predictions for this class ────────────────────────────────────────
        pr_b: np.ndarray = np.zeros((0, 4), dtype=np.float32)
        raw_pr_ids: np.ndarray = np.zeros(0, dtype=np.int32)

        if frame in pr_boxes:
            pc = pr_classes[frame]
            cls_mask_pr = (pc == target_cls)
            pr_b        = pr_boxes[frame][cls_mask_pr]
            raw_pr_ids  = pr_ids[frame][cls_mask_pr]

        # Filter predictions covered by ignored GT regions. The IoA helper
        # divides by the area of its first argument, therefore predictions
        # must be passed first: intersection(pred, ignore) / area(pred).
        if len(pr_b) > 0 and len(gt_b_ignored) > 0:
            ioa = _calculate_box_ious(pr_b, gt_b_ignored, do_ioa=True)
            # Shape: [num_predictions, num_ignored_regions].
            keep = ioa.max(axis=1) < IGNORE_IOA_THRESHOLD
            pr_b = pr_b[keep]
            raw_pr_ids = raw_pr_ids[keep]

        # Remap tracker IDs
        mapped_pr = np.array(
            [
                pr_id_map.setdefault(rid, len(pr_id_map))
                for rid in raw_pr_ids
            ],
            dtype=np.int32,
        )

        # ── Similarity matrix (IoU) ───────────────────────────────────────────
        if len(mapped_gt) > 0 and len(mapped_pr) > 0:
            sim = _calculate_box_ious(gt_b_active, pr_b)
        else:
            sim = np.zeros((len(mapped_gt), len(mapped_pr)), dtype=np.float32)

        out_gt_ids.append(mapped_gt)
        out_pr_ids.append(mapped_pr)
        out_sims.append(sim)

        num_gt_dets += len(mapped_gt)
        num_pr_dets += len(mapped_pr)

    return {
        "num_timesteps":     num_frames,
        "num_gt_ids":        len(gt_id_map),
        "num_tracker_ids":   len(pr_id_map),
        "num_gt_dets":       num_gt_dets,
        "num_tracker_dets":  num_pr_dets,
        "gt_ids":            out_gt_ids,
        "tracker_ids":       out_pr_ids,
        "similarity_scores": out_sims,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(gt_dir: Path, pred_dir: Path, iou_thresh: float) -> Dict:
    from trackeval.metrics.hota     import HOTA
    from trackeval.metrics.clear    import CLEAR
    from trackeval.metrics.identity import Identity

    # Instantiate metrics (CLEAR / Identity use configurable IoU threshold)
    hota_metric  = HOTA()
    clear_metric = CLEAR({"THRESHOLD": iou_thresh})
    id_metric    = Identity({"THRESHOLD": iou_thresh})

    # Accumulate per-class per-sequence results
    # Structure: class_id -> seq_name -> result_dict
    hota_seq:  Dict[int, Dict[str, Dict]] = {c: {} for c in CLASSES}
    clear_seq: Dict[int, Dict[str, Dict]] = {c: {} for c in CLASSES}
    id_seq:    Dict[int, Dict[str, Dict]] = {c: {} for c in CLASSES}
    # Accumulate unique GT/pred track counts per class (from the data dict,
    # not the metric result dict which does not carry these fields)
    track_counts: Dict[int, Dict[str, int]] = {
        c: {"gt_tracks": 0, "pred_tracks": 0} for c in CLASSES
    }

    gt_files = sorted(gt_dir.glob("*.txt"))
    pred_files = {p.name: p for p in pred_dir.glob("*.txt")}

    if not gt_files:
        print(f"[ERROR] No GT files found in {gt_dir}")
        sys.exit(1)

    sequences_evaluated = 0

    for gt_path in gt_files:
        seq_name = gt_path.stem
        if gt_path.name not in pred_files:
            raise FileNotFoundError(
                f"Missing prediction file for sequence '{seq_name}': "
                f"expected '{pred_dir / gt_path.name}'. "
                "Evaluation aborted so the sequence is not silently skipped."
            )

        pred_path = pred_files[gt_path.name]
        print(f"  Processing: {seq_name}", flush=True)

        gt_b, gt_i, gt_c, gt_f = load_annotations(gt_path)
        pr_b, pr_i, pr_c, _    = load_annotations(pred_path)

        for cls_id in CLASSES:
            data = build_sequence_data(
                gt_b, gt_i, gt_c, gt_f,
                pr_b, pr_i, pr_c,
                target_cls=cls_id,
            )
            hota_seq[cls_id][seq_name]  = hota_metric.eval_sequence(data)
            clear_seq[cls_id][seq_name] = clear_metric.eval_sequence(data)
            id_seq[cls_id][seq_name]    = id_metric.eval_sequence(data)
            # Accumulate unique track IDs from the data dict (available here)
            track_counts[cls_id]["gt_tracks"]   += data["num_gt_ids"]
            track_counts[cls_id]["pred_tracks"] += data["num_tracker_ids"]

        sequences_evaluated += 1

    print(f"\nEvaluated {sequences_evaluated}/{len(gt_files)} sequences.\n")

    # ── Combine sequences -> per-class combined results ───────────────────────
    per_class: Dict[int, Dict] = {}

    for cls_id, cls_name in CLASSES.items():
        h_comb = hota_metric.combine_sequences(hota_seq[cls_id])
        c_comb = clear_metric.combine_sequences(clear_seq[cls_id])
        i_comb = id_metric.combine_sequences(id_seq[cls_id])

        n_gt = int(sum(
            seq_res["CLR_TP"] + seq_res["CLR_FN"]
            for seq_res in clear_seq[cls_id].values()
        ))

        # Sum raw integer counts across sequences (not averaged)
        n_ids  = int(sum(s["IDSW"] for s in clear_seq[cls_id].values()))
        n_frag = int(sum(s["Frag"] for s in clear_seq[cls_id].values()))

        per_class[cls_id] = {
            "name":    cls_name,
            # ── Detection / track counts ──────────────────────────────────────
            "gt_dets":    n_gt,
            "pred_dets":  int(c_comb["CLR_TP"]) + int(c_comb["CLR_FP"]),
            "gt_tracks":  track_counts[cls_id]["gt_tracks"],
            "pred_tracks": track_counts[cls_id]["pred_tracks"],
            # ── HOTA family (mean over 19 α-thresholds, ×100) ─────────────────
            "HOTA":   float(np.mean(h_comb["HOTA"])  * 100),
            "DetA":   float(np.mean(h_comb["DetA"])  * 100),
            "AssA":   float(np.mean(h_comb["AssA"])  * 100),
            "LocA":   float(np.mean(h_comb["LocA"])  * 100),
            "DetPr":  float(np.mean(h_comb["DetPr"]) * 100),
            "DetRe":  float(np.mean(h_comb["DetRe"]) * 100),
            "AssPr":  float(np.mean(h_comb["AssPr"]) * 100),
            "AssRe":  float(np.mean(h_comb["AssRe"]) * 100),
            # ── CLEAR metrics (×100 for %, raw int for counts) ────────────────
            "MOTA":   float(c_comb["MOTA"] * 100),
            "MOTP":   float(c_comb["MOTP"] * 100),
            "MT":     int(c_comb["MT"]),
            "PT":     int(c_comb["PT"]),
            "ML":     int(c_comb["ML"]),
            "FP":     int(c_comb["CLR_FP"]),
            "FN":     int(c_comb["CLR_FN"]),
            "IDS":    n_ids,
            "Frag":   n_frag,
            # ── Identity metrics (×100) ───────────────────────────────────────
            "IDF1":   float(i_comb["IDF1"] * 100),
            "IDP":    float(i_comb["IDP"]  * 100),
            "IDR":    float(i_comb["IDR"]  * 100),
            # Raw combined results for class/det aggregation
            "_hota_comb":  h_comb,
            "_clear_comb": c_comb,
            "_id_comb":    i_comb,
        }

    # ── Class-averaged aggregation ────────────────────────────────────────────
    h_cls_avg = hota_metric.combine_classes_class_averaged(
        {c: per_class[c]["_hota_comb"]  for c in CLASSES}, ignore_empty_classes=True
    )
    c_cls_avg = clear_metric.combine_classes_class_averaged(
        {c: per_class[c]["_clear_comb"] for c in CLASSES}, ignore_empty_classes=True
    )
    i_cls_avg = id_metric.combine_classes_class_averaged(
        {c: per_class[c]["_id_comb"]    for c in CLASSES}, ignore_empty_classes=True
    )
    cls_avg = {
        "HOTA":  float(np.mean(h_cls_avg["HOTA"])  * 100),
        "DetA":  float(np.mean(h_cls_avg["DetA"])  * 100),
        "AssA":  float(np.mean(h_cls_avg["AssA"])  * 100),
        "LocA":  float(np.mean(h_cls_avg["LocA"])  * 100),
        "DetPr": float(np.mean(h_cls_avg["DetPr"]) * 100),
        "DetRe": float(np.mean(h_cls_avg["DetRe"]) * 100),
        "AssPr": float(np.mean(h_cls_avg["AssPr"]) * 100),
        "AssRe": float(np.mean(h_cls_avg["AssRe"]) * 100),
        "MOTA":  float(c_cls_avg["MOTA"] * 100),
        "MOTP":  float(c_cls_avg["MOTP"] * 100),
        "MT":    int(sum(per_class[c]["MT"]   for c in CLASSES)),
        "PT":    int(sum(per_class[c]["PT"]   for c in CLASSES)),
        "ML":    int(sum(per_class[c]["ML"]   for c in CLASSES)),
        "FP":    int(sum(per_class[c]["FP"]   for c in CLASSES)),
        "FN":    int(sum(per_class[c]["FN"]   for c in CLASSES)),
        "IDS":   int(sum(per_class[c]["IDS"]  for c in CLASSES)),
        "Frag":  int(sum(per_class[c]["Frag"] for c in CLASSES)),
        "IDF1":  float(i_cls_avg["IDF1"] * 100),
        "IDP":   float(i_cls_avg["IDP"]  * 100),
        "IDR":   float(i_cls_avg["IDR"]  * 100),
    }

    # ── Detection-averaged aggregation ────────────────────────────────────────
    h_det_avg = hota_metric.combine_classes_det_averaged(
        {c: per_class[c]["_hota_comb"]  for c in CLASSES}
    )
    c_det_avg = clear_metric.combine_classes_det_averaged(
        {c: per_class[c]["_clear_comb"] for c in CLASSES}
    )
    i_det_avg = id_metric.combine_classes_det_averaged(
        {c: per_class[c]["_id_comb"]    for c in CLASSES}
    )
    det_avg = {
        "HOTA":  float(np.mean(h_det_avg["HOTA"])  * 100),
        "DetA":  float(np.mean(h_det_avg["DetA"])  * 100),
        "AssA":  float(np.mean(h_det_avg["AssA"])  * 100),
        "LocA":  float(np.mean(h_det_avg["LocA"])  * 100),
        "DetPr": float(np.mean(h_det_avg["DetPr"]) * 100),
        "DetRe": float(np.mean(h_det_avg["DetRe"]) * 100),
        "AssPr": float(np.mean(h_det_avg["AssPr"]) * 100),
        "AssRe": float(np.mean(h_det_avg["AssRe"]) * 100),
        "MOTA":  float(c_det_avg["MOTA"] * 100),
        "MOTP":  float(c_det_avg["MOTP"] * 100),
        "MT":    int(sum(per_class[c]["MT"]   for c in CLASSES)),
        "PT":    int(sum(per_class[c]["PT"]   for c in CLASSES)),
        "ML":    int(sum(per_class[c]["ML"]   for c in CLASSES)),
        "FP":    int(sum(per_class[c]["FP"]   for c in CLASSES)),
        "FN":    int(sum(per_class[c]["FN"]   for c in CLASSES)),
        "IDS":   int(sum(per_class[c]["IDS"]  for c in CLASSES)),
        "Frag":  int(sum(per_class[c]["Frag"] for c in CLASSES)),
        "IDF1":  float(i_det_avg["IDF1"] * 100),
        "IDP":   float(i_det_avg["IDP"]  * 100),
        "IDR":   float(i_det_avg["IDR"]  * 100),
    }

    return {
        "per_class":          per_class,
        "class_averaged":     cls_avg,
        "detection_averaged": det_avg,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Pretty-print
# ──────────────────────────────────────────────────────────────────────────────

def print_results(results: Dict) -> None:
    per_class = results["per_class"]
    cls_avg   = results["class_averaged"]
    det_avg   = results["detection_averaged"]

    NW = 14  # name column width
    FW = 9   # float column width
    IW = 10  # integer column width

    def make_table(
        title: str,
        cols: List[str],
        col_w: List[int],
        fmt_fn,
    ) -> None:
        """Print a single metric table."""
        total_w = sum(col_w)
        eq_sep  = "=" * total_w
        sep     = "-" * total_w

        def row(*vals):
            return "".join(str(v).ljust(w) for v, w in zip(vals, col_w))

        print()
        print(eq_sep)
        print(f"  {title}")
        print(eq_sep)
        print(row("Class", *cols))
        print(sep)
        for cls_id, cls_name in CLASSES.items():
            d = per_class[cls_id]
            print(row(cls_name, *[fmt_fn(d, k) for k in cols]))
        print(sep)
        print(row("Class-avg", *[fmt_fn(cls_avg, k) for k in cols]))
        print(row("Det-avg",   *[fmt_fn(det_avg,  k) for k in cols]))
        print(eq_sep)

    def fmt_f(d, k):
        return f"{d[k]:.2f}"

    def fmt_i(d, k):
        return f"{d[k]:,}"

    print()
    print("#" * 60)
    print("#  VisDrone MOT Evaluation Results")
    print("#" * 60)

    # ── Table 0: Detection & track counts ────────────────────────────────────
    count_cols = ["GT Dets", "Pred Dets", "GT Tracks", "Pred Tracks"]
    count_keys = ["gt_dets", "pred_dets", "gt_tracks", "pred_tracks"]
    CW = 13
    count_col_w = [NW] + [CW] * len(count_cols)
    count_total_w = sum(count_col_w)

    def count_row(*vals):
        return "".join(str(v).ljust(w) for v, w in zip(vals, count_col_w))

    print()
    print("=" * count_total_w)
    print("  [0/4] Dataset Summary  (raw counts)")
    print("=" * count_total_w)
    print(count_row("Class", *count_cols))
    print("-" * count_total_w)
    total_gt_dets = total_pred_dets = total_gt_tr = total_pred_tr = 0
    for cls_id, cls_name in CLASSES.items():
        d = per_class[cls_id]
        print(count_row(
            cls_name,
            f"{d['gt_dets']:,}",
            f"{d['pred_dets']:,}",
            f"{d['gt_tracks']:,}",
            f"{d['pred_tracks']:,}",
        ))
        total_gt_dets   += d["gt_dets"]
        total_pred_dets += d["pred_dets"]
        total_gt_tr     += d["gt_tracks"]
        total_pred_tr   += d["pred_tracks"]
    print("-" * count_total_w)
    print(count_row("TOTAL",
        f"{total_gt_dets:,}",
        f"{total_pred_dets:,}",
        f"{total_gt_tr:,}",
        f"{total_pred_tr:,}",
    ))
    print("=" * count_total_w)

    # ── Table 1: HOTA ─────────────────────────────────────────────────────────
    make_table(
        "[1/4] HOTA  (alpha-averaged, all values in %, higher = better)",
        ["HOTA", "DetA", "AssA", "LocA", "DetPr", "DetRe", "AssPr", "AssRe"],
        [NW] + [FW] * 8,
        fmt_f,
    )

    # ── Table 2: CLEAR percentage metrics ─────────────────────────────────────
    make_table(
        "[2/4] CLEAR  (%, higher = better)",
        ["MOTA", "MOTP"],
        [NW] + [FW] * 2,
        fmt_f,
    )

    # ── Table 3: CLEAR count metrics ──────────────────────────────────────────
    make_table(
        "[3/4] CLEAR counts  (MT/PT: more = better | ML/FP/FN/IDS/Frag: less = better)",
        ["MT", "PT", "ML", "FP", "FN", "IDS", "Frag"],
        [NW] + [IW] * 7,
        fmt_i,
    )

    # ── Table 4: Identity metrics ──────────────────────────────────────────────
    make_table(
        "[4/4] Identity  (%, higher = better)",
        ["IDF1", "IDP", "IDR"],
        [NW] + [FW] * 3,
        fmt_f,
    )

    total_gt = sum(per_class[c]["gt_dets"] for c in CLASSES)




# ──────────────────────────────────────────────────────────────────────────────
# CLI entry-point
# ──────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gt-dir",   default=DEFAULT_GT_DIR,
                   help="GT annotations directory")
    p.add_argument("--pred-dir", default=DEFAULT_PRED_DIR,
                   help="Prediction annotations directory")
    p.add_argument("--iou",      default=DEFAULT_IOU, type=float,
                   help="IoU threshold for MOTA/IDF1 matching (default: 0.5)")
    p.add_argument("--out",      default=DEFAULT_OUT,
                   help="Output JSON path (default: evaluate_results.json)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    gt_dir   = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)

    print(f"GT   : {gt_dir.resolve()}")
    print(f"Pred : {pred_dir.resolve()}")
    print(f"IoU  : {args.iou}  (MOTA/IDF1 matching threshold)")
    print(f"HOTA : alpha in [0.05, 0.95] (19 thresholds, then averaged)")
    print()

    if not gt_dir.is_dir():
        print(f"[ERROR] GT directory not found: {gt_dir}")
        sys.exit(1)
    if not pred_dir.is_dir():
        print(f"[ERROR] Prediction directory not found: {pred_dir}")
        sys.exit(1)

    results = evaluate(gt_dir, pred_dir, args.iou)
    print_results(results)

    # Serialise (strip internal _*_comb keys)
    out_data = {
        "config": {
            "gt_dir":     str(gt_dir),
            "pred_dir":   str(pred_dir),
            "iou_thresh": args.iou,
            "ignored_classes": sorted(IGNORED_CLASSES),
            "ignore_ioa_thresh": IGNORE_IOA_THRESHOLD,
        },
        "per_class": {
            CLASSES[c]: {k: v for k, v in d.items() if not k.startswith("_")}
            for c, d in results["per_class"].items()
        },
        "class_averaged":     results["class_averaged"],
        "detection_averaged": results["detection_averaged"],
    }

    out_path = Path(args.out)
    out_path.write_text(json.dumps(out_data, indent=2))
    print(f"Results saved to: {out_path.resolve()}")


if __name__ == "__main__":
    main()