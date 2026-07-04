# 🛸 UAV-Tracking: End-to-End UAV Object Detection & Multi-Object Tracking

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8 / YOLOv11](https://img.shields.io/badge/YOLO-Ultralytics-00ffff.svg)](https://github.com/ultralytics/ultralytics)
[![Tracking BoT--SORT | ByteTrack](https://img.shields.io/badge/Tracking-BoT--SORT%20%7C%20ByteTrack-orange.svg)](#-multi-object-tracking-mot)
[![TensorRT FP16 / INT8](https://img.shields.io/badge/TensorRT-FP16%20%2F%20INT8-76b900.svg)](#-export--hardware-optimization)

**UAV-Tracking** is a comprehensive, high-performance deep learning framework designed for **Object Detection and Multi-Object Tracking (MOT)** in Unmanned Aerial Vehicle (UAV) and drone video streams. Built on top of state-of-the-art **YOLO** (You Only Look Once) detection architectures and advanced multi-object trackers (**BoT-SORT** and **ByteTrack**), this project provides an end-to-end pipeline tailored for aerial imagery challenges such as small object scales, dense packing, high camera motion, and real-time inference requirements.

---

## 📌 About the Project

Aerial video analysis presents unique computer vision challenges compared to standard ground-level imagery: objects are often tiny, orientations vary rapidly, and camera motion compensation is critical. This project addresses these challenges through a unified workflow:

1. **Dataset Harmonization & Preprocessing**:
   - Seamlessly converts standard aerial benchmarks—specifically **VisDrone2019 (DET & MOT)** and **DroneVehicle**—into standard YOLO directory structures and annotation formats.
   - Unifies diverse class taxonomies into a standardized **10-class UAV taxonomy** (`pedestrian`, `people`, `bicycle`, `car`, `van`, `truck`, `tricycle`, `awning-tricycle`, `bus`, `motor`).
   - Handles advanced preprocessing including stride sampling for video sequences, 100px white border cropping, polygon-to-horizontal-bounding-box (HBB) conversion, and class-priority sampling.

2. **Custom Detection Training & Rigorous Evaluation**:
   - Fine-tunes custom YOLO models (e.g., YOLOv8, YOLOv11) specifically on UAV distributions with specialized data augmentations (mosaic, HSV scaling, rotation, and scale gain).
   - Evaluates detection precision using industry-standard metrics (**mAP@50**, **mAP@50-95**, **Precision**, and **Recall**) both across average and individual vehicle/pedestrian classes.

3. **High-Precision Multi-Object Tracking (MOT)**:
   - Integrates **BoT-SORT** (with Sparse Optical Flow Camera Motion Compensation and optional ReID) and **ByteTrack** for robust trajectory association across video frames.
   - Implements two-stage detection association (`track_high_thresh` and `track_low_thresh`) to recover occluded or small targets without introducing false positive trajectories.

4. **Hardware Acceleration & Speed Benchmarking**:
   - Exports trained PyTorch weights (`.pt`) to NVIDIA **TensorRT** engines (`.engine`) in **FP16**, **FP32**, or **INT8** (quantized with calibration data) for ultra-low latency real-time edge deployment.
   - Provides end-to-end speed benchmarking tools that measure realistic throughput (FPS and milliseconds per frame) while isolating disk I/O via memory preloading.

5. **MOT Challenge Evaluation & Qualitative Visualization**:
   - Evaluates tracking results against ground-truth annotations using the **TrackEval** framework, computing authoritative metrics including **HOTA**, **CLEAR MOT** (`MOTA`, `MOTP`, `FP`, `FN`, `IDS`), and **Identity** metrics (`IDF1`, `IDP`, `IDR`).
   - Features rich visualization utilities that render high-resolution bounding boxes, track IDs, class-colored highlights, and custom legends onto video sequences (`.mp4`) or individual PNG frame folders.

---

## 📦 Requirements & Installation

### 1. Prerequisites
- **Operating System**: Linux (Recommended for TensorRT and TrackEval) or Windows / macOS.
- **Python**: Version **3.8** or newer (Python 3.9+ recommended).
- **GPU Acceleration**: NVIDIA GPU with CUDA drivers and cuDNN installed (highly recommended for training, tracking, and TensorRT inference).

### 2. Environment Setup
It is recommended to create an isolated Python virtual environment using `conda` or `venv`:

```bash
# Using conda (recommended)
conda create -n uav-tracking python=3.10 -y
conda activate uav-tracking

# OR using Python venv
python3 -m venv venv
source venv/bin/activate
```

### 3. Installing Core Dependencies
Install PyTorch (with CUDA support matching your driver) and the required dependencies using the provided `requirements.txt`:

```bash
# 1. Install PyTorch (Example for CUDA 11.8 / 12.1 - visit https://pytorch.org for exact commands)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 2. Install project requirements
pip install -r requirements.txt
```

#### Core Packages Included in `requirements.txt`:
* `ultralytics>=8.0.0` (YOLO models and built-in tracking engine)
* `opencv-python>=4.7.0` (Video processing and image rendering)
* `numpy>=1.22.0` (Array manipulations and bounding box math)
* `Pillow>=9.0.0` (Image I/O and conversion utilities)
* `pyyaml>=6.0` (YAML configuration parsing)

### 4. Optional / Specialized Dependencies

#### A. TrackEval (Required for `eval_visdrone.py`)
To compute official MOT Challenge metrics (`HOTA`, `MOTA`, `IDF1`), install the official **TrackEval** package directly from GitHub:
```bash
pip install git+https://github.com/JonathonLuiten/TrackEval.git
```

#### B. NVIDIA TensorRT (Required for `export_tensorrt.py` & `track_visdrone_tensorrt_fp16.py`)
To enable hardware-accelerated FP16/INT8 inference on NVIDIA GPUs, install the TensorRT Python SDK matching your CUDA version:
```bash
# Example for TensorRT via PyPI (ensure your environment matches NVIDIA specifications)
pip install tensorrt>=8.5.0
```

---

## 📂 Project Structure & Script Overview

The repository is structured into modular scripts designed to be executed sequentially or independently. Below is a quick-reference summary of all available scripts and configuration files:

| Category | File / Script Name | Description |
| :--- | :--- | :--- |
| **Data Processing** | [`convert_to_yolo.py`](#1-convert_to_yolopy) | Unifies and converts VisDrone (DET & MOT) and DroneVehicle datasets into standard YOLO format. |
| **Training & Eval** | [`train_detect.py`](#2-train_detectpy) | Trains custom YOLO object detection models optimized for UAV aerial datasets. |
| | [`eval_detect.py`](#3-eval_detectpy) | Evaluates detection accuracy (`mAP@50`, `mAP@50-95`, `P`, `R`) on validation or test splits. |
| **Tracking & Inference**| [`infer.py`](#4-inferpy) | Runs YOLO + BoT-SORT/ByteTrack on a single image sequence; outputs video, frames, or MOT logs. |
| | [`track_visdrone.py`](#5-track_visdronepy) | Batch executes PyTorch tracking across the entire VisDrone-MOT test-dev set and generates `.txt` logs. |
| | [`track_visdrone_tensorrt_fp16.py`](#6-track_visdrone_tensorrt_fp16py) | Hardware-accelerated batch tracking on VisDrone test-dev using TensorRT FP16 engines. |
| **Hardware Export** | [`export_tensorrt.py`](#7-export_tensorrtpy) | Exports PyTorch `.pt` models to optimized TensorRT `.engine` formats (FP16, FP32, INT8, dynamic). |
| **MOT Evaluation** | [`eval_visdrone.py`](#8-eval_visdronepy) | Computes category-aware MOT metrics (`HOTA`, `CLEAR`, `Identity`) against ground truth via TrackEval. |
| **Benchmarking** | [`benchmark_speed.py`](#9-benchmark_speedpy) | Profiles end-to-end tracking runtime and FPS latency with optional RAM preloading. |
| **Visualization** | [`visualize.py`](#10-visualizepy) | Core rendering engine that draws class-colored bounding boxes, IDs, and legends onto frames/video. |
| | [`visualize_botsort.py`](#11-visualize_botsortpy) | Batch visualization wrapper to generate qualitative videos across multiple tracked MOT sequences. |
| **Configs** | `visdrone_botsort.yaml` | Tracker configuration for BoT-SORT (two-stage thresholds, camera motion compensation, buffer). |
| | `visdrone_bytetrack.yaml` | Tracker configuration for ByteTrack (association thresholds and track buffer settings). |

---

## 🛠️ Detailed Script Guide & Usage Examples

### 1. `convert_to_yolo.py`
**What it does:**  
A unified dataset conversion utility that transforms complex aerial datasets into the standard YOLO directory structure (`images/train`, `labels/train`, etc.) and generates a ready-to-use dataset YAML configuration file.  
* **VisDrone-DET**: Extracts images and converts 12-class annotations into the project's standard 10-class taxonomy.
* **VisDrone-MOT**: Extracts frames from video sequences, samples them at a configurable frame stride (`--stride`), and converts tracking labels to static detection labels.
* **DroneVehicle**: Processes raw RGB images, automatically clips away 100px white image borders, converts oriented polygons to horizontal bounding boxes (HBB), merges car/truck variants, and applies class-priority sampling (`--dv-sample-ratio`) to balance rare classes.

**Key Arguments:**
* `--dataset`: Dataset type to convert (`det`, `mot`, `dronevehicle`, or `all`).
* `--src`: Root directory of the source dataset.
* `--out`: Target directory where the converted YOLO dataset will be saved.
* `--stride`: Frame sampling interval for MOT sequences (default: `1`).
* `--dv-sample-ratio`: Sampling ratio for DroneVehicle training images (default: `1.0`).

**Usage Examples:**
```bash
# 1. Convert VisDrone Detection dataset
python convert_to_yolo.py --dataset det --src datasets/VisDrone --out datasets/VisDrone-YOLO

# 2. Convert VisDrone MOT dataset with a frame stride of 5 (samples every 5th frame)
python convert_to_yolo.py --dataset mot --src datasets/VisDrone --out datasets/VisDrone-YOLO --stride 5

# 3. Convert DroneVehicle dataset with 50% class-priority sampling
python convert_to_yolo.py --dataset dronevehicle --src datasets/VisDrone-DroneVehicle --out datasets/VisDrone-YOLO --dv-sample-ratio 0.5

# 4. Convert and combine all available datasets into a single unified workspace
python convert_to_yolo.py --dataset all --src datasets/ --out datasets/VisDrone-YOLO --stride 5
```

---

### 2. `train_detect.py`
**What it does:**  
A streamlined training script designed to fine-tune Ultralytics YOLO models on converted UAV datasets. It configures optimal hyperparameters for aerial imagery, including image resolution scaling, mosaic augmentation, HSV color jittering, and cosine learning rate schedules.

**Key Arguments:**
* `--model`, `-m`: Base model architecture or pretrained checkpoint (default: `yolov8s-p2.yaml`).
* `--data`, `-d`: Path to the dataset YAML config (default: `datasets/VisDrone-YOLO/dataset_combined.yaml`).
* `--epochs`, `-e`: Number of training epochs (default: `100`).
* `--batch`, `-b`: Batch size; use `-1` for AutoBatch (default: `-1`).
* `--imgsz`: Input image resolution (default: `1280` — higher resolution helps detect tiny UAV targets).
* `--device`: Compute device to use (e.g., `'0'` for GPU 0, `'0,1'` for multi-GPU, or `'cpu'`).
* `--project`, `--name`: Output directory and experiment run name for saving weights and logs.

**Usage Examples:**
```bash
# 1. Start basic training on GPU 0 with default settings (1280px resolution, 100 epochs)
python train_detect.py --data datasets/VisDrone-YOLO/dataset_combined.yaml --device 0

# 2. Train a custom architecture for 200 epochs with a specific batch size and cosine LR
python train_detect.py --model yolov8m.pt --data datasets/VisDrone-YOLO/dataset_mot.yaml \
    --epochs 200 --batch 16 --imgsz 1280 --cos-lr --name uav_yolov8m_run1

# 3. Resume or run lightweight CPU training for debugging
python train_detect.py --model yolov8n.pt --epochs 10 --batch 4 --imgsz 640 --device cpu
```

---

### 3. `eval_detect.py`
**What it does:**  
Evaluates trained YOLO detection models against validation or test datasets. It calculates global average and per-class metrics including Mean Average Precision (`mAP@50`, `mAP@50-95`), Precision (`P`), and Recall (`R`). Results are displayed in a clean terminal table and saved to CSV for reporting.

**Key Arguments:**
* `--model`, `-m`: Path to the trained model checkpoint (e.g., `runs/detect/train/weights/best.pt`).
* `--data`, `-d`: Path to the dataset YAML config file.
* `--split`: Dataset split to evaluate on (`val`, `test`, or `train`; default: `test`).
* `--conf`: Confidence threshold for detections (default: `0.001` for unbiased mAP calculation).
* `--iou`: Non-Maximum Suppression (NMS) IoU threshold (default: `0.7`).
* `--imgsz`: Evaluation image resolution (default: `1280`).

**Usage Examples:**
```bash
# 1. Evaluate the best training checkpoint on the test split
python eval_detect.py --model runs/detect/train/weights/best.pt --data datasets/VisDrone-YOLO/dataset_mot.yaml --split test

# 2. Run validation evaluation with custom confidence and NMS thresholds
python eval_detect.py --model runs/detect/train/weights/best.pt --split val --conf 0.05 --iou 0.60 --imgsz 1280
```

---

### 4. `infer.py`
**What it does:**  
The core inference engine for running simultaneous object detection and multi-object tracking on a single image sequence or video stream. It dynamically couples custom YOLO weights with **BoT-SORT** or **ByteTrack** trackers. Outputs can be saved as an MP4 video, individual annotated image frames, or a standard MOT-Challenge text file (`<frame>,<id>,<left>,<top>,<width>,<height>,<conf>,<class>,-1,-1`).

**Key Arguments:**
* `--model`, `-m`: Path to custom weights `.pt` or `.engine` (default: `runs/detect/train/weights/best.pt`).
* `--source`, `-s`: Path to an image sequence directory or video file.
* `--tracker`, `-t`: Tracker configuration YAML (`visdrone_botsort.yaml` or `visdrone_bytetrack.yaml`).
* `--output-dir`: Directory to save visual and textual outputs.
* `--save-frames`: Save individual annotated PNG frames instead of compiling an MP4 video.
* `--save-mot`: Export tracking trajectories to an MOT-Challenge formatted `.txt` file.
* `--conf`: Detection confidence threshold (default: `0.10` — must match `track_low_thresh` in tracker YAML).
* `--iou`: NMS IoU threshold (default: `0.60`).

**Usage Examples:**
```bash
# 1. Run BoT-SORT tracking on an image sequence and compile an annotated MP4 video
python infer.py --model runs/detect/train/weights/best.pt \
    --source datasets/VisDrone/VisDrone2019-MOT-test-dev/sequences/uav0000086_00000_v \
    --tracker visdrone_botsort.yaml --output-dir runs/inference_video

# 2. Run ByteTrack, save individual annotated frames, and export MOT-Challenge track logs
python infer.py --model runs/detect/train/weights/best.pt \
    --source datasets/VisDrone/VisDrone2019-MOT-test-dev/sequences/uav0000086_00000_v \
    --tracker visdrone_bytetrack.yaml --save-frames --save-mot --output-dir runs/inference_frames
```

---

### 5. `track_visdrone.py`
**What it does:**  
A batch processing script that runs PyTorch YOLO detection coupled with BoT-SORT or ByteTrack across *all* sequences in the VisDrone2019-MOT test-dev dataset (or any custom directory of sequences). It automatically applies VisDrone-specific class mappings (converting 0-indexed YOLO predictions to VisDrone's 1-indexed benchmark taxonomy) and writes standard `.txt` tracking prediction files required for benchmark evaluation.

**Key Configuration / Arguments:**
* Configured via command-line flags or in-script defaults:
* `--model`, `-m`: Model weights path (default: `runs/detect/train/weights/best.pt`).
* `--sequences-dir`: Root directory containing VisDrone image sequences.
* `--tracker`: Tracker config (`visdrone_botsort.yaml` or `visdrone_bytetrack.yaml`).
* `--output-dir`: Destination folder for generated MOT `.txt` annotation files.
* `--imgsz`, `--conf`, `--iou`, `--device`: Standard inference hyperparameters.

**Usage Examples:**
```bash
# 1. Batch track all test-dev sequences using BoT-SORT and default best.pt weights
python track_visdrone.py

# 2. Batch track using ByteTrack on a custom GPU and specify output directory
python track_visdrone.py --model runs/detect/train/weights/best.pt \
    --tracker visdrone_bytetrack.yaml \
    --output-dir runs/tracking/visdrone_bytetrack_preds --device 0
```

---

### 6. `track_visdrone_tensorrt_fp16.py`
**What it does:**  
The hardware-accelerated counterpart to `track_visdrone.py`. It loads an exported NVIDIA **TensorRT FP16 engine** (`.engine`) along with its `.names` class mapping file to perform high-speed batch tracking across VisDrone test-dev sequences. Dramatically reduces inference latency while maintaining tracking accuracy.

**Key Arguments:**
* `--model`, `-m`: Path to the exported TensorRT engine (default: `runs/detect/train/weights/best.engine`).
* `--sequences-dir`: Root directory containing image sequences.
* `--tracker`: Tracker YAML config file (default: `visdrone_botsort.yaml`).
* `--output-dir`: Directory where MOT `.txt` files will be saved.
* `--imgsz`, `--conf`, `--iou`, `--device`: Inference settings (image size must match engine export resolution).

**Usage Examples:**
```bash
# 1. Run batch tracking using the default FP16 TensorRT engine and BoT-SORT
python track_visdrone_tensorrt_fp16.py

# 2. Run TensorRT tracking with ByteTrack and save predictions to a custom folder
python track_visdrone_tensorrt_fp16.py --model runs/detect/train/weights/best.engine \
    --tracker visdrone_bytetrack.yaml \
    --output-dir runs/tracking/test-dev-tensorrt-bytetrack --device 0
```

---

### 7. `export_tensorrt.py`
**What it does:**  
Exports PyTorch `.pt` model checkpoints into high-performance NVIDIA TensorRT `.engine` files. It automatically generates a accompanying `.names` JSON file preserving class labels. Supports multiple quantization precisions:
* **FP16** (Half-precision): Recommended default; offers ~2x speedup on NVIDIA GPUs with negligible accuracy loss.
* **FP32** (Single-precision): Standard floating-point export.
* **INT8** (8-bit Integer): Maximum acceleration; performs post-training quantization using a provided dataset calibration YAML.
* **Dynamic Shapes**: Configures dynamic batch sizes up to a specified maximum (`--batch`).

**Key Arguments:**
* `--weights`, `-w`: Path to input PyTorch weights (default: `runs/detect/train/weights/best.pt`).
* `--imgsz`: Target square resolution for the exported engine (default: `1280`).
* `--fp32`, `--int8`: Specify export precision (defaults to **FP16** if neither is set).
* `--data`: Dataset YAML required *only* for INT8 calibration data sampling.
* `--dynamic`: Enable dynamic input batching and shapes.
* `--batch`: Maximum batch size when using `--dynamic` (default: `1`).
* `--device`: GPU device ID to use for building the TensorRT engine.

**Usage Examples:**
```bash
# 1. Recommended: Export to FP16 static engine at 1280x1280 resolution
python export_tensorrt.py --weights runs/detect/train/weights/best.pt --imgsz 1280

# 2. Export to INT8 precision using training dataset for calibration
python export_tensorrt.py --weights runs/detect/train/weights/best.pt --int8 --data datasets/VisDrone-YOLO/dataset_mot.yaml

# 3. Export an FP16 engine with dynamic batch sizes up to batch=8
python export_tensorrt.py --weights runs/detect/train/weights/best.pt --dynamic --batch 8
```

---

### 8. `eval_visdrone.py`
**What it does:**  
Performs official category-aware Multi-Object Tracking evaluation by comparing predicted MOT `.txt` logs against ground-truth VisDrone annotations. Powered by the authoritative **TrackEval** package, it computes global and class-specific metrics across key vehicle and pedestrian categories (`pedestrian`, `car`, `van`, `truck`, `bus`).
* **HOTA Metrics**: `HOTA`, `DetA`, `AssA`, `LocA`
* **CLEAR MOT Metrics**: `MOTA`, `MOTP`, `FP`, `FN`, `IDS` (Identity Switches), `Frag`
* **Identity Metrics**: `IDF1`, `IDP`, `IDR`  
Outputs both class-averaged (equal weight) and detection-averaged (weighted by ground-truth frequency) results, saving a detailed summary JSON report.

**Key Arguments:**
* `--gt-dir`: Directory containing ground-truth MOT annotation files (default: `datasets/VisDrone/VisDrone2019-MOT-test-dev/annotations`).
* `--pred-dir`: Directory containing predicted MOT `.txt` files (e.g., generated by `track_visdrone.py`).
* `--iou`: IoU matching threshold between predictions and ground truth (default: `0.5`).
* `--out`: Path to save the output JSON evaluation summary (default: `evaluate_results.json`).

**Usage Examples:**
```bash
# 1. Evaluate predicted tracking logs against VisDrone test-dev ground truth
python eval_visdrone.py --gt-dir datasets/VisDrone/VisDrone2019-MOT-test-dev/annotations \
    --pred-dir runs/tracking/visdrone/test-dev/annotations \
    --out runs/tracking/eval_summary_botsort.json

# 2. Evaluate at a stricter IoU threshold of 0.75
python eval_visdrone.py --pred-dir runs/tracking/visdrone_bytetrack_preds --iou 0.75
```

---

### 9. `benchmark_speed.py`
**What it does:**  
Benchmarks execution speed and tracking latency (FPS and milliseconds per frame) for end-to-end YOLO + BoT-SORT/ByteTrack pipelines. It can profile a single sequence or loop through an entire dataset directory.  
* **Preloading (`--preload`)**: Optionally pre-reads all sequence images into RAM prior to starting the timer. This removes disk storage bottleneck from the measurement, reporting isolated compute and tracking performance.
* **Residual Overhead Breakdown**: Reports total wall-clock latency alongside Ultralytics profiler breakdowns (preprocess, inference, postprocess, and tracking association time).

**Key Arguments:**
* `--model`, `-m`: Model weights (`.pt` or TensorRT `.engine`).
* `--source`, `-s`: Target image sequence directory or parent directory containing multiple sequences.
* `--tracker`, `-t`: Tracker config YAML (`visdrone_botsort.yaml` or `visdrone_bytetrack.yaml`).
* `--preload`: Preload all frames into RAM before timing to eliminate disk read latency.
* `--imgsz`, `--conf`, `--iou`, `--device`: Standard inference arguments.

**Usage Examples:**
```bash
# 1. Benchmark speed on a single sequence with memory preloading (pure compute speed)
python benchmark_speed.py --model runs/detect/train/weights/best.engine \
    --source datasets/VisDrone/VisDrone2019-MOT-test-dev/sequences/uav0000086_00000_v \
    --tracker visdrone_botsort.yaml --preload --device 0

# 2. Benchmark PyTorch FP32 tracking speed across all sequences in test-dev
python benchmark_speed.py --model runs/detect/train/weights/best.pt \
    --source datasets/VisDrone/VisDrone2019-MOT-test-dev/sequences \
    --tracker visdrone_bytetrack.yaml --device 0
```

---

### 10. `visualize.py`
**What it does:**  
The standalone visualization rendering engine of the project. It overlays tracking or ground-truth bounding boxes onto sequence frames, drawing crisp class-colored borders, track ID labels, and an informative class legend. Can output smoothly encoded MP4 videos or individual image frames, with options to slice specific frame intervals or filter out low-confidence detections.

**Key Arguments:**
* `--seq`: Directory containing raw sequence image frames (JPEGs or PNGs).
* `--ann`: MOT-Challenge formatted `.txt` annotation file (predictions or ground truth).
* `--out-dir`: Directory to save the rendered output (defaults to sequence parent folder).
* `--name`: Output video filename (e.g., `sequence_tracked.mp4`).
* `--frames`: Save individual annotated PNG images instead of an MP4 video.
* `--start`, `--end`: Start and end frame numbers to visualize a specific clip subset.
* `--conf-min`: Minimum confidence score threshold to render a bounding box (default: `0.0`).
* `--tracked-only`: Highlight only the 5 primary tracked vehicle/pedestrian classes.

**Usage Examples:**
```bash
# 1. Generate an annotated video from predictions
python visualize.py \
    --seq datasets/VisDrone/VisDrone2019-MOT-test-dev/sequences/uav0000268_05773_v \
    --ann runs/tracking/visdrone/test-dev/annotations/uav0000268_05773_v.txt \
    --out-dir vis_output/ --name uav0000268_05773_v_tracked.mp4

# 2. Save individual annotated frames for frames 10 through 100 with confidence >= 0.25
python visualize.py \
    --seq datasets/VisDrone/VisDrone2019-MOT-train/sequences/uav0000013_00000_v \
    --ann datasets/VisDrone/VisDrone2019-MOT-train/annotations/uav0000013_00000_v.txt \
    --out-dir vis_output/frames/ --frames --start 10 --end 100 --conf-min 0.25
```

---

### 11. `visualize_botsort.py`
**What it does:**  
A convenient batch visualization wrapper built around `visualize.py`. It automates the rendering process by matching all sequence folders in a sequence directory (e.g., VisDrone test-dev) with corresponding BoT-SORT/ByteTrack prediction files in an annotation directory. Generates qualitative videos or frame folders for every sequence in a single run.

**Key Arguments:**
* `--seq-root`: Parent directory containing multiple image sequence folders (default: `datasets/VisDrone/VisDrone2019-MOT-test-dev/sequences`).
* `--ann-root`: Parent directory containing corresponding `.txt` annotation files (default: `botsort/visdrone/test-dev/annotations`).
* `--out-dir`: Destination root where rendered videos or frame folders will be stored (default: `vis_output/botsort`).
* `--frames`: Export individual frames instead of MP4 videos.
* `--conf-min`: Minimum confidence score to render (default: `0.0`).
* `--tracked-only`: Render only the 5 primary evaluated VisDrone categories.

**Usage Examples:**
```bash
# 1. Batch generate tracking videos for all test-dev sequences using default paths
python visualize_botsort.py

# 2. Batch visualize custom ByteTrack results, rendering only primary vehicle/pedestrian classes with conf >= 0.20
python visualize_botsort.py --ann-root runs/tracking/test-dev-tensorrt-bytetrack \
    --out-dir vis_output/bytetrack_videos --conf-min 0.20 --tracked-only
```

---

## ⚙️ Tracker Configuration Highlights

The project utilizes two primary tracker configuration files that control two-stage association and camera motion compensation:

* **`visdrone_botsort.yaml`**: Configures BoT-SORT with `track_high_thresh: 0.25` and `track_low_thresh: 0.10`. Enables `sparseOptFlow` for camera motion compensation (GMC), essential for stabilizing aerial video feeds.
* **`visdrone_bytetrack.yaml`**: Configures ByteTrack with identical two-stage association thresholds (`0.25` and `0.10`) without GMC or appearance embeddings, providing a lightweight, ultra-fast tracking baseline.

> [!IMPORTANT]  
> When invoking tracking via script or custom Python code, ensure the detection threshold aligns with the lower tracking bound so second-stage association can recover occluded objects:  
> `model.track(..., conf=0.10, iou=0.60, tracker="visdrone_botsort.yaml")`

---
*Developed for advanced agentic coding and aerial computer vision research.*
