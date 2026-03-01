# Technical Documentation — Tennis Play/Dead Time Detection Pipeline

> **Repository:** `rlxai/tennis-analytics-phase-1`
> **Branch:** `master`
> **Last updated:** March 2026

---

## Table of Contents

- [Technical Documentation — Tennis Play/Dead Time Detection Pipeline](#technical-documentation--tennis-playdead-time-detection-pipeline)
  - [Table of Contents](#table-of-contents)
  - [1. Project Overview](#1-project-overview)
  - [2. Architecture](#2-architecture)
  - [3. Directory Structure](#3-directory-structure)
  - [4. Installation \& Prerequisites](#4-installation--prerequisites)
    - [System Requirements](#system-requirements)
    - [Installation Steps](#installation-steps)
    - [Model Weights](#model-weights)
    - [Python Dependencies](#python-dependencies)
  - [5. Configuration Reference](#5-configuration-reference)
    - [Top-level Settings](#top-level-settings)
    - [`tracknet` — Ball Detection](#tracknet--ball-detection)
    - [`yolo` — Player Detection](#yolo--player-detection)
    - [`classifier` — State Classification](#classifier--state-classification)
    - [`court` — Court Region Detection](#court--court-region-detection)
  - [6. Pipeline Phases](#6-pipeline-phases)
    - [6.1 Phase 1 — Video Probing](#61-phase-1--video-probing)
    - [6.2 Phase 1b — Court Region Detection](#62-phase-1b--court-region-detection)
    - [6.3 Phase 2 — Chunked Detection (Ball \& Player)](#63-phase-2--chunked-detection-ball--player)
    - [6.4 Phase 3 — State Classification](#64-phase-3--state-classification)
    - [6.5 Phase 4 — Output Generation](#65-phase-4--output-generation)
  - [7. Module Reference](#7-module-reference)
    - [7.1 `config.py`](#71-configpy)
    - [7.2 `pipeline.py`](#72-pipelinepy)
    - [7.3 `ball_tracker.py`](#73-ball_trackerpy)
    - [7.4 `player_detector.py`](#74-player_detectorpy)
    - [7.5 `court_detector.py`](#75-court_detectorpy)
    - [7.6 `state_classifier.py`](#76-state_classifierpy)
    - [7.7 `renderer.py`](#77-rendererpy)
    - [7.8 `io_utils.py`](#78-io_utilspy)
  - [8. Classification Algorithm Deep Dive](#8-classification-algorithm-deep-dive)
    - [Step 1 — Per-frame Raw Signals](#step-1--per-frame-raw-signals)
    - [Step 2 — Windowed Feature Smoothing](#step-2--windowed-feature-smoothing)
    - [Step 3 — Normalised Weighted Play Score](#step-3--normalised-weighted-play-score)
    - [Step 4 — Gaussian Temporal Smoothing](#step-4--gaussian-temporal-smoothing)
    - [Step 5 — Hysteresis Thresholding](#step-5--hysteresis-thresholding)
    - [Step 6 — Ball-in-Bbox Gate](#step-6--ball-in-bbox-gate)
    - [Step 7 — Minimum Duration Filter](#step-7--minimum-duration-filter)
    - [Step 8 — Isolated Play Filter](#step-8--isolated-play-filter)
  - [9. Court Detection Algorithm](#9-court-detection-algorithm)
    - [Algorithm Steps](#algorithm-steps)
    - [Filtering Behaviour](#filtering-behaviour)
  - [10. Memory Management \& Chunking](#10-memory-management--chunking)
    - [Auto-Chunk Sizing](#auto-chunk-sizing)
    - [Per-Chunk Cleanup](#per-chunk-cleanup)
    - [TrackNet Overlap](#tracknet-overlap)
  - [11. System Benchmarking \& Hardware Requirements](#11-system-benchmarking--hardware-requirements)
    - [Concurrent Model Resource Profile](#concurrent-model-resource-profile)
    - [GPU VRAM Breakdown (Peak, During Phase 2)](#gpu-vram-breakdown-peak-during-phase-2)
    - [System RAM Breakdown (Peak, 1080p)](#system-ram-breakdown-peak-1080p)
    - [CPU Usage Profile](#cpu-usage-profile)
    - [Known Memory Safety Caveats](#known-memory-safety-caveats)
    - [Recommended System Configurations](#recommended-system-configurations)
    - [Config Knobs to Tune if Hitting OOM](#config-knobs-to-tune-if-hitting-oom)
  - [12. CLI Reference](#12-cli-reference)
  - [13. Batch Processing](#13-batch-processing)
  - [14. Output Formats](#14-output-formats)
    - [Annotated Video (`*_analysis.mp4`)](#annotated-video-_analysismp4)
    - [JSON Report (`*_report.json`)](#json-report-_reportjson)
  - [15. Performance Considerations](#15-performance-considerations)
    - [GPU Utilisation](#gpu-utilisation)
    - [Throughput Estimates](#throughput-estimates)
    - [Bottlenecks](#bottlenecks)
    - [Tips for Large Videos](#tips-for-large-videos)
  - [16. Extending the Pipeline](#16-extending-the-pipeline)
    - [Adding a New Detection Module](#adding-a-new-detection-module)
    - [Modifying the Classification Logic](#modifying-the-classification-logic)
    - [Custom Video Overlays](#custom-video-overlays)
    - [Supporting Non-Linux Systems](#supporting-non-linux-systems)

---

## 1. Project Overview

This project provides an end-to-end computer vision pipeline that automatically classifies every frame of a tennis broadcast video as either **play** (active rally) or **dead time** (between points, changeovers, replays, etc.).

The system combines three neural-network-based perception modules with a multi-signal temporal classifier:

| Component | Model | Purpose |
|---|---|---|
| **Ball Tracker** | TrackNet (CNN) | Detects the tennis ball position per frame |
| **Player Detector** | YOLOv8/YOLO26s | Detects and localises players on court |
| **Court Detector** | Classical CV (Hough + Canny) | Identifies the court region for ROI filtering |
| **State Classifier** | Signal processing (no ML) | Fuses signals into a play/dead classification |

**Key capabilities:**

- GPU-accelerated batched inference (CUDA, with FP16 mixed precision)
- Automatic memory-aware chunking for processing long videos on constrained hardware
- Configurable hysteresis-based state machine with temporal smoothing
- Ball-in-bounding-box gate to suppress false play transitions
- Isolated play segment filtering for noise reduction
- H.264 annotated video output via FFmpeg pipe (single-pass encoding)
- Machine-readable JSON reports with frame-level and segment-level data
- Batch processing of multiple videos with logging

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         Pipeline Entry                            │
│                        (pipeline.py)                              │
│                                                                   │
│  ┌─────────┐   ┌──────────────┐   ┌───────────────┐              │
│  │ Phase 1  │   │  Phase 1b    │   │   Phase 2     │              │
│  │ Probe    │──▶│ Court Detect │──▶│ Chunked Loop  │              │
│  │ Video    │   │ (1st frame)  │   │               │              │
│  └─────────┘   └──────────────┘   │ ┌───────────┐ │              │
│                                    │ │ BallTracker│ │              │
│                                    │ │ (TrackNet) │ │              │
│                                    │ └───────────┘ │              │
│                                    │ ┌───────────┐ │              │
│                                    │ │PlayerDetect│ │              │
│                                    │ │  (YOLO)    │ │              │
│                                    │ └───────────┘ │              │
│                                    │ ┌───────────┐ │              │
│                                    │ │Court Filter│ │              │
│                                    │ │  (ROI mask)│ │              │
│                                    │ └───────────┘ │              │
│                                    └───────┬───────┘              │
│                                            │                      │
│                                            ▼                      │
│                                    ┌───────────────┐              │
│                                    │   Phase 3     │              │
│                                    │ Classifier    │              │
│                                    │ (hysteresis + │              │
│                                    │  smoothing)   │              │
│                                    └───────┬───────┘              │
│                                            │                      │
│                                            ▼                      │
│                                    ┌───────────────┐              │
│                                    │   Phase 4     │              │
│                                    │ Output Gen    │              │
│                                    │ (video+JSON)  │              │
│                                    └───────────────┘              │
└──────────────────────────────────────────────────────────────────┘
```

**Data flow (per chunk):**

```
Raw video frames
      │
      ├──▶ BallTracker.detect()    ──▶ ball_positions[]
      │                                      │
      ├──▶ PlayerDetector.detect() ──▶ player_data[]
      │                                      │
      │         ┌────────────────────────────┘
      │         ▼
      │    CourtDetector.filter_*()  ──▶ filtered positions & boxes
      │                                      │
      │                                      ▼
      │                            StateClassifier.classify()
      │                                      │
      │                                      ▼
      │                            states[], play_scores[],
      │                            segments[], summary{}
      │                                      │
      │         ┌────────────────────────────┘
      │         ├──▶ render_output_video()  ──▶ annotated .mp4
      │         └──▶ write_json_report()    ──▶ .json report
```

> **Note:** Ball and player detection run sequentially (not in parallel).
> Court filtering takes detection outputs as input, not raw frames.

---

## 3. Directory Structure

```
tennis-analytics-phase-1/
├── config/
│   └── config.yaml            # User-editable configuration overrides
├── data/
│   ├── videos/                # Input videos (user-provided)
│   ├── output/                # Pipeline outputs (videos + JSON reports)
│   └── logs/                  # Per-video batch processing logs
├── src/
│   ├── __init__.py            # Package marker
│   ├── __main__.py            # Enables `python -m src` invocation
│   ├── ball_tracker.py        # TrackNet ball detection module
│   ├── config.py              # Dataclass configs + YAML loader
│   ├── court_detector.py      # Sideline-based court ROI detection
│   ├── io_utils.py            # Video I/O, JSON reports, system helpers
│   ├── pipeline.py            # Main pipeline orchestrator + CLI
│   ├── player_detector.py     # YOLO player detection module
│   ├── renderer.py            # Annotated video rendering
│   └── state_classifier.py    # Play/dead state classification engine
├── tracknet/                  # Git submodule → TrackNet model code + weights
├── requirements.txt           # Python dependencies
├── run_all.sh                 # Batch processing script
└── README.md                  # Quick-start guide
```

---

## 4. Installation & Prerequisites

### System Requirements

| Requirement | Details |
|---|---|
| **OS** | Linux (tested on Ubuntu; uses `/proc/meminfo` for RAM detection) |
| **Python** | 3.10+ (uses `type[T]` syntax and `list[T]` generics) |
| **GPU** | NVIDIA GPU with CUDA support (strongly recommended) |
| **FFmpeg** | Required for H.264 output encoding; falls back to `mp4v` if unavailable |
| **RAM** | Proportional to video resolution; auto-chunking allocates ~40% of available |

### Installation Steps

```bash
# Clone with submodules (TrackNet model code)
git clone --recurse-submodules https://github.com/rlxai/tennis-analytics-phase-1.git
cd tennis-analytics-phase-1

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r tracknet/requirements.txt

# System dependency
sudo apt install ffmpeg
```

### Model Weights

| Model | Source | Destination |
|---|---|---|
| TrackNet | Bundled via submodule | `tracknet/weights/model_best.pt` |
| YOLO26s | [Google Drive](https://drive.google.com/drive/folders/1V4KW6x1rKWAM_7HMYizFqRX92vCxSyt-) | `weights/yolo26s.pt` (project root) |

### Python Dependencies

| Package | Min Version | Purpose |
|---|---|---|
| `torch` | 2.0 | Neural network inference engine |
| `torchvision` | — | Vision utilities |
| `ultralytics` | 8.0 | YOLO model loading and inference |
| `opencv-python-headless` | 4.8 | Video I/O, image processing |
| `numpy` | 1.24 | Numerical operations |
| `scipy` | 1.10 | Gaussian filtering for temporal smoothing |
| `PyYAML` | 6.0 | Configuration file parsing |

---

## 5. Configuration Reference

Configuration is managed through a hierarchy of dataclasses in `src/config.py`. Default values are built into the code; any key set in `config/config.yaml` overrides the corresponding default.

### Top-level Settings

| Key | Type | Default | Description |
|---|---|---|---|
| `device` | `str` | `"cuda"` | Compute device (`"cuda"` or `"cpu"`). **Note:** Currently unused at runtime — the pipeline auto-selects based on `torch.cuda.is_available()`. |
| `use_fp16` | `bool` | `True` | Enable FP16 mixed-precision inference. **Note:** Currently unused — FP16 is unconditionally enabled in both TrackNet (`torch.amp.autocast`) and YOLO (`half=True`). Reserved for future use. |
| `chunk_frames` | `int` | `0` | Frames per processing chunk (`0` = auto-calculate from RAM) |
| `render_video` | `bool` | `True` | Generate annotated output video (`false` = JSON only) |

### `tracknet` — Ball Detection

| Key | Type | Default | Description |
|---|---|---|---|
| `model_path` | `str` | `"tracknet/weights/model_best.pt"` | Path to TrackNet weights |
| `input_width` | `int` | `640` | Model input width (pixels) |
| `input_height` | `int` | `360` | Model input height (pixels) |
| `batch_size` | `int` | `16` | Frames per GPU batch |
| `heatmap_threshold` | `int` | `127` | Binary threshold on heatmap |
| `hough_dp` | `int` | `1` | HoughCircles inverse resolution ratio |
| `hough_min_dist` | `int` | `1` | Minimum distance between circle centres |
| `hough_param1` | `int` | `50` | Canny upper threshold for HoughCircles |
| `hough_param2` | `int` | `2` | Accumulator threshold for circle detection |
| `hough_min_radius` | `int` | `2` | Minimum detected circle radius |
| `hough_max_radius` | `int` | `7` | Maximum detected circle radius |
| `outlier_max_dist` | `float` | `100.0` | Max pixel jump between frames before outlier removal |

### `yolo` — Player Detection

| Key | Type | Default | Description |
|---|---|---|---|
| `model_path` | `str` | `"weights/yolo26s.pt"` | Path to YOLO weights |
| `batch_size` | `int` | `32` | Frames per GPU batch |
| `imgsz` | `int` | `640` | YOLO input image size |
| `conf_threshold` | `float` | `0.45` | Detection confidence cutoff |
| `person_class_id` | `int` | `0` | COCO class index for person |
| `max_det` | `int` | `10` | Maximum detections per frame |
| `bbox_expand_h` | `float` | `0.60` | Horizontal bbox expansion ratio for ball gate |
| `bbox_expand_v` | `float` | `0.30` | Vertical bbox expansion ratio for ball gate |

### `classifier` — State Classification

| Key | Type | Default | Description |
|---|---|---|---|
| `window_size` | `int` | `25` | Rolling window size (frames) for feature smoothing |
| `weight_detection_rate` | `float` | `0.45` | Play score weight for ball detection rate |
| `weight_ball_speed` | `float` | `0.35` | Play score weight for ball speed |
| `weight_player_count` | `float` | `0.20` | Play score weight for player count |
| `detection_rate_cap` | `float` | `0.40` | Normalisation cap for detection rate |
| `ball_speed_cap` | `float` | `8.0` | Normalisation cap (px/frame at 1080p reference) |
| `player_count_cap` | `float` | `2.0` | Normalisation cap for player count |
| `ball_speed_ref_width` | `int` | `1920` | Reference width for speed normalisation |
| `ball_speed_ref_height` | `int` | `1080` | Reference height for speed normalisation |
| `smooth_sigma` | `float` | `12.5` | Gaussian sigma (frames) for temporal smoothing |
| `enter_play_threshold` | `float` | `0.55` | Hysteresis upper threshold (dead → play) |
| `exit_play_threshold` | `float` | `0.35` | Hysteresis lower threshold (play → dead) |
| `use_bbox_gate` | `bool` | `True` | Enable ball-in-bbox gate |
| `bbox_gate_lookback` | `int` | `13` | Lookback window for bbox gate (frames) |
| `bbox_gate_ratio` | `float` | `0.5` | Suppression ratio for bbox gate |
| `min_play_duration` | `float` | `1.5` | Minimum play segment duration (seconds) |
| `min_dead_duration` | `float` | `2.0` | Minimum dead segment duration (seconds) |
| `use_isolated_play_filter` | `bool` | `True` | Enable isolated play segment removal |
| `isolated_max_play` | `float` | `3.0` | Max play duration targeted by filter (seconds) |
| `isolated_dead_ratio` | `float` | `2.5` | Play-to-surrounding-dead ratio for removal |

### `court` — Court Region Detection

| Key | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool` | `True` | Enable court detection |
| `padding_px` | `int` | `350` | Perpendicular padding from sidelines (pixels) |
| `steep_min_angle` | `float` | `20.0` | Minimum angle from horizontal for sideline candidates |
| `hough_threshold` | `int` | `50` | HoughLinesP accumulator threshold |
| `min_line_length` | `int` | `150` | Minimum line segment length (pixels) |
| `max_line_gap` | `int` | `15` | Maximum gap for segment merging |
| `canny_low` | `int` | `30` | Canny lower hysteresis threshold |
| `canny_high` | `int` | `100` | Canny upper hysteresis threshold |
| `min_court_area_pct` | `float` | `0.05` | Minimum valid ROI as fraction of frame area |
| `max_court_area_pct` | `float` | `0.95` | Maximum valid ROI as fraction of frame area |
| `debug_overlay` | `bool` | `False` | Draw padded court boundary on output video |
| `baseline_cap_enabled` | `bool` | `True` | Enable far-court baseline cap |
| `baseline_search_frac` | `float` | `0.55` | Search top fraction of frame for baseline |
| `baseline_max_angle` | `float` | `15.0` | Max degrees from horizontal for baseline |
| `baseline_min_length` | `int` | `80` | Minimum baseline segment length |
| `baseline_hough_threshold` | `int` | `40` | Hough threshold for baseline detection |
| `baseline_pad_px` | `int` | `50` | Shift baseline upward (pixels) |

---

## 6. Pipeline Phases

### 6.1 Phase 1 — Video Probing

**Module:** `io_utils.probe_video()`

Reads video metadata (FPS, resolution, frame count) via OpenCV without decoding frames. Calculates estimated memory footprint and duration. If `chunk_frames` is set to `0`, the auto-chunking algorithm determines the optimal chunk size based on available system RAM (targeting ~40% of `MemAvailable`).

### 6.2 Phase 1b — Court Region Detection

**Module:** `CourtDetector.detect()`

Runs once on the first decoded frame. Uses classical computer vision (Canny edge detection → Hough line detection) to identify the two court sidelines. Constructs a padded binary mask defining the valid court region. All subsequent ball and player detections are filtered against this mask to remove off-court noise (e.g., spectators, ball boys, broadcast graphics).

If detection fails, the pipeline falls back to full-frame processing with no ROI filtering.

### 6.3 Phase 2 — Chunked Detection (Ball & Player)

**Modules:** `BallTracker.detect()`, `PlayerDetector.detect()`

The video is processed in memory-sized chunks. For each chunk:

1. **Decode** — Sequential frame reading from the video file (no seeking between chunks for efficiency).
2. **Ball detection** — TrackNet inference using 3-frame sliding windows with batched GPU forward passes. A 2-frame overlap is maintained between chunks so the sliding window is contiguous.
3. **Player detection** — YOLO batched inference on the same frames, producing bounding boxes, centres, and confidence scores.
4. **Court filtering** — Off-court detections are suppressed using the mask from Phase 1b.
5. **Cleanup** — Chunk memory is released, GPU cache is emptied, and `malloc_trim()` is called to return memory to the OS.

### 6.4 Phase 3 — State Classification

**Module:** `StateClassifier.classify()`

Converts raw per-frame detections into a binary play/dead classification using a multi-stage signal processing pipeline:

1. **Feature extraction** — Ball detection rate, ball speed, and player count per frame.
2. **Windowed smoothing** — Uniform (box) filter over a rolling window.
3. **Weighted scoring** — Normalised and capped signals are combined into a composite play score in [0, 1].
4. **Gaussian smoothing** — Temporal Gaussian filter reduces frame-to-frame noise.
5. **Hysteresis thresholding** — Dual-threshold state machine prevents rapid oscillation:
   - Dead → Play requires score ≥ `enter_play_threshold` (0.55)
   - Play → Dead requires score < `exit_play_threshold` (0.35)
6. **Ball-in-bbox gate** — Suppresses false play transitions caused by the ball being carried/held by a player (e.g., between points).
7. **Minimum duration filter** — Segments shorter than the configured minimum are merged into the surrounding state.
8. **Isolated play filter** — Short play segments surrounded by disproportionately long dead time are reclassified as dead.

### 6.5 Phase 4 — Output Generation

**Modules:** `renderer.render_output_video()`, `io_utils.write_json_report()`

Two outputs are produced:

- **Annotated video** (optional, controlled by `render_video`) — Re-reads the source video and overlays a colour-coded header bar, ball markers, player bounding boxes, play score, timestamp, and a full-video timeline bar. Encoding uses FFmpeg via stdin pipe for single-pass H.264 output; falls back to OpenCV `mp4v` codec if FFmpeg is unavailable.
- **JSON report** — Machine-readable file containing video metadata, classification summary, and per-segment timing data.

---

## 7. Module Reference

### 7.1 `config.py`

**Purpose:** Centralised configuration management using Python dataclasses.

**Key types:**

| Class | Description |
|---|---|
| `PipelineConfig` | Top-level container; aggregates all sub-configs |
| `TrackNetConfig` | Ball detection parameters (model path, batch size, Hough params) |
| `YOLOConfig` | Player detection parameters (model path, confidence, bbox expansion) |
| `ClassifierConfig` | Classification weights, thresholds, temporal parameters |
| `CourtDetectorConfig` | Court detection geometry and Hough/Canny parameters |

**Key function:**

```python
def load_config(yaml_path: str | Path | None = None) -> PipelineConfig
```

Creates a `PipelineConfig` with built-in defaults, then recursively overlays any values found in the YAML file. Missing YAML keys retain their defaults.

---

### 7.2 `pipeline.py`

**Purpose:** Main orchestrator that coordinates all pipeline phases and provides the CLI entry point.

**Key functions:**

| Function | Description |
|---|---|
| `run_pipeline(video_path, output_dir, config)` | Executes the full end-to-end pipeline |
| `main()` | CLI entry point; parses arguments, loads config, applies overrides |
| `parse_args()` | Defines the `argparse` CLI interface |

**Process management:**

- `_kill_stale_pipeline_processes()` — Scans `/proc` for zombie or leftover pipeline processes and terminates them.
- `_register_exit_cleanup()` — Registers `atexit` and signal handlers (`SIGTERM`, `SIGINT`) to clean up GPU cache and child processes on exit.

---

### 7.3 `ball_tracker.py`

**Purpose:** Tennis ball detection using the TrackNet convolutional neural network.

**Class:** `BallTracker`

| Method | Description |
|---|---|
| `__init__(config, device)` | Loads TrackNet weights; optionally applies `torch.compile` |
| `detect(frames) → list[tuple]` | Runs batched inference on all frames; returns `(x, y)` or `(None, None)` per frame |
| `_postprocess(feature_map)` | Extracts ball coordinates from the model's heatmap output via thresholding and `cv2.HoughCircles` |
| `_remove_outliers(ball_track)` | Removes spatially-inconsistent detections where a point is far from (or missing) both its predecessor and successor |

**TrackNet inference details:**

- Input: 3 consecutive frames concatenated along the channel dimension (9 channels total), resized to 640×360.
- Output: Per-pixel classification logits, from which a heatmap is derived via `argmax`.
- Ball location is extracted by thresholding the heatmap and fitting circles with `cv2.HoughCircles`.
- Coordinates are scaled back to the original video resolution.
- Mixed precision (`torch.amp.autocast` with FP16) is used on CUDA devices.

---

### 7.4 `player_detector.py`

**Purpose:** Player/person detection using the Ultralytics YOLO framework.

**Class:** `PlayerDetector`

| Method | Description |
|---|---|
| `__init__(config, device)` | Loads YOLO model |
| `detect(frames) → list[dict]` | Batched inference returning per-frame player data |

**Per-frame output dict:**

```python
{
    "boxes": [[x1, y1, x2, y2], ...],          # Tight bounding boxes
    "expanded_boxes": [[x1, y1, x2, y2], ...],  # Expanded for ball-in-bbox gate
    "centers": [[cx, cy], ...],                  # Box centre points
    "count": int,                                # Number of detected players
    "confidences": [float, ...],                 # Detection confidence scores
}
```

The `expanded_boxes` are produced by inflating each bounding box by a fraction of its own dimensions: each side is expanded by `bbox_expand_h / 2 × box_width` horizontally (left and right) and `bbox_expand_v / 2 × box_height` vertically (top and bottom). With defaults of 0.60 and 0.30, this adds 30% of box width per side and 15% of box height per side. These expanded regions are used by the classifier's ball-in-bbox gate.

---

### 7.5 `court_detector.py`

**Purpose:** Court region of interest (ROI) detection using classical computer vision.

**Class:** `CourtDetector`

| Method | Description |
|---|---|
| `detect(frame) → ndarray \| None` | Main detection entry point; returns the padded polygon or `None` on failure |
| `is_point_in_court(x, y) → bool` | Point-in-mask test |
| `filter_ball_positions(positions) → list` | Filters ball detections to court region |
| `filter_player_data(info) → dict` | Filters players by foot position (bottom-centre of bbox) |
| `filter_player_data_list(data) → list` | Batch version of `filter_player_data` |
| `set_padding(px) → ndarray \| None` | Adjusts padding and rebuilds mask without re-detection |

**Internal pipeline:**

1. Canny edge detection → Hough line segment detection
2. Filter to steep (near-vertical) segments as sideline candidates
3. Split into left/right groups by midpoint x-coordinate
4. `cv2.fitLine` per group → extend to full frame height
5. Shift lines outward by `padding_px` perpendicular to line direction
6. Detect far-court baseline for cap polygon (prevents over-narrow top)
7. Build binary mask via `cv2.fillPoly`
8. Validate area (reject if < 5% or > 95% of frame)

---

### 7.6 `state_classifier.py`

**Purpose:** Multi-signal temporal classification of play vs. dead states.

**Class:** `StateClassifier`

| Method | Description |
|---|---|
| `classify(ball_positions, player_data) → dict` | Full classification pipeline; returns states, scores, segments, and summary |
| `_compute_play_score(det_rate, speed, players) → ndarray` | Weighted combination of normalised signals |
| `_hysteresis(scores, ...) → (states, gate_count)` | Dual-threshold state machine with optional bbox gate |
| `_filter_short_segments(states, min_play, min_dead)` | Merges segments below minimum duration (iterative, up to 3 passes) |
| `_filter_isolated_play(states) → (states, killed_count)` | Removes short play segments surrounded by disproportionate dead time |
| `_extract_segments(states) → list[dict]` | Converts frame-level states to timed segment list |
| `_build_summary(states, segments) → dict` | Computes aggregate statistics |

**Resolution-invariant speed normalisation:**

Ball speeds are scaled by the ratio of a reference diagonal (1920×1080) to the actual video diagonal. This ensures the `ball_speed_cap` parameter behaves consistently across resolutions.

---

### 7.7 `renderer.py`

**Purpose:** Annotated video output with frame-by-frame overlays.

**Function:** `render_output_video()`

**Overlay elements per frame:**

| Element | Visual |
|---|---|
| **Header bar** | Green (play) or red (dead) semi-transparent rectangle at top |
| **State label** | `"> PLAY"` or `"|| DEAD"` |
| **Timestamp** | Current time / total duration |
| **Play score** | Numerical score value |
| **Ball marker** | Yellow concentric circles at detected ball position |
| **Player boxes** | Cyan tight bounding boxes (BGR `255,200,0`) + white expanded bounding boxes |
| **Timeline bar** | Colour-coded bar at frame bottom with white playhead indicator |
| **Court polygon** | Cyan outline (BGR `255,255,0`; only when `court.debug_overlay` is enabled) |

**Encoding:**  
Frames are piped to FFmpeg via `subprocess.Popen` stdin for efficient single-pass H.264 encoding with `libx264`, `preset=fast`, `CRF=23`. If FFmpeg is unavailable, OpenCV's `VideoWriter` with `mp4v` codec is used as a fallback.

---

### 7.8 `io_utils.py`

**Purpose:** Video I/O, reporting, and system resource helpers.

| Function | Description |
|---|---|
| `probe_video(path) → (fps, w, h, n_frames)` | Reads video metadata via OpenCV properties; falls back to manual frame counting if `CAP_PROP_FRAME_COUNT` returns ≤ 0 |
| `decode_chunk(path, start, count) → list[ndarray]` | Decodes a specific frame range |
| `write_json_report(path, video, fps, classification)` | Writes JSON report file |
| `fmt_time(seconds) → str` | Formats seconds as `MM:SS.ss` |
| `get_available_ram_gb() → float` | Reads `MemAvailable` from `/proc/meminfo` |
| `get_rss_mb() → float` | Reads process RSS from `/proc/self/status` |
| `auto_chunk_size(w, h) → int` | Calculates chunk size from RAM (40% budget, clamped 500–5000, rounded to 100) |

---

## 8. Classification Algorithm Deep Dive

The state classifier converts continuous perception signals into a binary play/dead decision through a carefully designed signal processing pipeline:

### Step 1 — Per-frame Raw Signals

Three raw signals are computed for every frame:

- **`detected[i]`** — Binary: 1.0 if ball was detected, 0.0 otherwise.
- **`speeds[i]`** — Euclidean distance (pixels) the ball moved since the previous frame. Scaled by the resolution normalisation factor. Zero if either frame has no detection.
- **`player_counts[i]`** — Number of players detected in the frame.

### Step 2 — Windowed Feature Smoothing

Each signal is smoothed with a uniform (box) filter of size `window_size` (default: 25 frames ≈ 0.83s at 30fps):

```
detection_rate[i] = mean(detected[i-w/2 : i+w/2])
speed_smooth[i]   = mean(speeds[i-w/2 : i+w/2])
player_smooth[i]  = mean(player_counts[i-w/2 : i+w/2])
```

### Step 3 — Normalised Weighted Play Score

Each smoothed signal is normalised to [0, 1] by clipping at its respective cap:

$$\text{score}[i] = w_d \cdot \min\!\left(\frac{\text{det\_rate}[i]}{\text{det\_cap}},\; 1\right) + w_s \cdot \min\!\left(\frac{\text{speed}[i]}{\text{spd\_cap}},\; 1\right) + w_p \cdot \min\!\left(\frac{\text{players}[i]}{\text{plr\_cap}},\; 1\right)$$

Default weights: $w_d = 0.45$, $w_s = 0.35$, $w_p = 0.20$.

### Step 4 — Gaussian Temporal Smoothing

A Gaussian filter ($\sigma = 12.5$ frames) further smooths the play score to reduce rapid fluctuations.

### Step 5 — Hysteresis Thresholding

A state machine with two thresholds prevents rapid oscillation:

```
State = DEAD:
    if score[i] ≥ enter_play_threshold (0.55)  →  transition to PLAY
        (unless suppressed by bbox gate)

State = PLAY:
    if score[i] < exit_play_threshold (0.35)   →  transition to DEAD
```

The gap between thresholds (0.55 − 0.35 = 0.20) creates a dead zone that prevents chattering near the decision boundary.

### Step 6 — Ball-in-Bbox Gate

Before a dead→play transition is accepted, the system examines the last `bbox_gate_lookback` (13) frames. Of those frames where the ball was successfully detected, it computes the fraction where the ball fell inside any expanded player bounding box. If that fraction exceeds `bbox_gate_ratio` (50%), the transition is suppressed — the ball is likely being held or bounced by a player between points, not in active play. Frames with no ball detection are excluded from the ratio calculation.

### Step 7 — Minimum Duration Filter

Segments shorter than `min_play_duration` (1.5s) or `min_dead_duration` (2.0s) are merged into the surrounding state. This runs iteratively (up to 3 passes) to handle cascading merges.

### Step 8 — Isolated Play Filter

Short play segments (< `isolated_max_play` = 3.0s) that are surrounded by dead segments on both sides, where the play duration is less than `isolated_dead_ratio` (2.5×) the combined surrounding dead time, are reclassified as dead. This catches brief false positives from replays or ball boys.

---

## 9. Court Detection Algorithm

The court detector identifies the playing area to filter out off-court noise. It operates on a single frame (the first decoded frame) and produces a binary mask.

### Algorithm Steps

1. **Preprocessing:** Convert to grayscale → Gaussian blur (5×5) → Canny edge detection.

2. **Line detection:** `cv2.HoughLinesP` detects line segments. Segments are filtered to keep only "steep" lines (angle ≥ 20° from horizontal) as sideline candidates.

3. **Left/right grouping:** Segments are split by their midpoint x-coordinate relative to frame centre.

4. **Line fitting:** `cv2.fitLine` (L2 distance) fits a single line through all endpoints in each group. The fitted line is extended to the full frame height (y=0 to y=frame_height).

5. **Perpendicular padding:** Each sideline is shifted outward by `padding_px` pixels perpendicular to its direction vector. The direction of "outward" is verified and corrected if needed.

6. **Baseline cap:** The longest near-horizontal line segment in the upper portion of the frame is detected as the far-court baseline. Above this line, the ROI extends to full frame width, preventing the converging padded sidelines from clipping the upper court.

7. **Mask generation:** The final polygon (either a 4-point trapezoid or an 8-point capped polygon) is rasterised into a binary mask via `cv2.fillPoly`.

8. **Validation:** The mask area must be between 5% and 95% of the total frame area.

### Filtering Behaviour

- **Ball positions:** Off-court `(x, y)` coordinates are replaced with `(None, None)`.
- **Player boxes:** Players are retained only if their foot position (bottom-centre of bounding box) falls within the court mask.
- **Graceful degradation:** If detection fails, all filtering is disabled and the pipeline processes the full frame.

---

## 10. Memory Management & Chunking

The pipeline is designed to process arbitrarily long videos on memory-constrained systems.

### Auto-Chunk Sizing

When `chunk_frames = 0`, the system reads `/proc/meminfo` and allocates 40% of available RAM to frame storage:

$$\text{chunk\_size} = \left\lfloor \frac{0.40 \times \text{MemAvailable}}{H \times W \times 3 + 360 \times 640 \times 3} \right\rfloor$$

The result is clamped to [500, 5000] and rounded down to the nearest 100.

### Per-Chunk Cleanup

After each chunk is processed:

1. Python references to frame arrays are deleted.
2. `torch.cuda.empty_cache()` releases GPU memory.
3. `gc.collect()` runs the Python garbage collector.
4. `malloc_trim(0)` (via `ctypes`) returns freed heap memory to the OS — important because NumPy's large array allocations can otherwise fragment the heap.

### TrackNet Overlap

TrackNet uses a 3-frame sliding window. To maintain continuity across chunk boundaries, the last 2 frames of each chunk are carried forward as overlap for the next chunk. The overlapping results are trimmed from the output to avoid duplication.

---

## 11. System Benchmarking & Hardware Requirements

This section provides hardware sizing guidance to run the pipeline without OOM errors, based on analysis of the models loaded concurrently and the memory-aware chunking system.

> **Tested configuration:** The pipeline has been rigorously tested on **NVIDIA L4 (24 GiB) × 1 GPU | 8 vCPUs | 32 GiB RAM**. This serves as the verified reference platform.

### Concurrent Model Resource Profile

During Phase 2 (chunked detection), both models are loaded on the GPU simultaneously:

| Component | Model | Parameters | Input Size | Default Batch Size | Precision |
|---|---|---|---|---|---|
| **Ball Tracker** | TrackNet (CNN) + `torch.compile` | ~2–3M | 640×360×9 (3-frame windows) | 16 | FP16 (`torch.amp.autocast`) |
| **Player Detector** | YOLO26s (Ultralytics) | 9.5M / 20.7 GFLOPs | 640×640 | 32 | FP16 (`half=True`) |

YOLO26s is an edge-optimised model (NMS-free, DFL removed) with roughly half the parameters and FLOPs of equivalent YOLOv6 models, enabling efficient inference on modest GPUs.

### GPU VRAM Breakdown (Peak, During Phase 2)

| Item | Estimated VRAM |
|---|---|
| TrackNet weights + `torch.compile` graph cache | ~0.5–1.0 GB |
| TrackNet batch activations (bs=16, FP16) | ~200–400 MB |
| YOLO26s weights (fused) | ~100–200 MB |
| YOLO26s batch activations (bs=32, FP16) | ~300–600 MB |
| CUDA context + fragmentation overhead | ~500 MB |
| **Peak total** | **~1.6–2.7 GB** |

### System RAM Breakdown (Peak, 1080p)

The pipeline's `auto_chunk_size()` allocates **40% of available RAM** for frame buffers. At 1920×1080 each frame consumes ~6.2 MB uncompressed (H×W×3 bytes).

| Item | Estimated RAM |
|---|---|
| Frame buffer per chunk (e.g., 2000 frames × 6.2 MB) | ~12.4 GB |
| TrackNet resized copies per batch (640×360) | ~100–200 MB |
| Accumulated results (ball positions + player dicts for full video) | ~200–500 MB |
| Python / PyTorch / OpenCV overhead | ~1–2 GB |
| OS + other processes | ~2–3 GB |
| **Peak total** | **~16–18 GB** |

### CPU Usage Profile

| Task | CPU Characteristic |
|---|---|
| OpenCV frame decoding | Sequential, single-core; benefits from high turbo clock |
| `cv2.HoughCircles` post-processing | Per-frame CPU-bound (called per batch window) |
| State classification | Vectorised NumPy/SciPy rolling-window operations |
| `torch.compile` JIT compilation | CPU-intensive one-time cost at model load |
| Court detection (Phase 1b) | Single-frame Canny + Hough; negligible |

### Known Memory Safety Caveats

Two design interactions in the current pipeline can cause OOM under tight resource constraints:

1. **Chunk size is calculated before models load.** `auto_chunk_size()` reads `MemAvailable` in Phase 1, but TrackNet and YOLO don't load until Phase 2. Model loading + `torch.compile` JIT compilation can consume ~1–2 GB of additional system RAM *after* the chunk size was already decided. On systems with ≤16 GB, the calculated chunk may be too aggressive for the memory that's actually free during processing.

2. **Minimum chunk clamp overrides budget.** `auto_chunk_size()` enforces a floor of 500 frames (`max(500, ...)`). At 1080p, 500 frames = ~3.1 GB. On a busy 16 GB system where `MemAvailable` is ~7 GB at calculation time, the budget would suggest ~400 frames, but the clamp forces 500 — which, after model loading, can push the system into swap or OOM.

3. **`torch.compile` + `autocast` memory amplification.** The BallTracker sends FP32 tensors into a `torch.compile`'d model wrapped in `torch.amp.autocast(dtype=torch.float16)`. This specific pattern is subject to a [known PyTorch issue](https://github.com/pytorch/pytorch/issues/133637) where certain compiled + autocasted ops can spike GPU memory by up to 8×. On 4 GB GPUs this can trigger CUDA OOM unpredictably.

4. **Display GPUs have reduced usable VRAM.** A GPU driving a monitor (e.g., GTX 1650 as primary display) may have only ~3.5 GB effective VRAM after framebuffer allocation, further tightening the budget.

### Recommended System Configurations

#### Minimum (functional; OOM-safe with tuning)

| Resource | Specification | Notes |
|---|---|---|
| **CPU** | 4 cores / 8 threads @ 3.0 GHz | e.g., Intel i5-12400 / AMD Ryzen 5 5600 |
| **RAM** | 16 GB DDR4 | Requires manual `chunk_frames` override (see below) |
| **GPU** | 6 GB VRAM | e.g., GTX 1660 Super / RTX 2060; 4 GB cards risk CUDA OOM due to `torch.compile` + `autocast` spikes |

> ⚠️ **16 GB RAM requires manual configuration.** Set `chunk_frames: 300` in `config.yaml` to prevent the auto-chunker (which runs before model loading) from over-allocating. Also reduce `yolo.batch_size` to `16` and `tracknet.batch_size` to `8` to keep peak system + GPU memory in check. Close other applications before running.

> ⚠️ **4 GB VRAM GPUs (GTX 1650, etc.) are not recommended.** While models theoretically fit at ~1.6 GB peak, `torch.compile` + `autocast` memory amplification and display-GPU VRAM overhead make 4 GB unreliable. If you must use 4 GB, disable `torch.compile` by adding `model.eval()` without the compile step (requires a code change in `ball_tracker.py`), and reduce both batch sizes to 8.

#### Recommended (smooth operation, no OOM risk)

| Resource | Specification | Notes |
|---|---|---|
| **CPU** | 6 cores / 12 threads @ 3.5 GHz | e.g., Intel i5-13600K / AMD Ryzen 5 7600 |
| **RAM** | 32 GB DDR4/DDR5 | Chunk size ~2000+ frames at 1080p; comfortable headroom even after model loading |
| **GPU** | 8 GB VRAM | e.g., RTX 4060 / RTX 3060 8 GB; room for `torch.compile` overhead and larger batches |

#### Optimal (fast processing, 4K-ready)

| Resource | Specification | Notes |
|---|---|---|
| **CPU** | 8+ cores / 16 threads @ 4.0+ GHz | e.g., Intel i7-14700K / AMD Ryzen 7 7800X3D |
| **RAM** | 64 GB DDR5 | Large chunks (~4000–5000 frames); can push `chunk_frames` near max |
| **GPU** | 12+ GB VRAM | e.g., RTX 4070 Ti / RTX 3060 12 GB; safely increase batch sizes to 32+ / 64+ |

#### Reference Throughput (1080p, 10 min @ 30 fps ≈ 18K frames)

| Tier | Expected Speed | Wall Time |
|---|---|---|
| Minimum | ~1–2× realtime | ~5–10 min |
| Recommended | ~2–4× realtime | ~2.5–5 min |
| Optimal | ~4–6× realtime | ~1.5–2.5 min |

### Config Knobs to Tune if Hitting OOM

| Config Key | Default | Effect on Memory | Minimum-spec Override |
|---|---|---|---|
| `chunk_frames` | `0` (auto) | Lower → less system RAM per chunk | Set to `300` on 16 GB systems |
| `tracknet.batch_size` | `16` | Lower → less GPU VRAM | Set to `8` on 6 GB GPUs |
| `yolo.batch_size` | `32` | Lower → less GPU VRAM | Set to `16` on 6 GB GPUs |
| `render_video` | `false` | `true` adds a second full-video pass in RAM | Keep `false` on minimum spec |

---

## 12. CLI Reference

```
usage: pipeline.py [-h] --video VIDEO [--output OUTPUT] [--config CONFIG]
                   [--tracknet-batch N] [--yolo-batch N]
                   [--enter-threshold F] [--exit-threshold F]
                   [--bbox-expand-h F] [--bbox-expand-v F]
                   [--no-bbox-gate] [--no-court-detection]
                   [--court-padding N] [--court-debug]
```

| Argument | Default | Description |
|---|---|---|
| `--video` | *(required)* | Path to input video file |
| `--output` | `data/output` | Output directory |
| `--config` | `config/config.yaml` (if exists) | Path to YAML config file |
| `--tracknet-batch` | From config | TrackNet batch size |
| `--yolo-batch` | From config | YOLO batch size |
| `--enter-threshold` | From config | Hysteresis enter-play threshold |
| `--exit-threshold` | From config | Hysteresis exit-play threshold |
| `--bbox-expand-h` | From config | Horizontal bbox expansion ratio |
| `--bbox-expand-v` | From config | Vertical bbox expansion ratio |
| `--no-bbox-gate` | `False` | Disable ball-in-bbox gate |
| `--no-court-detection` | `False` | Disable court region detection |
| `--court-padding` | From config | Court sideline padding (pixels) |
| `--court-debug` | `False` | Draw court boundary on output video |

**Invocation methods:**

```bash
# Direct module execution
python src/pipeline.py --video data/videos/clip.mp4

# Package execution
python -m src --video data/videos/clip.mp4 --output data/output

# With custom config and overrides
python -m src.pipeline --video clip.mp4 --config my_config.yaml --enter-threshold 0.50
```

**Configuration precedence:** CLI arguments > YAML file > Built-in defaults.

---

## 13. Batch Processing

The `run_all.sh` script processes all `.mp4` files in a specified directory:

```bash
bash run_all.sh
```

**Behaviour:**

- Scans `01. Videos/` for `.mp4` files.
- Activates `venv/` if present.
- Runs the pipeline on each video sequentially.
- Writes outputs to `data/output/`, logs to `data/logs/`.
- Prints per-video success/failure markers.
- Prints a summary with total, succeeded, and failed counts.
- Uses `set -euo pipefail` for strict error handling (but individual video failures are caught and counted).

---

## 14. Output Formats

### Annotated Video (`*_analysis.mp4`)

An H.264-encoded copy of the input with overlaid annotations:

- Colour-coded header bar (green = play, red = dead)
- State label, timestamp, and play score
- Ball detection markers (yellow circles)
- Player bounding boxes (cyan = tight, white = expanded)
- Timeline bar at bottom with playhead indicator

### JSON Report (`*_report.json`)

```json
{
  "video": "clip.mp4",
  "fps": 30.0,
  "total_frames": 9000,
  "total_duration": 300.0,
  "play_frames": 5400,
  "dead_frames": 3600,
  "play_time": 180.0,
  "dead_time": 120.0,
  "play_pct": 60.0,
  "dead_pct": 40.0,
  "num_segments": 24,
  "num_play_segments": 12,
  "num_dead_segments": 12,
  "segments": [
    {
      "type": "dead",
      "start_frame": 0,
      "end_frame": 150,
      "start_sec": 0.0,
      "end_sec": 5.0,
      "duration_sec": 5.0
    },
    {
      "type": "play",
      "start_frame": 150,
      "end_frame": 600,
      "start_sec": 5.0,
      "end_sec": 20.0,
      "duration_sec": 15.0
    }
  ]
}
```

---

## 15. Performance Considerations

### GPU Utilisation

- Both TrackNet and YOLO use **batched inference** to maximise GPU throughput.
- **FP16 mixed precision** (`torch.amp.autocast`) is enabled for TrackNet on CUDA devices.
- YOLO runs with `half=True` for FP16 inference.
- `torch.compile` is attempted on the TrackNet model for additional optimisation (PyTorch 2.0+).

### Throughput Estimates

- Typical processing speed is ~24× slower than realtime on consumer GPUs (estimated ~0.042s per frame).
- The rendering phase is I/O-bound; FFmpeg pipe encoding is significantly faster than OpenCV's `VideoWriter`.

### Bottlenecks

| Phase | Bottleneck | Mitigation |
|---|---|---|
| Video decode | CPU-bound sequential read | Sequential reading avoids seek overhead |
| Ball detection | GPU compute | Batching, FP16, torch.compile |
| Player detection | GPU compute | Batching, FP16 |
| Classification | CPU (negligible) | Vectorised NumPy/SciPy operations |
| Rendering | I/O + CPU | FFmpeg stdin pipe (single-pass) |

### Tips for Large Videos

- Increase `chunk_frames` if you have ample RAM to reduce chunk transitions.
- Increase `tracknet.batch_size` and `yolo.batch_size` if GPU VRAM permits.
- Set `render_video: false` to skip video rendering and only generate the JSON report.
- Use `--no-court-detection` if the camera angle doesn't allow reliable sideline detection.

---

## 16. Extending the Pipeline

### Adding a New Detection Module

1. Create a new file in `src/` (e.g., `src/serve_detector.py`).
2. Define a dataclass config in `config.py` and add it to `PipelineConfig`.
3. Update `load_config()` to handle the new config section.
4. Instantiate and call the module in the appropriate phase within `pipeline.py`.
5. Feed its output into `StateClassifier.classify()` or create a new post-processing step.

### Modifying the Classification Logic

The classifier's scoring weights and thresholds are fully exposed via configuration. For deeper changes:

- Add new signal components in `StateClassifier.classify()` (Step 1).
- Add corresponding weights and caps in `ClassifierConfig`.
- Update `_compute_play_score()` to include the new signal.

### Custom Video Overlays

Add new drawing calls in `renderer._annotate_frame()`. The function receives all per-frame data (ball position, player info, state, score) and returns the annotated frame.

### Supporting Non-Linux Systems

The following components are Linux-specific and would need platform-conditional alternatives:

- `/proc/meminfo` reading in `io_utils.get_available_ram_gb()`
- `/proc/self/status` reading in `io_utils.get_rss_mb()`
- `/proc` scanning in `pipeline._kill_stale_pipeline_processes()`
- `libc.so.6` `malloc_trim` call in the chunked loop
