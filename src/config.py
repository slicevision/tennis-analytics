from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


# ===================================================================
#  YAML loader
# ===================================================================
def _apply_dict(dc_instance: Any, overrides: dict) -> None:
    """Recursively apply a flat dict of overrides onto a dataclass instance."""
    for key, value in overrides.items():
        if not hasattr(dc_instance, key):
            continue
        current = getattr(dc_instance, key)
        # If the current attribute is itself a dataclass, recurse
        if hasattr(current, "__dataclass_fields__") and isinstance(value, dict):
            _apply_dict(current, value)
        else:
            setattr(dc_instance, key, type(current)(value))


def load_config(yaml_path: str | Path | None = None) -> "PipelineConfig":
    """Create a PipelineConfig, optionally loading overrides from a YAML file.

    Any key present in the YAML overrides the built-in default; keys
    absent from the YAML keep their default values.
    """
    config = PipelineConfig()
    if yaml_path is None:
        return config

    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        return config

    # Top-level scalar overrides (device, chunk_frames)
    for key in ("device", "chunk_frames", "use_fp16", "render_video"):
        if key in raw:
            setattr(config, key, type(getattr(config, key))(raw[key]))

    # Sub-config sections
    section_map = {
        "tracknet": config.tracknet,
        "yolo": config.yolo,
        "classifier": config.classifier,
        "court": config.court,
    }
    for section_name, dc_instance in section_map.items():
        section = raw.get(section_name)
        if isinstance(section, dict):
            _apply_dict(dc_instance, section)

    return config


@dataclass
class TrackNetConfig:
    """TrackNet ball detection settings."""
    model_path: str = "tracknet/weights/model_best.pt"
    input_width: int = 640
    input_height: int = 360
    batch_size: int = 16
    heatmap_threshold: int = 127
    # HoughCircles params
    hough_dp: int = 1
    hough_min_dist: int = 1
    hough_param1: int = 50
    hough_param2: int = 2
    hough_min_radius: int = 2
    hough_max_radius: int = 7
    # Outlier removal
    outlier_max_dist: float = 100.0   # max px distance between frames


@dataclass
class YOLOConfig:
    """YOLO player detection settings."""
    model_path: str = "weights/yolo26s.pt"
    batch_size: int = 32
    imgsz: int = 640
    conf_threshold: float = 0.45
    person_class_id: int = 0
    max_det: int = 10
    # Bounding box expansion for ball-in-bbox gate
    bbox_expand_h: float = 0.60       # horizontal expansion ratio
    bbox_expand_v: float = 0.30       # vertical expansion ratio


@dataclass
class ClassifierConfig:
    """Play/dead state classification settings."""
    # Rolling window for feature computation (frames)
    window_size: int = 25

    # --- Play score component weights ---
    weight_detection_rate: float = 0.45
    weight_ball_speed: float = 0.35
    weight_player_count: float = 0.20

    # --- Normalization caps ---
    detection_rate_cap: float = 0.40
    ball_speed_cap: float = 8.0       # px/frame at reference resolution
    player_count_cap: float = 2.0
    # Reference resolution for ball_speed_cap (speeds are scaled to this)
    ball_speed_ref_width: int = 1920
    ball_speed_ref_height: int = 1080

    # --- Temporal smoothing ---
    smooth_sigma: float = 12.5        # Gaussian sigma (frames)

    # --- Hysteresis thresholds ---
    enter_play_threshold: float = 0.55
    exit_play_threshold: float = 0.35

    # --- Ball-in-bbox dead→play transition gate ---
    use_bbox_gate: bool = True
    bbox_gate_lookback: int = 13      # frames to check
    bbox_gate_ratio: float = 0.5      # suppression threshold

    # --- Minimum segment durations (seconds) ---
    min_play_duration: float = 1.5
    min_dead_duration: float = 2.0

    # --- Isolated play filter ---
    use_isolated_play_filter: bool = True
    isolated_max_play: float = 3.0    # max play duration to target (seconds)
    isolated_dead_ratio: float = 2.5  # play vs surrounding dead ratio


@dataclass
class CourtDetectorConfig:
    """Sideline-based court region detection settings."""
    enabled: bool = True
    # Perpendicular padding from each detected sideline (pixels)
    padding_px: int = 350
    # Steep line detection (sideline candidates)
    steep_min_angle: float = 20.0     # min degrees from horizontal
    hough_threshold: int = 50
    min_line_length: int = 150        # min segment length (pixels)
    max_line_gap: int = 15            # max gap to merge segments
    # Canny edge detection
    canny_low: int = 30
    canny_high: int = 100
    # Polygon validation (fraction of frame area)
    min_court_area_pct: float = 0.05
    max_court_area_pct: float = 0.95
    # Debug: draw padded boundary on output video
    debug_overlay: bool = False
    # Baseline cap: use the longest horizontal line in the upper frame
    # to prevent the converging padded sidelines from excluding the
    # upper region.  Above the baseline, full frame width is valid.
    baseline_cap_enabled: bool = True
    baseline_search_frac: float = 0.55   # search top N fraction of frame
    baseline_max_angle: float = 15.0     # max degrees from horizontal
    baseline_min_length: int = 80        # min segment length (pixels)
    baseline_hough_threshold: int = 40
    baseline_pad_px: int = 50            # shift baseline upward (pixels)


@dataclass
class PipelineConfig:
    """Top-level pipeline configuration."""
    tracknet: TrackNetConfig = field(default_factory=TrackNetConfig)
    yolo: YOLOConfig = field(default_factory=YOLOConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    court: CourtDetectorConfig = field(default_factory=CourtDetectorConfig)
    device: str = "cuda"
    use_fp16: bool = True
    # Chunk size for processing (frames). 0 = auto-calculate from available RAM.
    chunk_frames: int = 0
    # Whether to render an annotated output video (false = JSON report only)
    render_video: bool = True
