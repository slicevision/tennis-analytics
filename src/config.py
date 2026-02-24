from dataclasses import dataclass, field


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
    model_path: str = "yolo26s.pt"
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
    ball_speed_cap: float = 8.0       # px/frame
    player_count_cap: float = 2.0

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
