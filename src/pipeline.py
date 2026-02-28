import argparse
import atexit
import ctypes
import gc
import os
import signal
import sys
import time

import cv2
import numpy as np
import torch

# Force line-buffered stdout for redirected output
try:
    sys.stdout.reconfigure(line_buffering=True)
except AttributeError:
    pass

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)

from .config import PipelineConfig, load_config   # noqa: E402
from .ball_tracker import BallTracker              # noqa: E402
from .player_detector import PlayerDetector        # noqa: E402
from .state_classifier import StateClassifier      # noqa: E402
from .court_detector import CourtDetector          # noqa: E402
from .io_utils import (                            # noqa: E402
    probe_video, decode_chunk, write_json_report,
    fmt_time, get_available_ram_gb, get_rss_mb, auto_chunk_size,
)
from .renderer import render_output_video          # noqa: E402


# ===================================================================
#  Internal helpers
# ===================================================================
def _print_gpu_info():
    """Print GPU name and memory."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU: {name}  |  VRAM: {mem:.1f} GB")
    else:
        print("  GPU: not available – running on CPU (will be slow)")


# ===================================================================
#  Process Cleanup
# ===================================================================
def _kill_stale_pipeline_processes():
    """Kill leftover pipeline processes and zombies from previous runs."""
    my_pid = os.getpid()
    script_patterns = (b"pipeline.py", b"tennis_pipeline.py")
    killed = []

    # Reap zombie children
    try:
        while True:
            pid, _ = os.waitpid(-1, os.WNOHANG)
            if pid == 0:
                break
    except ChildProcessError:
        pass

    # Scan /proc for stale processes
    try:
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            pid = int(entry)
            if pid == my_pid:
                continue
            try:
                cmdline = open(f"/proc/{pid}/cmdline", "rb").read()
                status = open(f"/proc/{pid}/status", "r").read()
            except (FileNotFoundError, PermissionError):
                continue

            is_pipeline = any(p in cmdline for p in script_patterns)
            is_zombie = "State:\tZ" in status

            if is_pipeline or is_zombie:
                try:
                    os.kill(pid, signal.SIGKILL)
                    killed.append(pid)
                except (ProcessLookupError, PermissionError):
                    pass
    except Exception:
        pass

    if killed:
        print(f"  [Cleanup] Killed {len(killed)} stale/zombie process(es): "
              f"{killed}")


def _register_exit_cleanup():
    """Register cleanup handlers for process exit and signals."""
    def _cleanup():
        try:
            while True:
                pid, _ = os.waitpid(-1, os.WNOHANG)
                if pid == 0:
                    break
        except ChildProcessError:
            pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    atexit.register(_cleanup)

    def _signal_handler(signum, frame):
        print(f"\n  [Cleanup] Caught signal {signum}, cleaning up …")
        _cleanup()
        sys.exit(1)

    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)


# ===================================================================
#  Main Pipeline
# ===================================================================
def run_pipeline(video_path: str, output_dir: str, config: PipelineConfig):
    """Run the end-to-end play/dead time detection pipeline."""
    total_t0 = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    video_path = os.path.abspath(video_path)
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(video_path))[0]

    print("=" * 64)
    print(" Tennis Play/Dead Time Detection Pipeline")
    print("=" * 64)
    _kill_stale_pipeline_processes()
    _register_exit_cleanup()
    _print_gpu_info()
    print()

    # ------------------------------------------------------------------
    # Phase 1 – Probe video
    # ------------------------------------------------------------------
    print("[Phase 1] Probing video …")
    fps, vid_w, vid_h, total_frames = probe_video(video_path)

    if config.chunk_frames > 0:
        chunk_size = config.chunk_frames
    else:
        chunk_size = auto_chunk_size(vid_w, vid_h)

    tracknet_overlap = 2
    n_chunks = (total_frames + chunk_size - 1) // chunk_size

    chunk_ram_gb = chunk_size * vid_h * vid_w * 3 / 1e9
    sys_ram_gb = get_available_ram_gb()
    print(f"  System RAM: {sys_ram_gb:.1f} GB available")
    print(f"  Chunk size: {chunk_size} frames (~{chunk_ram_gb:.1f} GB)  "
          f"×  {n_chunks} chunk(s)")
    est_secs = n_chunks * chunk_size * 0.042
    print(f"  Estimated time: ~{fmt_time(est_secs)}")
    print()

    # ------------------------------------------------------------------
    # Phase 1b – Court region detection
    # ------------------------------------------------------------------
    court_detector = None
    if config.court.enabled:
        print("[Phase 1b] Detecting court region …")
        court_detector = CourtDetector(config.court)
        sample_frames = decode_chunk(video_path, 0, 1)
        if sample_frames:
            court_detector.detect(sample_frames[0])
            del sample_frames
        if not court_detector.detected:
            print("  Court detection failed — processing full frame "
                  "(no ROI filtering)")
            court_detector = None
        print()

    # ------------------------------------------------------------------
    # Phase 2 – Chunked detection
    # ------------------------------------------------------------------
    print("[Phase 2] Loading models …")
    ball_tracker = BallTracker(config.tracknet, device=device)
    player_detector = PlayerDetector(config.yolo, device=device)
    print()

    all_ball_positions = []
    all_player_data = []

    # Open video once and read sequentially (avoids per-chunk seek overhead)
    cap = cv2.VideoCapture(video_path)
    overlap_frames: list[np.ndarray] = []

    for c_idx in range(n_chunks):
        chunk_start = c_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, total_frames)
        n_in_chunk = chunk_end - chunk_start

        print(f"[Chunk {c_idx+1}/{n_chunks}] Frames {chunk_start}–{chunk_end-1} "
              f"({n_in_chunk} frames, {fmt_time(n_in_chunk/fps)})")

        # --- Decode chunk (sequential read, no seeking) ---
        t_dec = time.perf_counter()
        new_frames: list[np.ndarray] = []
        for _ in range(n_in_chunk):
            ret, frame = cap.read()
            if not ret:
                break
            new_frames.append(frame)

        # Prepend overlap frames for TrackNet's 3-frame sliding window
        if c_idx > 0 and overlap_frames:
            chunk_frames = overlap_frames + new_frames
        else:
            chunk_frames = new_frames

        # Keep last frames for next chunk's overlap
        if len(new_frames) >= tracknet_overlap:
            overlap_frames = new_frames[-tracknet_overlap:]
        else:
            overlap_frames = list(new_frames)

        print(f"  Decoded {len(chunk_frames)} frames in "
              f"{time.perf_counter()-t_dec:.1f}s")

        # --- Ball detection ---
        ball_pos = ball_tracker.detect(chunk_frames)

        # Trim overlap frames from ball results (first chunk has no overlap)
        if c_idx > 0:
            ball_pos = ball_pos[tracknet_overlap:]

        # --- Player detection ---
        if c_idx > 0:
            yolo_frames = chunk_frames[tracknet_overlap:]
        else:
            yolo_frames = chunk_frames
        player_dat = player_detector.detect(yolo_frames)

        # --- Court ROI filtering ---
        if court_detector is not None and court_detector.detected:
            pre_ball = sum(1 for p in ball_pos if p[0] is not None)
            ball_pos = court_detector.filter_ball_positions(ball_pos)
            post_ball = sum(1 for p in ball_pos if p[0] is not None)

            pre_players = sum(d["count"] for d in player_dat)
            player_dat = court_detector.filter_player_data_list(player_dat)
            post_players = sum(d["count"] for d in player_dat)

            print(f"  [CourtROI] Balls kept: {post_ball}/{pre_ball}  |  "
                  f"Player boxes kept: {post_players}/{pre_players}")

        # --- Accumulate ---
        all_ball_positions.extend(ball_pos)
        all_player_data.extend(player_dat)

        del chunk_frames, new_frames, yolo_frames, ball_pos, player_dat
        torch.cuda.empty_cache()

        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            pass

        rss_mb = get_rss_mb()
        elapsed_so_far = time.perf_counter() - total_t0
        chunks_done = c_idx + 1
        if chunks_done < n_chunks:
            eta = elapsed_so_far / chunks_done * (n_chunks - chunks_done)
            eta_str = f"  |  ETA: ~{fmt_time(eta)}"
        else:
            eta_str = ""
        print(f"  Chunk complete  |  Accumulated: {len(all_ball_positions)} frames"
              f"  |  RSS: {rss_mb:.0f} MB{eta_str}")
        print()

    cap.release()
    del overlap_frames

    # Free models
    del ball_tracker, player_detector
    torch.cuda.empty_cache()

    actual_frames = len(all_ball_positions)
    if actual_frames != total_frames:
        print(f"  Note: Probed {total_frames} frames but decoded {actual_frames} "
              f"— using actual count")
        total_frames = actual_frames

    assert len(all_player_data) == total_frames, \
        f"Player data {len(all_player_data)} != {total_frames} frames"

    # ------------------------------------------------------------------
    # Phase 3 – State classification
    # ------------------------------------------------------------------
    print("[Phase 3] Classifying play/dead states …")
    classifier = StateClassifier(config.classifier, fps=fps,
                                  frame_width=vid_w, frame_height=vid_h)
    classification = classifier.classify(all_ball_positions, all_player_data)
    print()

    states = classification["states"]
    play_scores = classification["play_scores"]

    # ------------------------------------------------------------------
    # Phase 4 – Output generation
    # ------------------------------------------------------------------
    print("[Phase 4] Generating outputs …")

    # 4a. Annotated video
    out_video = None
    if config.render_video:
        out_video = os.path.join(output_dir, f"{base_name}_analysis.mp4")
        court_poly = (
            court_detector.padded_polygon
            if court_detector is not None and court_detector.detected
            and config.court.debug_overlay
            else None
        )
        render_output_video(
            video_path, out_video,
            all_ball_positions, all_player_data,
            states, play_scores, fps,
            court_polygon=court_poly,
        )
    else:
        print("  [Render] Skipped (render_video: false)")

    # 4b. JSON report
    out_json = os.path.join(output_dir, f"{base_name}_report.json")
    write_json_report(out_json, video_path, fps, classification)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    total_elapsed = time.perf_counter() - total_t0
    summary = classification["summary"]

    print()
    print("=" * 64)
    print(" Pipeline complete")
    print("=" * 64)
    print(f"  Total wall time   : {total_elapsed:.1f}s")
    print(f"  Video duration    : {fmt_time(summary['total_duration'])}")
    print(f"  Speed             : {summary['total_duration']/total_elapsed:.1f}× realtime")
    if court_detector is not None and court_detector.detected:
        print(f"  Court filtering   : Active "
              f"(padding {config.court.padding_px}px)")
    else:
        print(f"  Court filtering   : "
              f"{'Disabled' if not config.court.enabled else 'Not detected'}")
    print(f"  Play time         : {summary['play_time']:.1f}s  "
          f"({summary['play_pct']:.1f}%)")
    print(f"  Dead time         : {summary['dead_time']:.1f}s  "
          f"({summary['dead_pct']:.1f}%)")
    print(f"  Segments          : {summary['num_segments']} total  "
          f"({summary['num_play_segments']} play, "
          f"{summary['num_dead_segments']} dead)")
    if out_video:
        print(f"  Output video      : {out_video}")
    else:
        print(f"  Output video      : (skipped)")
    print(f"  Output report     : {out_json}")
    print("=" * 64)


# ===================================================================
#  CLI
# ===================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Tennis Play/Dead Time Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--video", required=True,
                   help="Path to input video file")
    p.add_argument("--output", default="data/output",
                   help="Output directory (default: data/output)")
    p.add_argument("--config", default=None,
                   help="Path to YAML config file (default: config.yaml if it exists)")

    # Optional overrides
    p.add_argument("--tracknet-batch", type=int, default=None,
                   help="TrackNet batch size (default: 64)")
    p.add_argument("--yolo-batch", type=int, default=None,
                   help="YOLO batch size (default: 32)")
    p.add_argument("--enter-threshold", type=float, default=None,
                   help="Hysteresis enter-play threshold (default: 0.55)")
    p.add_argument("--exit-threshold", type=float, default=None,
                   help="Hysteresis exit-play threshold (default: 0.35)")
    p.add_argument("--bbox-expand-h", type=float, default=None,
                   help="Horizontal bbox expansion ratio (default: 0.60)")
    p.add_argument("--bbox-expand-v", type=float, default=None,
                   help="Vertical bbox expansion ratio (default: 0.30)")
    p.add_argument("--no-bbox-gate", action="store_true",
                   help="Disable the ball-in-bbox dead→play gate")
    p.add_argument("--no-court-detection", action="store_true",
                   help="Disable automatic court region detection")
    p.add_argument("--court-padding", type=int, default=None,
                   help="Court sideline padding in pixels (default: 350)")
    p.add_argument("--court-debug", action="store_true",
                   help="Draw padded court boundary on output video")
    return p.parse_args()


def main():
    args = parse_args()

    # Resolve config file: explicit --config > default config.yaml > built-in defaults
    config_path = args.config
    if config_path is None:
        default_path = os.path.join(_PROJECT_ROOT, "config", "config.yaml")
        if os.path.isfile(default_path):
            config_path = default_path

    config = load_config(config_path)
    if config_path:
        print(f"  Loaded config: {config_path}")

    # Apply CLI overrides
    if args.tracknet_batch is not None:
        config.tracknet.batch_size = args.tracknet_batch
    if args.yolo_batch is not None:
        config.yolo.batch_size = args.yolo_batch
    if args.enter_threshold is not None:
        config.classifier.enter_play_threshold = args.enter_threshold
    if args.exit_threshold is not None:
        config.classifier.exit_play_threshold = args.exit_threshold
    if args.bbox_expand_h is not None:
        config.yolo.bbox_expand_h = args.bbox_expand_h
    if args.bbox_expand_v is not None:
        config.yolo.bbox_expand_v = args.bbox_expand_v
    if args.no_bbox_gate:
        config.classifier.use_bbox_gate = False
    if args.no_court_detection:
        config.court.enabled = False
    if args.court_padding is not None:
        config.court.padding_px = args.court_padding
    if args.court_debug:
        config.court.debug_overlay = True

    run_pipeline(args.video, args.output, config)


if __name__ == "__main__":
    main()
