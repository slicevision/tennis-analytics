"""Video I/O, JSON reporting, and system-resource helpers."""

from __future__ import annotations

import json
import os

import cv2
import numpy as np


# ===================================================================
#  Video I/O
# ===================================================================
def probe_video(video_path: str) -> tuple[float, int, int, int]:
    """Read video metadata without loading frames.

    Returns:
        fps, width, height, total_frames
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if total_frames <= 0:
        cap = cv2.VideoCapture(video_path)
        total_frames = 0
        while cap.read()[0]:
            total_frames += 1
        cap.release()

    duration = total_frames / fps
    mem_gb = total_frames * height * width * 3 / 1e9
    print(f"  Video: {total_frames} frames  |  {width}×{height} @ {fps:.1f} fps  |  "
          f"Duration {fmt_time(duration)}  |  Full RAM ~{mem_gb:.1f} GB")
    return fps, width, height, total_frames


def decode_chunk(
    video_path: str,
    start_frame: int,
    count: int,
) -> list[np.ndarray]:
    """Decode a specific range of frames from a video file.

    Args:
        video_path:   path to video
        start_frame:  0-based index of first frame to read
        count:        number of frames to read

    Returns:
        list of BGR uint8 numpy arrays (may be shorter if video ends)
    """
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    for _ in range(count):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


# ===================================================================
#  JSON Report
# ===================================================================
def write_json_report(
    output_path: str,
    video_path: str,
    fps: float,
    classification: dict,
):
    """Write a machine-readable JSON report."""
    summary = classification["summary"]
    segments = classification["segments"]

    report = {
        "video": os.path.basename(video_path),
        "fps": fps,
        **summary,
        "segments": segments,
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  [Report] {output_path}")


# ===================================================================
#  Formatting Utilities
# ===================================================================
def fmt_time(seconds: float) -> str:
    """Format seconds as MM:SS.s"""
    m = int(seconds) // 60
    s = seconds - m * 60
    return f"{m:02d}:{s:05.2f}"


# ===================================================================
#  Memory / System Helpers
# ===================================================================
def get_available_ram_gb() -> float:
    """Read MemAvailable from /proc/meminfo (Linux)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1e6
    except (FileNotFoundError, ValueError):
        pass
    return 16.0


def get_rss_mb() -> float:
    """Read current process RSS from /proc/self/status."""
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except (FileNotFoundError, ValueError):
        pass
    return 0.0


def auto_chunk_size(vid_w: int, vid_h: int) -> int:
    """Calculate a safe chunk size based on available system RAM."""
    available_gb = get_available_ram_gb()
    budget_bytes = available_gb * 0.40 * 1e9

    frame_bytes = vid_h * vid_w * 3
    resized_bytes = 360 * 640 * 3
    per_frame = frame_bytes + resized_bytes

    chunk = int(budget_bytes / per_frame)
    chunk = max(500, min(chunk, 5000))
    chunk = (chunk // 100) * 100
    return chunk
