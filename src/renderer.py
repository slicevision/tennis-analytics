"""Annotated video rendering (header overlay, ball/player markers, timeline)."""

from __future__ import annotations

import os
import subprocess
import time

import cv2
import numpy as np

from .io_utils import fmt_time


# ===================================================================
#  Public API
# ===================================================================
def render_output_video(
    video_path: str,
    output_path: str,
    ball_positions: list[tuple],
    player_data: list[dict],
    states: np.ndarray,
    play_scores: np.ndarray,
    fps: float,
    court_polygon: np.ndarray | None = None,
):
    """Re-read the source video and write an annotated copy."""
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = len(states)

    timeline_colours = np.zeros((width, 3), dtype=np.uint8)
    for px in range(width):
        fi = min(int(px / width * total_frames), total_frames - 1)
        if states[fi] == 1:
            timeline_colours[px] = (0, 180, 0)
        else:
            timeline_colours[px] = (0, 0, 180)

    # Try to pipe directly to ffmpeg for single-pass H.264 encoding
    ffmpeg_proc = None
    writer = None
    try:
        ffmpeg_proc = subprocess.Popen(
            [
                "ffmpeg", "-y", "-loglevel", "error",
                "-f", "rawvideo", "-vcodec", "rawvideo",
                "-s", f"{width}x{height}", "-pix_fmt", "bgr24",
                "-r", str(fps),
                "-i", "-",
                "-c:v", "libx264", "-preset", "fast",
                "-crf", "23", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart",
                "-an", output_path,
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError:
        # ffmpeg not installed — fall back to OpenCV VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print("  [Render] ffmpeg not found — falling back to mp4v codec")

    t0 = time.perf_counter()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= total_frames:
            break

        annotated = _annotate_frame(
            frame, frame_idx, total_frames,
            ball_positions[frame_idx],
            player_data[frame_idx],
            states[frame_idx],
            play_scores[frame_idx],
            fps, width, height,
            timeline_colours,
        )
        if court_polygon is not None:
            pts = court_polygon.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated, [pts], True, (255, 255, 0), 1,
                          cv2.LINE_AA)

        if ffmpeg_proc is not None:
            try:
                ffmpeg_proc.stdin.write(annotated.tobytes())
            except BrokenPipeError:
                print(f"  [Render] ffmpeg pipe broke at frame {frame_idx}")
                break
        else:
            writer.write(annotated)
        frame_idx += 1

        if frame_idx % 500 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  [Render] {frame_idx}/{total_frames} frames  "
                  f"({elapsed:.1f}s)")

    cap.release()

    if ffmpeg_proc is not None:
        ffmpeg_proc.stdin.close()
        ffmpeg_proc.wait()
        if ffmpeg_proc.returncode == 0:
            print(f"  [Render] Encoded to H.264 (single pass)")
        else:
            stderr = ffmpeg_proc.stderr.read().decode(errors="replace")
            print(f"  [Render] ffmpeg error: {stderr[:300]}")
    elif writer is not None:
        writer.release()

    t_total = time.perf_counter() - t0
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"  [Render] Done in {t_total:.1f}s  |  "
          f"{output_path}  ({size_mb:.1f} MB)")


# ===================================================================
#  Internal helpers
# ===================================================================
def _annotate_frame(
    frame: np.ndarray,
    idx: int,
    total: int,
    ball_pos: tuple,
    player_info: dict,
    state: int,
    score: float,
    fps: float,
    w: int,
    h: int,
    timeline_colours: np.ndarray,
) -> np.ndarray:
    """Draw all overlays on a single frame."""
    out = frame.copy()

    # --- Header bar ---
    HEADER_H = 44
    header_col = (0, 180, 0) if state == 1 else (0, 0, 180)
    overlay = out.copy()
    cv2.rectangle(overlay, (0, 0), (w, HEADER_H), header_col, -1)
    cv2.addWeighted(overlay, 0.40, out, 0.60, 0, out)

    # State label
    label = "> PLAY" if state == 1 else "|| DEAD"
    cv2.putText(out, label, (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2,
                cv2.LINE_AA)

    # Time
    cur_sec = idx / fps
    tot_sec = total / fps
    time_str = f"{fmt_time(cur_sec)} / {fmt_time(tot_sec)}"
    _put_text_right(out, time_str, w - 12, 32, 0.65, (255, 255, 255), 2)

    # Score
    score_str = f"Score {score:.2f}"
    cv2.putText(out, score_str, (w // 2 - 50, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1,
                cv2.LINE_AA)

    # --- Ball marker ---
    if ball_pos[0] is not None:
        bx, by = int(round(ball_pos[0])), int(round(ball_pos[1]))
        cv2.circle(out, (bx, by), 9, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(out, (bx, by), 3, (0, 255, 255), -1, cv2.LINE_AA)

    # --- Player bounding boxes ---
    for box in player_info["boxes"]:
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        cv2.rectangle(out, (x1, y1), (x2, y2), (255, 200, 0), 2)

    # --- Expanded bounding boxes ---
    for ebox in player_info.get("expanded_boxes", []):
        ex1, ey1, ex2, ey2 = (int(round(v)) for v in ebox)
        cv2.rectangle(out, (ex1, ey1), (ex2, ey2), (255, 255, 255), 1)

    # --- Timeline bar ---
    BAR_H = 14
    bar_top = h - BAR_H
    for row in range(bar_top, h):
        out[row, :] = timeline_colours
    cv2.addWeighted(out[bar_top:h], 0.75, frame[bar_top:h], 0.25, 0,
                    out[bar_top:h])
    px_x = int(idx / total * w)
    px_x = min(px_x, w - 1)
    cv2.line(out, (px_x, bar_top - 3), (px_x, h), (255, 255, 255), 2)

    return out


def _put_text_right(img, text, right_x, y, scale, colour, thickness):
    """Put text aligned to a right-edge x coordinate."""
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale,
                                 thickness)
    cv2.putText(img, text, (right_x - tw, y),
                cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thickness,
                cv2.LINE_AA)
