import gc
import sys
import os
import time
import numpy as np
import cv2
import torch

# Resolve TrackNet model import
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "tracknet"))
from model import BallTrackerNet  # noqa: E402

from config import TrackNetConfig  # noqa: E402


class BallTracker:
    """TrackNet-based tennis ball detector with batched GPU inference."""

    def __init__(self, config: TrackNetConfig, device: str = "cuda"):
        self.cfg = config
        self.device = device
        self.model = self._load_model()

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------
    def _load_model(self) -> BallTrackerNet:
        model = BallTrackerNet()
        ckpt_path = os.path.join(_PROJECT_ROOT, self.cfg.model_path)
        state = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        model.load_state_dict(state)
        model = model.to(self.device)
        model.eval()
        return model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect(self, frames: list[np.ndarray]) -> list[tuple]:
        """
        Run ball detection on all video frames.

        Args:
            frames: list of BGR uint8 frames at original resolution.

        Returns:
            ball_positions: list of (x, y) in original video coordinates,
                            or (None, None) when no ball is detected.
        """
        n = len(frames)
        h, w = self.cfg.input_height, self.cfg.input_width
        video_h, video_w = frames[0].shape[:2]
        scale_x = video_w / w
        scale_y = video_h / h

        # --- Step 1: pre-resize all frames ---
        t0 = time.perf_counter()
        resized = np.empty((n, h, w, 3), dtype=np.uint8)
        for i in range(n):
            resized[i] = cv2.resize(frames[i], (w, h))
        t_resize = time.perf_counter() - t0
        print(f"  [BallTracker] Resized {n} frames in {t_resize:.1f}s")

        # --- Step 2: batched inference ------------------------------------
        ball_positions = [(None, None)] * 2   # first 2 frames: no data
        num_windows = n - 2
        bs = self.cfg.batch_size
        n_batches = (num_windows + bs - 1) // bs

        t0 = time.perf_counter()
        with torch.no_grad():
            for b_idx in range(n_batches):
                start = b_idx * bs
                end = min(start + bs, num_windows)
                actual_bs = end - start

                # Build 3-frame windows
                frame_indices = np.arange(start + 2, end + 2)
                curr = resized[frame_indices]
                prev = resized[frame_indices - 1]
                pprev = resized[frame_indices - 2]

                windows = np.concatenate([curr, prev, pprev], axis=3)
                windows = np.ascontiguousarray(windows.transpose(0, 3, 1, 2))

                batch_tensor = (
                    torch.from_numpy(windows)
                    .to(self.device, dtype=torch.float32)
                    .div_(255.0)
                )

                with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                    out = self.model(batch_tensor)

                predictions = out.argmax(dim=1).cpu().numpy()

                for j in range(actual_bs):
                    x, y = self._postprocess(predictions[j])
                    if x is not None:
                        x = x * scale_x
                        y = y * scale_y
                    ball_positions.append((x, y))

                if (b_idx + 1) % max(1, n_batches // 5) == 0 or b_idx == n_batches - 1:
                    elapsed = time.perf_counter() - t0
                    print(f"  [BallTracker] Batch {b_idx+1}/{n_batches}  "
                          f"({elapsed:.1f}s elapsed)")

        del resized
        gc.collect()

        t_infer = time.perf_counter() - t0
        det_count = sum(1 for p in ball_positions if p[0] is not None)
        print(f"  [BallTracker] Inference done in {t_infer:.1f}s  |  "
              f"Raw detections: {det_count}/{n} ({100*det_count/n:.1f}%)")

        # --- Step 3: outlier removal --------------------------------------
        ball_positions = self._remove_outliers(ball_positions)
        det_count_clean = sum(1 for p in ball_positions if p[0] is not None)
        print(f"  [BallTracker] After outlier removal: {det_count_clean}/{n} "
              f"({100*det_count_clean/n:.1f}%)")

        return ball_positions

    # ------------------------------------------------------------------
    # Post-processing (per-frame, CPU)
    # ------------------------------------------------------------------
    def _postprocess(self, feature_map: np.ndarray):
        """Extract ball (x, y) in TrackNet coordinate space from argmax output."""
        h, w = self.cfg.input_height, self.cfg.input_width
        fm = (feature_map * 255).reshape((h, w)).astype(np.uint8)
        _, heatmap = cv2.threshold(fm, self.cfg.heatmap_threshold, 255,
                                   cv2.THRESH_BINARY)
        circles = cv2.HoughCircles(
            heatmap, cv2.HOUGH_GRADIENT,
            dp=self.cfg.hough_dp,
            minDist=self.cfg.hough_min_dist,
            param1=self.cfg.hough_param1,
            param2=self.cfg.hough_param2,
            minRadius=self.cfg.hough_min_radius,
            maxRadius=self.cfg.hough_max_radius,
        )
        if circles is not None and len(circles) == 1:
            return float(circles[0][0][0]), float(circles[0][0][1])
        return None, None

    # ------------------------------------------------------------------
    # Outlier removal
    # ------------------------------------------------------------------
    def _remove_outliers(self, ball_track: list[tuple]) -> list[tuple]:
        """
        Remove spatially-inconsistent detections.

        A detection is removed if the distance to both its predecessor
        and successor exceeds the threshold — indicating a false positive
        rather than genuine fast motion.
        """
        max_d = self.cfg.outlier_max_dist
        n = len(ball_track)

        # Compute inter-frame distances
        dists = [None] * n
        for i in range(1, n):
            if ball_track[i][0] is not None and ball_track[i - 1][0] is not None:
                dx = ball_track[i][0] - ball_track[i - 1][0]
                dy = ball_track[i][1] - ball_track[i - 1][1]
                dists[i] = (dx * dx + dy * dy) ** 0.5
            else:
                dists[i] = None

        # Remove outliers: far from predecessor AND far from successor
        cleaned = list(ball_track)
        for i in range(1, n - 1):
            if cleaned[i][0] is None:
                continue
            d_prev = dists[i]
            d_next = dists[i + 1] if i + 1 < n else None

            prev_far = (d_prev is not None and d_prev > max_d) or d_prev is None
            next_far = (d_next is not None and d_next > max_d) or d_next is None

            if prev_far and next_far:
                cleaned[i] = (None, None)

        return cleaned
