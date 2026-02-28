import os
import time

import numpy as np
from ultralytics import YOLO

from .config import YOLOConfig

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class PlayerDetector:
    """YOLO-based player/person detector with batched inference."""

    def __init__(self, config: YOLOConfig, device: str = "cuda"):
        self.cfg = config
        self.device = device
        model_path = os.path.join(_PROJECT_ROOT, self.cfg.model_path)
        self.model = YOLO(model_path)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def detect(self, frames: list[np.ndarray]) -> list[dict]:
        """
        Run player detection on all video frames.

        Args:
            frames: list of BGR uint8 frames at original resolution.

        Returns:
            player_data: per-frame dicts with keys:
                - 'boxes': list of [x1, y1, x2, y2] in video coords
                - 'expanded_boxes': list of [x1, y1, x2, y2] expanded by cfg ratio
                - 'centers': list of [cx, cy]
                - 'count': int
                - 'confidences': list of float
        """
        n = len(frames)
        bs = self.cfg.batch_size
        n_batches = (n + bs - 1) // bs
        all_results = []

        t0 = time.perf_counter()

        for b_idx in range(n_batches):
            start = b_idx * bs
            end = min(start + bs, n)
            batch_frames = frames[start:end]

            results = self.model.predict(
                source=batch_frames,
                imgsz=self.cfg.imgsz,
                conf=self.cfg.conf_threshold,
                classes=[self.cfg.person_class_id],
                half=True,
                max_det=self.cfg.max_det,
                verbose=False,
            )

            for r in results:
                info = {"boxes": [], "expanded_boxes": [],
                        "centers": [], "count": 0,
                        "confidences": []}
                if r.boxes is not None and len(r.boxes) > 0:
                    xyxy = r.boxes.xyxy.cpu().numpy()
                    confs = r.boxes.conf.cpu().numpy()
                    cx = (xyxy[:, 0] + xyxy[:, 2]) / 2.0
                    cy = (xyxy[:, 1] + xyxy[:, 3]) / 2.0
                    info["boxes"] = xyxy.tolist()
                    info["centers"] = np.stack([cx, cy], axis=1).tolist()
                    info["count"] = len(xyxy)
                    info["confidences"] = confs.tolist()

                    # Expanded bboxes for ball-in-bbox gate
                    half_h = self.cfg.bbox_expand_h / 2.0
                    half_v = self.cfg.bbox_expand_v / 2.0
                    widths = xyxy[:, 2] - xyxy[:, 0]
                    heights = xyxy[:, 3] - xyxy[:, 1]
                    expanded = np.stack([
                        xyxy[:, 0] - widths * half_h,
                        xyxy[:, 1] - heights * half_v,
                        xyxy[:, 2] + widths * half_h,
                        xyxy[:, 3] + heights * half_v,
                    ], axis=1)
                    info["expanded_boxes"] = expanded.tolist()
                all_results.append(info)

            if (b_idx + 1) % max(1, n_batches // 5) == 0 or b_idx == n_batches - 1:
                elapsed = time.perf_counter() - t0
                print(f"  [PlayerDetector] Batch {b_idx+1}/{n_batches}  "
                      f"({elapsed:.1f}s elapsed)")

        t_total = time.perf_counter() - t0
        avg_count = np.mean([d["count"] for d in all_results])
        print(f"  [PlayerDetector] Done in {t_total:.1f}s  |  "
              f"Avg players/frame: {avg_count:.1f}")

        return all_results
