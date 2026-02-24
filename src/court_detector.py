"""Tennis court sideline detection for ROI filtering.

Detects the two *sidelines* (the long court boundaries that run roughly
top-to-bottom in the frame) by:

    1.  Running Canny edge detection on the grayscale frame.
    2.  Detecting steep (non-horizontal) line segments via HoughLinesP
        — these are sideline candidates.
    3.  Splitting candidates into left / right groups by midpoint x.
    4.  Fitting each group with ``cv2.fitLine`` and extending to the
        full frame height.
    5.  Shifting each fitted sideline outward by a configurable number
        of pixels (perpendicular to the line direction) to create
        padded boundary lines.
    6.  Building a binary mask from the trapezoid formed by the padded
        lines.

Designed for fixed-camera setups.  Runs once on the first frame — cost
is negligible compared to per-frame neural-network inference.
"""

from __future__ import annotations

import cv2
import numpy as np

from config import CourtDetectorConfig


class CourtDetector:
    """Detects the tennis court sidelines and builds an ROI mask.

    Public workflow
    ---------------
    1. ``detect(frame)``  — find the sidelines in a single BGR frame.
    2. ``filter_ball_positions(…)`` / ``filter_player_data_list(…)``
       — keep only on-court detections.
    3. ``set_padding(px)`` — adjust padding without re-running detection.
    """

    def __init__(self, config: CourtDetectorConfig) -> None:
        self.cfg = config

        # Set by detect()
        self.court_polygon: np.ndarray | None = None    # 4×2 raw sideline trapezoid
        self.padded_polygon: np.ndarray | None = None   # 4×2 padded trapezoid
        self.court_mask: np.ndarray | None = None       # H×W uint8 (255 inside)

        # Raw sideline data: (p_top, p_bot, vx, vy) per side
        self._left_sideline: tuple | None = None
        self._right_sideline: tuple | None = None

        self._frame_h: int = 0
        self._frame_w: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def detected(self) -> bool:
        """True if a valid court region has been established."""
        return self.court_mask is not None

    # ------------------------------------------------------------------
    # Main detection entry point
    # ------------------------------------------------------------------
    def detect(self, frame: np.ndarray) -> np.ndarray | None:
        """Detect the court sidelines from a single BGR frame.

        Populates ``court_polygon``, ``padded_polygon``, ``court_mask``,
        and the internal sideline data on success.

        Returns
        -------
        np.ndarray | None
            Padded polygon (4×2 float64) or *None* on failure.
        """
        self._frame_h, self._frame_w = frame.shape[:2]
        h, w = self._frame_h, self._frame_w

        # Step 1 — edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, self.cfg.canny_low, self.cfg.canny_high)

        # Step 2 — detect steep (sideline-like) segments
        raw = cv2.HoughLinesP(
            edges, 1, np.pi / 180,
            threshold=self.cfg.hough_threshold,
            minLineLength=self.cfg.min_line_length,
            maxLineGap=self.cfg.max_line_gap,
        )
        if raw is None:
            print("  [CourtDetector] No lines detected — skipping")
            return None

        lines = raw.reshape(-1, 4).astype(np.float64)
        angles = np.degrees(np.arctan2(
            np.abs(lines[:, 3] - lines[:, 1]),
            np.abs(lines[:, 2] - lines[:, 0]),
        ))
        steep_mask = angles >= self.cfg.steep_min_angle
        steep = lines[steep_mask]

        if len(steep) < 2:
            print(f"  [CourtDetector] Only {len(steep)} steep line(s) "
                  f"(need ≥2) — skipping")
            return None
        print(f"  [CourtDetector] Steep lines: {len(steep)} "
              f"(from {len(lines)} total)")

        # Step 3 — split into left / right groups by midpoint x
        mid_x = (steep[:, 0] + steep[:, 2]) / 2.0
        left_mask = mid_x < w / 2
        right_mask = ~left_mask

        if left_mask.sum() == 0 or right_mask.sum() == 0:
            print("  [CourtDetector] Lines only on one side of frame "
                  "— skipping")
            return None

        # Step 4 — fit and extend each group
        left_top, left_bot, lvx, lvy = self._fit_and_extend(steep[left_mask])
        right_top, right_bot, rvx, rvy = self._fit_and_extend(steep[right_mask])

        # Ensure left is actually left (smaller x at mid-frame)
        left_mid = (left_top[0] + left_bot[0]) / 2.0
        right_mid = (right_top[0] + right_bot[0]) / 2.0
        if left_mid > right_mid:
            left_top, right_top = right_top, left_top
            left_bot, right_bot = right_bot, left_bot
            lvx, rvx = rvx, lvx
            lvy, rvy = rvy, lvy

        self._left_sideline = (left_top, left_bot, lvx, lvy)
        self._right_sideline = (right_top, right_bot, rvx, rvy)

        # Raw sideline trapezoid (no padding)
        self.court_polygon = np.array([
            [left_top[0], left_top[1]],
            [right_top[0], right_top[1]],
            [right_bot[0], right_bot[1]],
            [left_bot[0], left_bot[1]],
        ], dtype=np.float64)

        print(f"  [CourtDetector] Left sideline:  "
              f"({left_top[0]:.0f},{left_top[1]:.0f}) → "
              f"({left_bot[0]:.0f},{left_bot[1]:.0f})")
        print(f"  [CourtDetector] Right sideline: "
              f"({right_top[0]:.0f},{right_top[1]:.0f}) → "
              f"({right_bot[0]:.0f},{right_bot[1]:.0f})")

        # Step 5 — apply padding and build mask
        self._rebuild_mask()

        # Validate area
        mask_area = cv2.countNonZero(self.court_mask)
        frac = mask_area / (h * w)
        if frac < self.cfg.min_court_area_pct:
            print(f"  [CourtDetector] ROI too small ({frac*100:.1f}%) "
                  f"— skipping")
            self.court_mask = None
            return None
        if frac > self.cfg.max_court_area_pct:
            print(f"  [CourtDetector] ROI too large ({frac*100:.1f}%) "
                  f"— skipping")
            self.court_mask = None
            return None

        print(f"  [CourtDetector] Court ROI: {frac*100:.1f}% of frame "
              f"(padding {self.cfg.padding_px}px)")

        return self.padded_polygon

    # ------------------------------------------------------------------
    # Padding management
    # ------------------------------------------------------------------
    def set_padding(self, padding_px: int) -> np.ndarray | None:
        """Change padding and rebuild mask without re-running detection."""
        if self._left_sideline is None:
            return None
        self.cfg.padding_px = padding_px
        self._rebuild_mask()
        return self.padded_polygon

    # ------------------------------------------------------------------
    # Filtering API
    # ------------------------------------------------------------------
    def is_point_in_court(self, x: float, y: float) -> bool:
        """Check whether *(x, y)* falls inside the padded court mask."""
        if self.court_mask is None:
            return True  # no mask → pass everything
        ix, iy = int(round(x)), int(round(y))
        if 0 <= iy < self._frame_h and 0 <= ix < self._frame_w:
            return self.court_mask[iy, ix] > 0
        return False

    def filter_ball_positions(
        self, ball_positions: list[tuple],
    ) -> list[tuple]:
        """Return a copy with off-court positions replaced by (None, None)."""
        if not self.detected:
            return ball_positions
        out: list[tuple] = []
        for pos in ball_positions:
            if pos[0] is None:
                out.append(pos)
            elif self.is_point_in_court(pos[0], pos[1]):
                out.append(pos)
            else:
                out.append((None, None))
        return out

    def filter_player_data(self, player_info: dict) -> dict:
        """Keep only players whose feet (box bottom-centre) are on court."""
        if not self.detected:
            return player_info

        filtered: dict = {
            "boxes": [],
            "expanded_boxes": [],
            "centers": [],
            "count": 0,
            "confidences": [],
        }
        for i, box in enumerate(player_info.get("boxes", [])):
            x1, y1, x2, y2 = box
            foot_x = (x1 + x2) / 2.0
            foot_y = float(y2)
            if self.is_point_in_court(foot_x, foot_y):
                filtered["boxes"].append(box)
                if i < len(player_info.get("expanded_boxes", [])):
                    filtered["expanded_boxes"].append(
                        player_info["expanded_boxes"][i]
                    )
                if i < len(player_info.get("centers", [])):
                    filtered["centers"].append(player_info["centers"][i])
                if i < len(player_info.get("confidences", [])):
                    filtered["confidences"].append(
                        player_info["confidences"][i]
                    )
        filtered["count"] = len(filtered["boxes"])
        return filtered

    def filter_player_data_list(
        self, player_data: list[dict],
    ) -> list[dict]:
        """Filter a list of per-frame player-data dicts."""
        if not self.detected:
            return player_data
        return [self.filter_player_data(d) for d in player_data]

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _fit_and_extend(
        self, group_lines: np.ndarray,
    ) -> tuple[tuple[float, float], tuple[float, float], float, float]:
        """Fit a line through all endpoints in a group and extend to frame edges.

        Returns
        -------
        p_top : (x, y) at y=0
        p_bot : (x, y) at y=frame_height
        vx, vy : unit direction vector (vy > 0, pointing downward)
        """
        pts = np.vstack([
            group_lines[:, :2],
            group_lines[:, 2:4],
        ]).astype(np.float32)
        vx, vy, x0, y0 = cv2.fitLine(
            pts, cv2.DIST_L2, 0, 0.01, 0.01,
        ).flatten()
        # Ensure vy > 0 (direction goes top to bottom)
        if vy < 0:
            vx, vy = -vx, -vy
        t_top = (0.0 - y0) / vy
        t_bot = (float(self._frame_h) - y0) / vy
        p_top = (float(x0 + vx * t_top), 0.0)
        p_bot = (float(x0 + vx * t_bot), float(self._frame_h))
        return p_top, p_bot, float(vx), float(vy)

    @staticmethod
    def _shift_line(
        p_top: tuple[float, float],
        p_bot: tuple[float, float],
        vx: float,
        vy: float,
        px_outward: float,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Shift a line perpendicular to its direction by *px_outward* pixels."""
        # Normal perpendicular to direction vector
        nx, ny = -vy, vx
        norm = (nx * nx + ny * ny) ** 0.5
        nx, ny = nx / norm, ny / norm
        return (
            (p_top[0] + nx * px_outward, p_top[1]),
            (p_bot[0] + nx * px_outward, p_bot[1]),
        )

    def _rebuild_mask(self) -> None:
        """(Re)compute ``padded_polygon`` and ``court_mask``.

        Shifts each sideline outward by ``padding_px`` pixels
        perpendicular to its direction, then fills the resulting
        trapezoid as a binary mask.
        """
        left_top, left_bot, lvx, lvy = self._left_sideline
        right_top, right_bot, rvx, rvy = self._right_sideline
        pad = self.cfg.padding_px

        # Shift left sideline outward (left = negative x direction)
        lp_top, lp_bot = self._shift_line(left_top, left_bot, lvx, lvy, -pad)
        # If it moved inward instead of outward, flip
        if lp_top[0] > left_top[0]:
            lp_top, lp_bot = self._shift_line(
                left_top, left_bot, lvx, lvy, pad,
            )

        # Shift right sideline outward (right = positive x direction)
        rp_top, rp_bot = self._shift_line(right_top, right_bot, rvx, rvy, pad)
        # If it moved inward instead of outward, flip
        if rp_top[0] < right_top[0]:
            rp_top, rp_bot = self._shift_line(
                right_top, right_bot, rvx, rvy, -pad,
            )

        # Build padded trapezoid: TL, TR, BR, BL
        self.padded_polygon = np.array([
            [lp_top[0], lp_top[1]],
            [rp_top[0], rp_top[1]],
            [rp_bot[0], rp_bot[1]],
            [lp_bot[0], lp_bot[1]],
        ], dtype=np.float64)

        # Fill trapezoid → binary mask
        mask = np.zeros((self._frame_h, self._frame_w), dtype=np.uint8)
        pts = self.padded_polygon.astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)
        self.court_mask = mask
