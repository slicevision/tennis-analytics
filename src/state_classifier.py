import numpy as np
from scipy.ndimage import uniform_filter1d, gaussian_filter1d

from config import ClassifierConfig, BounceConfig


class StateClassifier:
    """Classifies each video frame as play (1) or dead (0)."""

    def __init__(self, config: ClassifierConfig, fps: float,
                 bounce_config: BounceConfig | None = None):
        self.cfg = config
        self.fps = fps
        self.bounce_cfg = bounce_config

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def classify(
        self,
        ball_positions: list[tuple],
        player_data: list[dict],
        bounce_analysis: dict | None = None,
    ) -> dict:
        """
        Classify every frame as play or dead.

        Returns dict with:
            states        – np.ndarray[int], 0=dead 1=play, len=n_frames
            play_scores   – np.ndarray[float] in [0,1] (smoothed)
            raw_scores    – np.ndarray[float] in [0,1] (before smoothing)
            features      – dict of intermediate per-frame arrays
            segments      – list of {type, start_frame, end_frame, start_sec, ...}
            summary       – dict with play_time, dead_time, etc.
        """
        n = len(ball_positions)

        # ------ Step 1: per-frame raw signals ----------------------------
        detected = np.array(
            [1.0 if p[0] is not None else 0.0 for p in ball_positions]
        )

        speeds = np.zeros(n)
        for i in range(1, n):
            if ball_positions[i][0] is not None and ball_positions[i-1][0] is not None:
                dx = ball_positions[i][0] - ball_positions[i-1][0]
                dy = ball_positions[i][1] - ball_positions[i-1][1]
                speeds[i] = np.sqrt(dx * dx + dy * dy)

        player_counts = np.array(
            [d["count"] for d in player_data], dtype=np.float64
        )

        # ------ Step 2: windowed features --------------------------------
        w = self.cfg.window_size
        detection_rate = uniform_filter1d(detected, size=w, mode="nearest")

        speed_smooth = uniform_filter1d(speeds, size=w, mode="nearest")

        player_smooth = uniform_filter1d(player_counts, size=w, mode="nearest")

        # ------ Step 2b: dribble score from bounce analysis -----
        dribble_score = None
        if bounce_analysis is not None:
            dribble_score = bounce_analysis.get("dribble_score")

        # ------ Step 3: weighted scoring ---------------------------------
        raw_scores = self._compute_play_score(
            detection_rate, speed_smooth, player_smooth, dribble_score
        )

        # ------ Step 4: Gaussian temporal smoothing ----------------------
        play_scores = gaussian_filter1d(raw_scores, sigma=self.cfg.smooth_sigma)

        # ------ Step 5: hysteresis thresholding --------------------------

        expanded_boxes = [d.get("expanded_boxes", []) for d in player_data]
        states, gate_suppressions = self._hysteresis(
            play_scores, ball_positions, expanded_boxes
        )

        # ------ Step 6: minimum-duration filtering -----------------------
        min_play_frames = int(self.cfg.min_play_duration * self.fps)
        min_dead_frames = int(self.cfg.min_dead_duration * self.fps)
        states = self._filter_short_segments(states, min_play_frames,
                                             min_dead_frames)

        # ------ Step 6b: isolated play filter ----------------------------
        isolated_kills = 0
        if self.cfg.use_isolated_play_filter:
            states, isolated_kills = self._filter_isolated_play(states)

        # ------ Build output metadata ------------------------------------
        segments = self._extract_segments(states)
        summary = self._build_summary(states, segments)

        features = {
            "detected": detected,
            "speeds": speeds,
            "player_counts": player_counts,
            "detection_rate": detection_rate,
            "speed_smooth": speed_smooth,
            "player_smooth": player_smooth,
            "dribble_score": dribble_score,
        }

        print(f"  [Classifier] Ball detected: "
              f"{int(detected.sum())}/{n} frames "
              f"({100*detected.mean():.1f}%)")
        print(f"  [Classifier] Avg ball speed (when detected): "
              f"{speeds[speeds > 0].mean():.1f} px/frame" if speeds.any()
              else "  [Classifier] No ball speed data")
        if dribble_score is not None:
            drib_pct = 100 * np.mean(dribble_score > 0.1)
            print(f"  [Classifier] Dribble signal active in "
                  f"{drib_pct:.1f}% of frames")
        if self.cfg.use_bbox_gate:
            print(f"  [Classifier] Bbox gate suppressed {gate_suppressions} "
                  f"dead\u2192play transitions")
        if self.cfg.use_isolated_play_filter:
            print(f"  [Classifier] Isolated play filter removed "
                  f"{isolated_kills} segment(s)")
        print(f"  [Classifier] Play: {summary['play_time']:.1f}s "
              f"({summary['play_pct']:.1f}%)  |  "
              f"Dead: {summary['dead_time']:.1f}s "
              f"({summary['dead_pct']:.1f}%)")
        print(f"  [Classifier] Segments: {len(segments)} "
              f"({sum(1 for s in segments if s['type']=='play')} play, "
              f"{sum(1 for s in segments if s['type']=='dead')} dead)")

        return {
            "states": states,
            "play_scores": play_scores,
            "features": features,
            "segments": segments,
            "summary": summary,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _compute_play_score(
        self,
        detection_rate: np.ndarray,
        speed_smooth: np.ndarray,
        player_smooth: np.ndarray,
        dribble_score: np.ndarray | None = None,
    ) -> np.ndarray:
        """Weighted combination of normalised signals → play score in [0, 1]."""
        c = self.cfg

        det_norm = np.clip(detection_rate / c.detection_rate_cap, 0.0, 1.0)
        spd_norm = np.clip(speed_smooth / c.ball_speed_cap, 0.0, 1.0)
        plr_norm = np.clip(player_smooth / c.player_count_cap, 0.0, 1.0)

        score = (c.weight_detection_rate * det_norm
                 + c.weight_ball_speed * spd_norm
                 + c.weight_player_count * plr_norm)

        # Apply dribble penalty
        if dribble_score is not None and self.bounce_cfg is not None:
            penalty = self.bounce_cfg.weight_dribble * np.clip(
                dribble_score, 0.0, 1.0
            )
            score = np.clip(score - penalty, 0.0, 1.0)

        return score

    def _hysteresis(
        self,
        scores: np.ndarray,
        ball_positions: list[tuple] = None,
        expanded_boxes: list[list] = None,
    ) -> tuple[np.ndarray, int]:
        """
        Hysteresis thresholding with optional ball-in-bbox gate.

        Returns:
            states – np.ndarray[int32], 0=dead 1=play
            gate_suppressions – count of suppressed dead→play transitions
        """
        n = len(scores)
        states = np.zeros(n, dtype=np.int32)
        current = 0  # start in DEAD state
        enter = self.cfg.enter_play_threshold
        leave = self.cfg.exit_play_threshold

        use_gate = (
            self.cfg.use_bbox_gate
            and ball_positions is not None
            and expanded_boxes is not None
        )
        gate_suppressions = 0

        for i in range(n):
            if current == 0 and scores[i] >= enter:
                if use_gate and self._ball_inside_bbox_recently(
                    ball_positions, expanded_boxes, i
                ):
                    current = 0
                    gate_suppressions += 1
                else:
                    current = 1
            elif current == 1 and scores[i] < leave:
                current = 0
            states[i] = current
        return states, gate_suppressions

    # ------------------------------------------------------------------
    # Ball-in-bbox helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _ball_inside_any_bbox(
        ball_pos: tuple,
        frame_expanded_boxes: list,
    ) -> bool:
        """Check if ball position falls inside any expanded player bbox."""
        if ball_pos[0] is None:
            return False
        bx, by = ball_pos
        for box in frame_expanded_boxes:
            x1, y1, x2, y2 = box
            if x1 <= bx <= x2 and y1 <= by <= y2:
                return True
        return False

    def _ball_inside_bbox_recently(
        self,
        ball_positions: list[tuple],
        expanded_boxes: list[list],
        frame_idx: int,
    ) -> bool:
        """Check if ball was inside any player bbox for most of the lookback window."""
        lookback = self.cfg.bbox_gate_lookback
        start = max(0, frame_idx - lookback + 1)
        inside = 0
        checked = 0

        for j in range(start, frame_idx + 1):
            if ball_positions[j][0] is not None:
                checked += 1
                if self._ball_inside_any_bbox(
                    ball_positions[j], expanded_boxes[j]
                ):
                    inside += 1

        if checked == 0:
            return False

        return (inside / checked) > self.cfg.bbox_gate_ratio

    @staticmethod
    def _filter_short_segments(
        states: np.ndarray,
        min_play: int,
        min_dead: int,
    ) -> np.ndarray:
        """Remove segments shorter than the minimum duration."""
        result = states.copy()
        for _ in range(3):
            changed = False
            # Find contiguous runs
            diffs = np.diff(result, prepend=-1)
            starts = np.where(diffs != 0)[0]
            ends = np.append(starts[1:], len(result))

            for s, e in zip(starts, ends):
                length = e - s
                state_val = result[s]
                if state_val == 1 and length < min_play:
                    result[s:e] = 0
                    changed = True
                elif state_val == 0 and length < min_dead:
                    result[s:e] = 1
                    changed = True
            if not changed:
                break
        return result

    def _filter_isolated_play(
        self, states: np.ndarray
    ) -> tuple[np.ndarray, int]:
        """
        Remove short play segments isolated between long dead time.

        Returns:
            states       – modified array
            killed_count – number of segments eliminated
        """
        result = states.copy()
        max_play_frames = int(self.cfg.isolated_max_play * self.fps)
        ratio = self.cfg.isolated_dead_ratio

        # Extract contiguous runs
        diffs = np.diff(result, prepend=-1)
        starts = np.where(diffs != 0)[0]
        ends = np.append(starts[1:], len(result))
        run_states = result[starts]
        run_lengths = ends - starts
        n_runs = len(run_states)

        killed = 0
        for i in range(n_runs):
            if run_states[i] != 1:
                continue

            play_len = run_lengths[i]
            if play_len >= max_play_frames:
                continue

            # Need a dead segment on both sides
            if i == 0 or i == n_runs - 1:
                continue
            if run_states[i - 1] != 0 or run_states[i + 1] != 0:
                continue

            dead_before = run_lengths[i - 1]
            dead_after = run_lengths[i + 1]

            if play_len < ratio * (dead_before + dead_after):
                result[starts[i]:ends[i]] = 0
                killed += 1

        return result, killed

    def _extract_segments(self, states: np.ndarray) -> list[dict]:
        """Convert frame-level states into a list of timed segments."""
        segments = []
        diffs = np.diff(states, prepend=-1)
        starts = np.where(diffs != 0)[0]
        ends = np.append(starts[1:], len(states))

        for s, e in zip(starts, ends):
            seg_type = "play" if states[s] == 1 else "dead"
            segments.append({
                "type": seg_type,
                "start_frame": int(s),
                "end_frame": int(e),
                "start_sec": round(s / self.fps, 2),
                "end_sec": round(e / self.fps, 2),
                "duration_sec": round((e - s) / self.fps, 2),
            })
        return segments

    def _build_summary(self, states, segments) -> dict:
        n = len(states)
        total_sec = n / self.fps
        play_frames = int(states.sum())
        dead_frames = n - play_frames
        play_sec = play_frames / self.fps
        dead_sec = dead_frames / self.fps
        return {
            "total_frames": n,
            "total_duration": round(total_sec, 2),
            "play_frames": play_frames,
            "dead_frames": dead_frames,
            "play_time": round(play_sec, 2),
            "dead_time": round(dead_sec, 2),
            "play_pct": round(100 * play_sec / total_sec, 1) if total_sec else 0,
            "dead_pct": round(100 * dead_sec / total_sec, 1) if total_sec else 0,
            "num_segments": len(segments),
            "num_play_segments": sum(1 for s in segments if s["type"] == "play"),
            "num_dead_segments": sum(1 for s in segments if s["type"] == "dead"),
        }
