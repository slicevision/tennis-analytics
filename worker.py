#!/usr/bin/env python3
"""
SliceVision Analytics GPU Worker (basic tier only)
=================================================
Pulls jobs from Supabase Edge Functions, downloads video from S3,
runs analytics, uploads timeline.json (or equivalent).

Production contract (delivery-gated credits):
- After analytics JSON is uploaded: mark job status = analytics_done (credit still reserved).
- Only after highlight.mp4 is uploaded: mark delivered (credit captured).
- If highlight fails: mark delivery failed (credit released/refunded via Supabase RPC).

Security:
- No Supabase service-role key on worker.
- Worker calls Edge Functions using anon + x-worker-key.
- S3 access via EC2 instance role (no static AWS keys).

On-demand:
- If no jobs for IDLE_GRACE_MINUTES, stop the instance (ENABLE_SELF_STOP=true)
  or exit (ENABLE_SELF_STOP=false).
"""

import glob
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Optional, List, Tuple
import shlex
import boto3
import requests as http_requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
log = logging.getLogger("analytics-worker")

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "")
ANALYTICS_WORKER_KEY = os.environ.get("ANALYTICS_WORKER_KEY", "")
WORKER_ID = os.environ.get("WORKER_ID", platform.node())[:64]

AWS_REGION = os.environ.get("AWS_REGION", "eu-south-1")

ANALYTICS_CMD_BASIC = os.environ.get("ANALYTICS_CMD_BASIC", "")
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "10"))
HEARTBEAT_INTERVAL = int(os.environ.get("HEARTBEAT_INTERVAL", "60"))

RESET_STALE_ON_START = os.environ.get("RESET_STALE_ON_START", "true").lower() in (
    "true",
    "1",
    "yes",
)
RESET_STALE_MAX_AGE_MINUTES = int(os.environ.get("RESET_STALE_MAX_AGE_MINUTES", "360"))

IDLE_GRACE_MINUTES = int(os.environ.get("IDLE_GRACE_MINUTES", "10"))
ENABLE_SELF_STOP = os.environ.get("ENABLE_SELF_STOP", "true").lower() in ("true", "1", "yes")

REPO_DIR = os.environ.get("REPO_DIR", "/opt/analytics_repo")

FUNCTIONS_BASE = f"{SUPABASE_URL}/functions/v1"

# ── Highlight cutter (delivery-gated) ─────────────────────────────────────────
ENABLE_HIGHLIGHT = os.environ.get("ENABLE_HIGHLIGHT", "true").lower() in ("true", "1", "yes")
HIGHLIGHT_MIN_SEG_SECONDS = float(os.environ.get("HIGHLIGHT_MIN_SEG_SECONDS", "2.0"))
HIGHLIGHT_MAX_TOTAL_SECONDS = float(os.environ.get("HIGHLIGHT_MAX_TOTAL_SECONDS", "1800"))  # 30 min cap
KEEP_WORK_DIR = False

def _call_fn(path: str, body: dict, timeout: int = 30) -> http_requests.Response:
    url = f"{FUNCTIONS_BASE}/{path}"
    headers = {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
        "Content-Type": "application/json",
        "x-worker-key": ANALYTICS_WORKER_KEY,
    }
    return http_requests.post(url, json=body, headers=headers, timeout=timeout)

def _safe_json(resp: http_requests.Response):
    ctype = (resp.headers.get("content-type") or "").lower()
    if "application/json" not in ctype:
        return None
    try:
        return resp.json()
    except Exception:
        return None

def _log_http_error(prefix: str, resp: http_requests.Response, max_body: int = 500):
    ctype = (resp.headers.get("content-type") or "").lower()
    body = (resp.text or "")[:max_body].replace("\n", " ").replace("\r", " ")
    log.warning("%s http=%d ctype=%s body=%s", prefix, resp.status_code, ctype, body)
def _s3_client():
    return boto3.client("s3", region_name=AWS_REGION)


def _imds_token(timeout: int = 2) -> str:
    r = http_requests.put(
        "http://169.254.169.254/latest/api/token",
        headers={"X-aws-ec2-metadata-token-ttl-seconds": "21600"},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.text


def _imds_get(path: str, token: str, timeout: int = 2) -> str:
    r = http_requests.get(
        f"http://169.254.169.254/latest/{path}",
        headers={"X-aws-ec2-metadata-token": token},
        timeout=timeout,
    )
    r.raise_for_status()
    return r.text


def stop_self():
    if not ENABLE_SELF_STOP:
        log.info("Self-stop disabled; exiting worker.")
        sys.exit(0)

    try:
        token = _imds_token()
        instance_id = _imds_get("meta-data/instance-id", token).strip()
        ident_doc = _imds_get("dynamic/instance-identity/document", token).strip()
        region = json.loads(ident_doc).get("region") or AWS_REGION

        log.info("Stopping self instance_id=%s region=%s", instance_id, region)
        ec2 = boto3.client("ec2", region_name=region)
        ec2.stop_instances(InstanceIds=[instance_id])
    except Exception as exc:
        log.error("Self-stop failed: %s", exc)

    sys.exit(0)


class HeartbeatThread(threading.Thread):
    def __init__(self, job_id: str, worker_id: str, interval: int):
        super().__init__(daemon=True)
        self.job_id = job_id
        self.worker_id = worker_id
        self.interval = interval
        self._stop_event = threading.Event()

    def run(self):
        while not self._stop_event.wait(self.interval):
            try:
                resp = _call_fn(
                    "analytics-heartbeat-job",
                    {"job_id": self.job_id, "worker_id": self.worker_id},
                )
                if resp.status_code != 200:
                    log.warning("Heartbeat returned %d for job %s", resp.status_code, self.job_id)
            except Exception as exc:
                log.warning("Heartbeat error for job %s: %s", self.job_id, exc)

    def stop(self):
        self._stop_event.set()


def reset_stale_jobs():
    if not RESET_STALE_ON_START:
        log.info("Stale reset disabled (RESET_STALE_ON_START != true)")
        return

    log.info("Resetting stale jobs (max_age_minutes=%d)", RESET_STALE_MAX_AGE_MINUTES)
    try:
        resp = _call_fn(
            "analytics-reset-stale-jobs",
            {"max_age_minutes": RESET_STALE_MAX_AGE_MINUTES},
        )
        if resp.status_code == 200:
            log.info("Stale reset result: %s", resp.json())
        else:
            log.warning("Stale reset returned %d: %s", resp.status_code, resp.text)
    except Exception as exc:
        log.warning("Stale reset failed (best-effort): %s", exc)


def complete_job(job_id: str, status: str, error: Optional[str] = None):
    body: dict = {"job_id": job_id, "worker_id": WORKER_ID, "status": status}
    if error:
        body["error"] = error[:2000]
    resp = _call_fn("analytics-complete-job", body, timeout=30)
    log.info("Complete job %s status=%s http=%d", job_id, status, resp.status_code)
    if resp.status_code not in (200, 409):
        raise RuntimeError(f"analytics-complete-job unexpected {resp.status_code}: {resp.text}")


def mark_delivered(job_id: str, highlight_s3_key: str):
    body = {"job_id": job_id, "worker_id": WORKER_ID, "highlight_s3_key": highlight_s3_key}
    resp = _call_fn("analytics-mark-delivered", body, timeout=30)
    log.info("Mark delivered job %s http=%d", job_id, resp.status_code)
    if resp.status_code not in (200, 409):
        raise RuntimeError(f"analytics-mark-delivered unexpected {resp.status_code}: {resp.text}")


def mark_delivery_failed(job_id: str, error: str):
    body = {"job_id": job_id, "worker_id": WORKER_ID, "error": (error or "")[:2000]}
    resp = _call_fn("analytics-mark-delivery-failed", body, timeout=30)
    log.info("Mark delivery failed job %s http=%d", job_id, resp.status_code)
    if resp.status_code not in (200, 409):
        raise RuntimeError(f"analytics-mark-delivery-failed unexpected {resp.status_code}: {resp.text}")


def _derive_output_timeline_key(job: dict) -> Optional[str]:
    prefix = job.get("output_s3_prefix")
    if prefix:
        if not prefix.endswith("/"):
            prefix += "/"
        return f"{prefix}timeline.json"
    return job.get("output_timeline_key")


def _derive_output_highlight_key(job: dict) -> Optional[str]:
    prefix = job.get("output_s3_prefix")
    if prefix:
        if not prefix.endswith("/"):
            prefix += "/"
        return f"{prefix}highlight.mp4"
    return job.get("output_highlight_key")


def _choose_output_json(out_dir: str) -> Tuple[str, List[str]]:
    """
    Return (path_to_upload, missing_candidates).

    Prefers:
      1) timeline.json
      2) input_video_report.json
    Fallback:
      - any *.json in out_dir (largest file wins)
    """
    candidates = [
        ("timeline.json", os.path.join(out_dir, "timeline.json")),
        ("input_video_report.json", os.path.join(out_dir, "input_video_report.json")),
    ]

    missing: List[str] = []
    for name, path in candidates:
        if os.path.isfile(path):
            return path, missing
        missing.append(name)

    # Fallback: pick any json in out_dir (largest wins)
    try:
        json_files = []
        for fn in os.listdir(out_dir):
            if fn.lower().endswith(".json"):
                p = os.path.join(out_dir, fn)
                if os.path.isfile(p):
                    json_files.append(p)

        if json_files:
            json_files.sort(key=lambda p: os.path.getsize(p), reverse=True)
            chosen = json_files[0]
            log.warning("Fallback JSON chosen: %s (expected missing: %s)", chosen, ", ".join(missing))
            return chosen, missing

        # Debug: show out_dir contents
        log.warning("No JSON outputs found in out_dir=%s. Contents:\n%s",
                    out_dir,
                    subprocess.getoutput(f"ls -lah {out_dir}"))
    except Exception as exc:
        log.warning("Failed listing out_dir=%s: %s", out_dir, exc)

    # None exist; return canonical timeline.json path for placeholder creation
    return os.path.join(out_dir, "timeline.json"), missing


def _run(cmd: List[str], cwd: Optional[str] = None, timeout: int = 7200):
    log.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=cwd, timeout=timeout)


def _extract_play_segments(json_path: str) -> List[Tuple[float, float]]:
    """
    Returns list of (start_sec, end_sec) for segments where type == "play".
    Expects schema: {"segments":[{"type":"play","start_sec":...,"end_sec":...}, ...]}
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    segs = data.get("segments") or []
    out: List[Tuple[float, float]] = []
    total = 0.0

    for s in segs:
        if (s.get("type") or "").lower() != "play":
            continue
        try:
            a = float(s["start_sec"])
            b = float(s["end_sec"])
        except Exception:
            continue
        if b <= a:
            continue
        if (b - a) < HIGHLIGHT_MIN_SEG_SECONDS:
            continue

        dur = b - a
        if total + dur > HIGHLIGHT_MAX_TOTAL_SECONDS:
            remaining = HIGHLIGHT_MAX_TOTAL_SECONDS - total
            if remaining <= 0:
                break
            b = a + remaining
            dur = b - a

        out.append((a, b))
        total += dur

        if total >= HIGHLIGHT_MAX_TOTAL_SECONDS:
            break

    return out


def build_highlight(input_video: str, report_json: str, work_dir: str) -> Optional[str]:
    """
    Creates highlight.mp4 by clipping each play segment and concatenating them.
    Final encode uses NVENC + AAC and faststart.
    Returns output path or None if no play segments.
    """
    segs = _extract_play_segments(report_json)
    if not segs:
        return None

    clips_dir = os.path.join(work_dir, "hl_clips")
    os.makedirs(clips_dir, exist_ok=True)

    tmp_clips: List[str] = []
    for i, (a, b) in enumerate(segs):
        out = os.path.join(clips_dir, f"seg_{i:03d}.mp4")
        dur = max(b - a, 0.01)

        # Try stream copy first (fast). If it fails, fallback to re-encode that segment.
        try:
            _run(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    f"{a:.3f}",
                    "-i",
                    input_video,
                    "-t",
                    f"{dur:.3f}",
                    "-map",
                    "0",
                    "-c",
                    "copy",
                    out,
                ]
            )
        except Exception:
            _run(
                [
                    "ffmpeg",
                    "-y",
                    "-ss",
                    f"{a:.3f}",
                    "-i",
                    input_video,
                    "-t",
                    f"{dur:.3f}",
                    "-map",
                    "0",
                    "-c:v",
                    "h264_nvenc",
                    "-preset",
                    "p4",
                    "-rc",
                    "vbr",
                    "-cq",
                    "19",
                    "-pix_fmt",
                    "yuv420p",
                    "-color_range",
                    "tv",
                    "-colorspace",
                    "bt709",
                    "-color_primaries",
                    "bt709",
                    "-color_trc",
                    "bt709",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "128k",
                    out,
                ]
            )

        tmp_clips.append(out)

    if not tmp_clips:
        return None

    concat_list = os.path.join(work_dir, "hl_concat.txt")
    with open(concat_list, "w") as f:
        for p in tmp_clips:
            f.write(f"file '{Path(p).as_posix()}'\n")

    highlight_path = os.path.join(work_dir, "highlight.mp4")

    # Concatenate + encode once (NVENC) for consistent output; preserve audio; faststart for web playback
    _run(
        [
            "ffmpeg",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_list,
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p4",
            "-rc",
            "vbr",
            "-cq",
            "19",
            "-pix_fmt",
            "yuv420p",
            "-color_range",
            "tv",
            "-colorspace",
            "bt709",
            "-color_primaries",
            "bt709",
            "-color_trc",
            "bt709",
            "-c:a",
            "aac",
            "-b:a",
            "128k",
            "-movflags",
            "faststart",
            highlight_path,
        ]
    )

    return highlight_path


def process_job(job: dict):
    job_id = job["id"]
    tier = job.get("tier", "basic")
    if tier != "basic":
        complete_job(job_id, "failed", "advanced tier not supported yet")
        return

    source_bucket = job["source_s3_bucket"]
    source_key = job["source_s3_key"]
    output_timeline_key = _derive_output_timeline_key(job)
    output_highlight_key = _derive_output_highlight_key(job) if ENABLE_HIGHLIGHT else None

    if not output_timeline_key:
        complete_job(job_id, "failed", "missing output_s3_prefix/output_timeline_key")
        return

    log.info("Processing job %s tier=%s", job_id, tier)

    heartbeat = HeartbeatThread(job_id, WORKER_ID, HEARTBEAT_INTERVAL)
    heartbeat.start()

    work_dir = tempfile.mkdtemp(prefix=f"job_{job_id}_")
    input_path = os.path.join(work_dir, "input_video.mp4")
    out_dir = os.path.join(work_dir, "output")
    os.makedirs(out_dir, exist_ok=True)

    s3 = _s3_client()

    try:
        log.info("Downloading s3://%s/%s", source_bucket, source_key)
        s3.download_file(source_bucket, source_key, input_path)

        template = shlex.split(ANALYTICS_CMD_BASIC)
        cmd = [arg.replace("{input}", input_path).replace("{out_dir}", out_dir) for arg in template]
        log.info("Running analytics command (job=%s) cwd=%s", job_id, REPO_DIR)

        command_failed = False
        error_msg = ""

        try:
            subprocess.run(cmd, check=True, cwd=REPO_DIR, timeout=7200)
        except subprocess.CalledProcessError as exc:
            command_failed = True
            error_msg = f"Command exited with code {exc.returncode}"
            log.error("Command failed: %s", error_msg)
        except subprocess.TimeoutExpired:
            command_failed = True
            error_msg = "Command timed out after 7200s"
            log.error(error_msg)

        upload_path, missing_candidates = _choose_output_json(out_dir)

        missing_msg = ""
        if missing_candidates and not os.path.isfile(upload_path):
            missing_msg = f"Missing expected outputs: {', '.join(missing_candidates)}"
            log.warning(missing_msg)
            error_msg = f"{error_msg}; {missing_msg}" if error_msg else missing_msg

        is_failure = command_failed and not os.path.isfile(upload_path)

        # Only create placeholder if analytics truly failed and produced no usable JSON.
        if command_failed and (not os.path.isfile(upload_path)):
            placeholder = {"placeholder": True, "reason": error_msg or "unknown", "job_id": job_id}
            with open(os.path.join(out_dir, "timeline.json"), "w") as f:
                json.dump(placeholder, f)
            upload_path = os.path.join(out_dir, "timeline.json")
            log.info("Generated placeholder timeline.json")

        # 1) Upload analytics JSON (always: either real report or placeholder)
        log.info("Uploading timeline -> s3://%s/%s (local=%s)", source_bucket, output_timeline_key, upload_path)
        s3.upload_file(upload_path, source_bucket, output_timeline_key)

        # 2) If analytics truly failed (no usable output), fail job (refund via analytics-complete-job)
        if is_failure:
            complete_job(job_id, "failed", error_msg)
            return

        # 3) Analytics phase is over: stop heartbeat before leaving processing state
        heartbeat.stop()

        # Mark analytics done (credit still reserved)
        complete_job(job_id, "analytics_done")

        # 4) Delivery: highlight must be produced (credit captured only on success)
        if not ENABLE_HIGHLIGHT:
            mark_delivery_failed(job_id, "Highlight generation disabled (ENABLE_HIGHLIGHT=false)")
            return

        if not output_highlight_key:
            mark_delivery_failed(job_id, "Missing output_highlight_key/output_s3_prefix for highlight")
            return

        try:
            highlight_path = build_highlight(input_path, upload_path, work_dir)
            if not highlight_path:
                mark_delivery_failed(job_id, "No play segments found in segments[]")
                return

            log.info("Uploading highlight -> s3://%s/%s", source_bucket, output_highlight_key)
            s3.upload_file(highlight_path, source_bucket, output_highlight_key)

            mark_delivered(job_id, output_highlight_key)

        except Exception as exc:
            mark_delivery_failed(job_id, f"Highlight generation/upload failed: {exc}")
            return

    except Exception as exc:
        log.exception("Job %s failed with exception", job_id)
        try:
            complete_job(job_id, "failed", f"Unhandled error: {str(exc)}")
        except Exception:
            pass
    finally:
        heartbeat.stop()
        if KEEP_WORK_DIR:
            log.warning("KEEP_WORK_DIR=true; preserving work_dir=%s", work_dir)
        else:
            shutil.rmtree(work_dir, ignore_errors=True)


def main():
    required = {
        "SUPABASE_URL": SUPABASE_URL,
        "SUPABASE_ANON_KEY": SUPABASE_ANON_KEY,
        "ANALYTICS_WORKER_KEY": ANALYTICS_WORKER_KEY,
        "ANALYTICS_CMD_BASIC": ANALYTICS_CMD_BASIC,
    }
    missing = [k for k, v in required.items() if not v]
    if missing:
        log.error("Missing required env vars: %s", ", ".join(missing))
        sys.exit(1)

    log.info(
        "Worker starting | id=%s region=%s poll=%ds idle_grace=%dmin self_stop=%s highlight=%s",
        WORKER_ID,
        AWS_REGION,
        POLL_INTERVAL,
        IDLE_GRACE_MINUTES,
        ENABLE_SELF_STOP,
        ENABLE_HIGHLIGHT,
    )

    reset_stale_jobs()

    idle_since = None
    log.info("Entering poll loop (interval=%ds)", POLL_INTERVAL)

    while True:
        try:
            resp = _call_fn("analytics-pull-job", {"worker_id": WORKER_ID, "tier": "basic"})

            if resp.status_code == 204:
                if idle_since is None:
                    idle_since = time.time()
                idle_secs = time.time() - idle_since
                if idle_secs >= IDLE_GRACE_MINUTES * 60:
                    log.info("No jobs for %.1f minutes; stopping.", idle_secs / 60.0)
                    stop_self()
                time.sleep(POLL_INTERVAL)
                continue

            idle_since = None

            if resp.status_code >= 500:
                _log_http_error("Pull-job transient error", resp)
                time.sleep(30)
                continue

            if resp.status_code == 200:
                data = _safe_json(resp)
                if not data:
                    _log_http_error("Pull-job 200 non-JSON", resp)
                    time.sleep(POLL_INTERVAL)
                    continue

                job = data.get("job")
                if not job:
                    log.warning("200 but no job in response")
                    time.sleep(POLL_INTERVAL)
                    continue

                process_job(job)
                continue

            _log_http_error("Pull-job unexpected response", resp)
            time.sleep(POLL_INTERVAL)

        except Exception as exc:
            log.exception("Poll loop error: %s", exc)
            time.sleep(POLL_INTERVAL)


if __name__ == "__main__":
    main()
