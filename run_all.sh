#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

VIDEO_DIR="01. Videos"
OUTPUT_DIR="data/output"
LOG_DIR="data/logs"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

# Activate virtualenv if present
if [[ -f venv/bin/activate ]]; then
    source venv/bin/activate
fi

TOTAL=$(find "$VIDEO_DIR" -maxdepth 1 -name '*.mp4' | wc -l)
CURRENT=0
FAILED=0

echo "=========================================="
echo " Batch pipeline run — $TOTAL videos"
echo " Started: $(date)"
echo "=========================================="

for VIDEO in "$VIDEO_DIR"/*.mp4; do
    CURRENT=$((CURRENT + 1))
    BASENAME="$(basename "$VIDEO" .mp4)"
    LOGFILE="$LOG_DIR/${BASENAME}.log"

    echo ""
    echo "------------------------------------------"
    echo " [$CURRENT/$TOTAL] $BASENAME"
    echo " Log: $LOGFILE"
    echo "------------------------------------------"

    if python -m src.pipeline --video "$VIDEO" --output "$OUTPUT_DIR" 2>&1 | tee "$LOGFILE"; then
        echo " ✓ Done: $BASENAME"
    else
        echo " ✗ FAILED: $BASENAME (exit $?)"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
echo "=========================================="
echo " Batch complete: $(date)"
echo " Total: $TOTAL | Succeeded: $((TOTAL - FAILED)) | Failed: $FAILED"
echo "=========================================="
