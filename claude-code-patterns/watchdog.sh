#!/usr/bin/env bash
# Pattern 6: The Watchdog
# Blog post: https://dadops.dev/blog/claude-code-patterns/
#
# A supervisor process that monitors the headless loop and restarts it
# when something goes wrong. Three checks:
#   1. Is the loop process alive?
#   2. Is the log file growing? (stuck detection)
#   3. Has a stop file appeared?
#
# Prerequisites: headless_loop.sh running with loop.pid written
# Usage: ./watchdog.sh
set -euo pipefail

STUCK_TIMEOUT=1800  # 30 minutes
CHECK_INTERVAL=30
LAST_LOG_SIZE=0
LAST_LOG_CHANGE="$(date +%s)"

start_loop() {
    echo "Starting headless loop..."
    bash headless_loop.sh &
    echo "Loop started with PID $!"
}

while true; do
    sleep "$CHECK_INTERVAL"

    # Check 1: Is the loop process alive?
    PID="$(cat loop.pid 2>/dev/null)"
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "Loop died. Restarting..."
        start_loop
        continue
    fi

    # Check 2: Is the log file growing?
    CURRENT_SIZE="$(stat -c%s loop.log 2>/dev/null || echo 0)"
    NOW="$(date +%s)"

    if [[ "$CURRENT_SIZE" -ne "$LAST_LOG_SIZE" ]]; then
        LAST_LOG_SIZE="$CURRENT_SIZE"
        LAST_LOG_CHANGE="$NOW"
    else
        ELAPSED=$(( NOW - LAST_LOG_CHANGE ))
        if [[ "$ELAPSED" -ge "$STUCK_TIMEOUT" ]]; then
            echo "STUCK for ${ELAPSED}s. Killing and restarting."
            kill "$PID" 2>/dev/null
            start_loop
            LAST_LOG_SIZE=0
            LAST_LOG_CHANGE="$(date +%s)"
        fi
    fi

    # Check 3: Has the stop file appeared?
    [[ -f "loop_stop.md" ]] && echo "Stop file found. Shutting down." && exit 0
done
