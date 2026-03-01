#!/usr/bin/env bash
# Pattern 1: The Headless Loop
# Blog post: https://dadops.dev/blog/claude-code-patterns/
#
# Runs Claude Code in a non-interactive loop. Each iteration reads the prompt
# from a file, runs Claude, and checks for a stop signal.
#
# Prerequisites: claude CLI installed and authenticated
# Usage: ./headless_loop.sh
set -euo pipefail

PROMPT_FILE="prompt.md"
STOP_FILE="loop_stop.md"

echo $$ > loop.pid  # write PID for watchdog

while true; do
    # Graceful shutdown via file
    [[ -f "$STOP_FILE" ]] && echo "Stop file found. Exiting." && exit 0

    # Pattern 2: Hot-swappable prompts — re-read every iteration
    PROMPT="$(cat "$PROMPT_FILE")"

    # Pattern 2 extension: validate prompt isn't empty
    if [[ -z "$PROMPT" ]]; then
        echo "WARNING: prompt.md is empty — waiting 5s..."
        sleep 5
        continue
    fi

    # Run Claude in prompt mode with streaming JSON output
    claude -p "$PROMPT" \
        --output-format stream-json \
        --dangerously-skip-permissions \
        2>&1 | tee -a stream.jsonl

    # Check again (Claude may have written the stop file)
    [[ -f "$STOP_FILE" ]] && echo "Stop file found. Exiting." && exit 0

    sleep 3
done
