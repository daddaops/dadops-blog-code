#!/usr/bin/env bash
# Pattern 5: Stream JSON Parser for Real-Time Visibility
# Blog post: https://dadops.dev/blog/claude-code-patterns/
#
# Parses Claude's --output-format stream-json output into a human-readable feed.
# Also acts as a "flight recorder" — appends all raw JSON to a log file.
#
# Prerequisites: jq installed
# Usage: claude -p "..." --output-format stream-json 2>&1 | ./parse_stream.sh
#    or: cat sample_stream.jsonl | ./parse_stream.sh
set -euo pipefail

RAW_LOG="${1:-stream_raw.jsonl}"

parse_stream() {
    while IFS= read -r line; do
        echo "$line" >> "$RAW_LOG"  # flight recorder

        type=$(echo "$line" | jq -r '.type // empty' 2>/dev/null) || continue

        case "$type" in
            assistant)
                text=$(echo "$line" | jq -r '.message.content[]? |
                    select(.type == "text") | .text // empty' 2>/dev/null)
                [[ -n "$text" ]] && echo "Claude: $text"
                ;;
            tool_use)
                tool=$(echo "$line" | jq -r '.tool_name' 2>/dev/null)
                echo "Tool: $tool"
                ;;
            tool_result)
                is_err=$(echo "$line" | jq -r '.is_error // false' 2>/dev/null)
                [[ "$is_err" == "true" ]] && echo "ERROR in tool result"
                ;;
            result)
                cost=$(echo "$line" | jq -r '.cost_usd // "?"' 2>/dev/null)
                dur=$(echo "$line" | jq -r '.duration_ms // "?"' 2>/dev/null)
                echo "Done — Cost: \$$cost | Time: ${dur}ms"
                ;;
        esac
    done
}

parse_stream
