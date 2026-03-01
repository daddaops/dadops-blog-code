# Practical Claude Code Patterns for Real Projects

Blog post: https://dadops.dev/blog/claude-code-patterns/

## What's Here

This post is a **recipe book** of 7 patterns for running Claude Code autonomously.
The code is mostly bash scripts, JSON config, and markdown templates — patterns
you copy and adapt, not standalone benchmarks.

## Files

- `headless_loop.sh` — Pattern 1: The core while-true loop calling `claude -p`
- `parse_stream.sh` — Pattern 5: Stream JSON parser for real-time visibility
- `watchdog.sh` — Pattern 6: Stuck detection and automatic restart
- `settings_local.json` — Pattern 3: Example `.claude/settings.local.json` permissions
- `status_template.md` — Pattern 4: Example STATUS.md for phase tracking

## Running

Most scripts require the `claude` CLI installed and configured. The headless loop
and watchdog are designed for unattended operation and will incur API costs.

The `parse_stream.sh` can be tested with sample JSONL input (see `output/sample_stream.jsonl`).

## Patterns Not Extracted

- Pattern 2 (Hot-Swappable Prompts): A one-line change shown inline in the blog
- Pattern 7 (Structured Commits): Git log format convention, not a script
