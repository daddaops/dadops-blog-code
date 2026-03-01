#!/usr/bin/env bash
# Verify that the commit hashes shown in Pattern 7 of the blog post
# match the real git history.
#
# Blog post: https://dadops.dev/blog/claude-code-patterns/
# This script must be run from the dadops-site repo root.
#
# Usage: cd /path/to/dadops-site && bash /path/to/verify_commits.sh
set -euo pipefail

echo "=== Verifying Pattern 7 Commit Hashes ==="
echo ""

# Commits claimed in the blog (Pattern 7 code block)
declare -A EXPECTED=(
    ["7839cac"]="[claude-code-patterns] Phase 1 RESEARCH complete"
    ["41d9676"]="[ralph-loop-explained] Phase 4 INTEGRATE complete"
    ["46e3ee8"]="[ralph-loop-explained] Phase 3 POLISH complete"
    ["29382ae"]="[ralph-loop-explained] Phase 2 WRITE complete"
    ["fbf7830"]="[ralph-loop-explained] Phase 1 RESEARCH complete"
    ["f5e7e07"]="[attention-from-scratch] Phase 4 INTEGRATE complete"
    ["ce0bcb4"]="[attention-from-scratch] Phase 3 POLISH complete"
    ["45813bd"]="[attention-from-scratch] Phase 2 WRITE complete"
)

PASS=0
FAIL=0

for hash in "${!EXPECTED[@]}"; do
    expected="${EXPECTED[$hash]}"
    # Get the actual commit message (first line) for this hash
    actual=$(git log --format="%s" -1 "$hash" 2>/dev/null | head -c 60) || actual="NOT FOUND"

    if [[ "$actual" == *"$expected"* ]] || [[ "$expected" == *"$(echo "$actual" | cut -d— -f1 | xargs)"* ]]; then
        echo "PASS: $hash — $expected"
        PASS=$((PASS + 1))
    else
        echo "FAIL: $hash"
        echo "  Expected: $expected"
        echo "  Actual:   $actual"
        FAIL=$((FAIL + 1))
    fi
done

echo ""
echo "Results: $PASS passed, $FAIL failed out of ${#EXPECTED[@]} commits"

# Also verify that 'git log --oneline | grep "Phase"' produces results
echo ""
echo "=== Phase Commit Count ==="
PHASE_COUNT=$(git log --oneline | grep -c "Phase" || true)
echo "Total phase-format commits in repo: $PHASE_COUNT"

# Expected output:
# All 8 commits should PASS
# Phase commit count should be > 100 (this repo has 170+ blog posts with 4 phases each)
