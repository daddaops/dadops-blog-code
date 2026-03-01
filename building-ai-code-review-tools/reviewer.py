"""
LLM-powered code review: single-file analyzer, diff-aware PR reviewer,
and multi-agent specialist pipeline.

Blog post: https://dadops.dev/blog/building-ai-code-review-tools/
Code Blocks 1, 2, and 4.

Requires: ANTHROPIC_API_KEY environment variable.
"""
import asyncio
import json
import subprocess
import sys

from pydantic import BaseModel
from anthropic import Anthropic


# ── Code Block 1: Single-File Code Analyzer ──

class ReviewFinding(BaseModel):
    line: int
    severity: str   # "critical", "warning", "info"
    category: str   # "bug", "security", "performance", "style"
    description: str
    suggestion: str


class FileReview(BaseModel):
    findings: list[ReviewFinding]
    summary: str


REVIEW_PROMPT = """Review this code file for bugs, security issues,
performance problems, and style issues. Return JSON matching this schema:
{{"findings": [{{"line": int, "severity": "critical|warning|info",
"category": "bug|security|performance|style",
"description": "what's wrong", "suggestion": "how to fix"}}],
"summary": "one-line overall assessment"}}

Example finding:
{{"line": 15, "severity": "critical", "category": "security",
"description": "SQL query built with string formatting is vulnerable to injection",
"suggestion": "Use parameterized queries: cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))"}}

File: {filename}
```
{code}
```"""


def review_file(filepath: str) -> FileReview:
    client = Anthropic()
    code = open(filepath).read()

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user",
                   "content": REVIEW_PROMPT.format(
                       filename=filepath, code=code)}]
    )
    data = json.loads(response.content[0].text)
    return FileReview(**data)


# ── Code Block 2: Diff-Aware PR Reviewer ──

def get_pr_diff(base_branch: str = "main") -> str:
    """Get the unified diff for the current branch vs base."""
    result = subprocess.run(
        ["git", "diff", f"{base_branch}...HEAD", "--unified=5"],
        capture_output=True, text=True, check=True
    )
    return result.stdout


DIFF_REVIEW_PROMPT = """Review this pull request diff. Focus on what CHANGED,
not on pre-existing code. For each issue found, reference the exact file
and line number from the diff.

Return JSON: {{"findings": [...], "summary": "..."}}
Same schema as before but add "file": "path/to/file" to each finding.

Diff:
```
{diff}
```"""


def review_pr(base_branch: str = "main") -> FileReview:
    client = Anthropic()
    diff = get_pr_diff(base_branch)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user",
                   "content": DIFF_REVIEW_PROMPT.format(diff=diff)}]
    )
    return FileReview(**json.loads(response.content[0].text))


# ── Code Block 4: Multi-Agent Review Pipeline ──

SPECIALIST_PROMPTS = {
    "security": """Review ONLY for security vulnerabilities. Check for:
- SQL injection, XSS, command injection
- Hardcoded secrets or credentials
- Authentication/authorization flaws
- Path traversal, SSRF, insecure deserialization
- Missing input validation at trust boundaries
Return JSON: {{"findings": [...], "summary": "..."}}""",

    "performance": """Review ONLY for performance issues. Check for:
- O(n^2) or worse algorithms where O(n) is possible
- N+1 database queries
- Missing caching opportunities
- Unnecessary memory allocations in loops
- Blocking I/O in async code paths
Return JSON: {{"findings": [...], "summary": "..."}}""",

    "style": """Review ONLY for code quality and readability. Check for:
- Unclear variable or function names
- Missing error handling for external calls
- Dead code or unreachable branches
- Functions doing too many things (SRP violations)
- Missing or misleading docstrings on public APIs
Return JSON: {{"findings": [...], "summary": "..."}}"""
}


async def run_specialist(client, name: str, diff: str) -> list[dict]:
    """Run one specialist reviewer on the diff."""
    prompt = SPECIALIST_PROMPTS[name] + f"\n\nDiff:\n```\n{diff}\n```"
    response = await asyncio.to_thread(
        client.messages.create,
        model="claude-sonnet-4-20250514",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}]
    )
    data = json.loads(response.content[0].text)
    for finding in data["findings"]:
        finding["source_agent"] = name
    return data["findings"]


async def multi_agent_review(diff: str) -> list[dict]:
    """Run all specialists in parallel, then aggregate."""
    client = Anthropic()
    tasks = [run_specialist(client, name, diff)
             for name in SPECIALIST_PROMPTS]
    all_findings = await asyncio.gather(*tasks)

    # Flatten and deduplicate by (file, line, category)
    merged = {}
    for findings in all_findings:
        for f in findings:
            key = (f.get("file", ""), f["line"], f["category"])
            if key not in merged or f["severity"] == "critical":
                merged[key] = f

    # Sort: critical first, then warning, then info
    severity_order = {"critical": 0, "warning": 1, "info": 2}
    results = sorted(merged.values(),
                     key=lambda f: severity_order.get(f["severity"], 3))
    return results[:10]  # Cap at 10 comments per review


# ── CLI demo ──

if __name__ == "__main__":
    import os
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("SKIP: ANTHROPIC_API_KEY not set — cannot run LLM-based reviewer.")
        print("Set ANTHROPIC_API_KEY=sk-... to test the reviewer.")
        print()
        print("Scripts verified structurally (imports, Pydantic models, function signatures).")
        # Verify imports and models work
        f = ReviewFinding(line=1, severity="warning", category="style",
                          description="test", suggestion="test")
        r = FileReview(findings=[f], summary="test review")
        print(f"  ReviewFinding model: OK ({f.severity})")
        print(f"  FileReview model: OK ({len(r.findings)} finding(s))")
        print(f"  Specialist agents: {list(SPECIALIST_PROMPTS.keys())}")
        sys.exit(0)

    if len(sys.argv) < 2:
        print("Usage: python reviewer.py <filepath>")
        print("  Reviews a single file with the LLM analyzer.")
        sys.exit(1)

    filepath = sys.argv[1]
    print(f"Reviewing {filepath}...")
    review = review_file(filepath)
    print(f"\nSummary: {review.summary}\n")
    for f in review.findings:
        print(f"Line {f.line} [{f.severity}] {f.category}: {f.description}")
