"""
Standalone tool functions for AI agents.

Blog post: https://dadops.dev/blog/building-ai-agents/
Code Blocks 6, 8, 10, 11: Tool functions, error handling, convergence
detection, and result truncation.

These functions run WITHOUT any API keys.
"""
import glob
import json
import os


# ── Code Block 6: Research Assistant Tool Functions ──

def search_files(pattern, directory="."):
    """Find files matching a glob pattern."""
    matches = glob.glob(os.path.join(directory, pattern), recursive=True)
    return {"files": matches[:20], "total": len(matches)}


def read_file(path, max_lines=50):
    """Read the first N lines of a file."""
    try:
        with open(path) as f:
            all_lines = f.readlines()
        return {
            "content": "".join(all_lines[:max_lines]),
            "total_lines": len(all_lines),
        }
    except FileNotFoundError:
        return {"error": f"File not found: {path}"}


def calculate(expression):
    """Safely evaluate a math expression."""
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return {"error": "Invalid characters in expression"}
    return {"result": eval(expression)}


# ── Code Block 11: Result Truncation ──

def truncate_result(result, max_chars=2000):
    """Keep tool results from blowing up the context window."""
    text = json.dumps(result) if not isinstance(result, str) else result
    if len(text) > max_chars:
        return text[:max_chars] + f"\n... (truncated, {len(text)} chars total)"
    return text


# ── Self-tests ──

def test_search_files():
    """Test search_files with known patterns."""
    print("=== search_files Test ===")

    # Search for Python files in current directory
    result = search_files("*.py")
    print(f"  *.py in current dir: {result['total']} files found")
    assert result["total"] >= 1, "Should find at least this file"
    assert isinstance(result["files"], list)

    # Search for non-existent pattern
    result = search_files("*.nonexistent_xyz_123")
    assert result["total"] == 0
    print(f"  Non-existent pattern: {result['total']} files (correct)")
    print("  PASS\n")


def test_read_file():
    """Test read_file with this script itself."""
    print("=== read_file Test ===")

    # Read this file
    result = read_file(__file__, max_lines=5)
    assert "content" in result
    assert result["total_lines"] > 0
    print(f"  Read {__file__}: {result['total_lines']} total lines, "
          f"returned first 5")

    # Try non-existent file
    result = read_file("/nonexistent/path/file.txt")
    assert "error" in result
    print(f"  Non-existent file: {result['error']}")
    print("  PASS\n")


def test_calculate():
    """Test calculate with valid and invalid expressions."""
    print("=== calculate Test ===")

    # Valid expressions
    assert calculate("2 + 3")["result"] == 5
    assert calculate("(10 + 20) / 3")["result"] == 10.0
    assert abs(calculate("3.14 * 2")["result"] - 6.28) < 0.001
    print("  2 + 3 = 5, (10+20)/3 = 10.0, 3.14*2 = 6.28")

    # Invalid expression (contains letters)
    result = calculate("import os")
    assert "error" in result
    print(f"  Blocked 'import os': {result['error']}")
    print("  PASS\n")


def test_truncate_result():
    """Test truncate_result with short and long inputs."""
    print("=== truncate_result Test ===")

    # Short result passes through
    short = truncate_result({"key": "value"})
    assert "truncated" not in short
    print(f"  Short result: {short}")

    # Long result gets truncated
    long_result = {"data": "x" * 3000}
    truncated = truncate_result(long_result, max_chars=100)
    assert "truncated" in truncated
    assert len(truncated) < 200  # 100 chars + truncation message
    print(f"  Long result truncated to: {len(truncated)} chars")
    print("  PASS\n")


def test_error_handling_patterns():
    """Test the enhanced error handling pattern from Code Block 8."""
    print("=== Error Handling Pattern Test ===")

    tool_functions = {
        "search_files": search_files,
        "read_file": read_file,
        "calculate": calculate,
    }

    # Unknown tool
    name = "nonexistent_tool"
    try:
        result = tool_functions[name]()
    except KeyError:
        result = f"Unknown tool '{name}'. Available: {list(tool_functions)}"
    print(f"  Unknown tool: {result}")

    # Wrong arguments
    name = "calculate"
    try:
        result = tool_functions[name](wrong_arg="bad")
    except TypeError as e:
        result = f"Wrong arguments for {name}: {e}."
    print(f"  Wrong args: {result}")

    print("  PASS\n")


if __name__ == "__main__":
    test_search_files()
    test_read_file()
    test_calculate()
    test_truncate_result()
    test_error_handling_patterns()
    print("All tool function tests passed!")
