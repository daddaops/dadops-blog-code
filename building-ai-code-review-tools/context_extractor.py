"""
AST-based repository context extraction and diff position mapping.

Blog post: https://dadops.dev/blog/building-ai-code-review-tools/
Code Blocks 3 & 5 (first function): extract_file_context, enrich_diff_with_context,
map_line_to_diff_position.

These functions run WITHOUT any API keys.
"""
import ast
import re


# ── Code Block 3: Repository Context Extractor ──

def extract_file_context(filepath: str) -> dict:
    """Extract structural context: imports, function signatures, classes."""
    source = open(filepath).read()
    tree = ast.parse(source)

    context = {"imports": [], "functions": [], "classes": []}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            context["imports"].append(ast.get_source_segment(source, node))
        elif isinstance(node, ast.FunctionDef):
            args = [a.arg for a in node.args.args]
            context["functions"].append(
                f"def {node.name}({', '.join(args)}) -> line {node.lineno}"
            )
        elif isinstance(node, ast.ClassDef):
            context["classes"].append(
                f"class {node.name} -> line {node.lineno}"
            )
    return context


def enrich_diff_with_context(diff_text: str) -> str:
    """Add file structure context to each changed file in the diff."""
    enriched_parts = [diff_text, "\n--- Repository Context ---"]
    seen_files = set()

    for line in diff_text.splitlines():
        if line.startswith("+++ b/") and line[6:].endswith(".py"):
            filepath = line[6:]
            if filepath not in seen_files:
                seen_files.add(filepath)
                try:
                    ctx = extract_file_context(filepath)
                    enriched_parts.append(f"\n{filepath}:")
                    enriched_parts.append(f"  Imports: {ctx['imports'][:10]}")
                    enriched_parts.append(f"  Functions: {ctx['functions']}")
                    enriched_parts.append(f"  Classes: {ctx['classes']}")
                except (FileNotFoundError, SyntaxError):
                    pass

    return "\n".join(enriched_parts)


# ── Code Block 5 (first function): Diff Position Mapping ──

def map_line_to_diff_position(diff_text: str) -> dict:
    """Build a mapping from (file, line_number) to diff position."""
    position_map = {}
    current_file = None
    diff_position = 0
    current_line = 0

    for raw_line in diff_text.splitlines():
        if raw_line.startswith("+++ b/"):
            current_file = raw_line[6:]
            diff_position = 0
        elif raw_line.startswith("@@ "):
            match = re.search(r"\+(\d+)", raw_line)
            current_line = int(match.group(1)) - 1 if match else 0
            diff_position += 1
        elif current_file and diff_position > 0:
            diff_position += 1
            if not raw_line.startswith("-"):
                current_line += 1
            if raw_line.startswith("+"):
                position_map[(current_file, current_line)] = diff_position

    return position_map


# ── Self-tests ──

def test_extract_file_context():
    """Test AST extraction on this file itself."""
    print("=== extract_file_context Test ===")
    ctx = extract_file_context(__file__)

    assert len(ctx["imports"]) > 0, "Should find ast and re imports"
    assert len(ctx["functions"]) > 0, "Should find function definitions"
    print(f"  Imports found: {len(ctx['imports'])}")
    print(f"  Functions found: {len(ctx['functions'])}")
    print(f"  Classes found: {len(ctx['classes'])}")

    # Check specific functions are found
    func_names = [f.split("(")[0] for f in ctx["functions"]]
    assert any("extract_file_context" in f for f in func_names)
    assert any("map_line_to_diff_position" in f for f in func_names)
    print("  PASS\n")


def test_map_line_to_diff_position():
    """Test diff position mapping with a synthetic diff."""
    print("=== map_line_to_diff_position Test ===")

    sample_diff = """\
--- a/app/auth.py
+++ b/app/auth.py
@@ -10,6 +10,8 @@ def authenticate(user):
     if user.is_valid():
         return True
+    # Added logging
+    log.info(f"Auth attempt: {user.name}")
     return False
"""
    pos_map = map_line_to_diff_position(sample_diff)
    print(f"  Position map: {pos_map}")

    # The + lines should be mapped
    assert ("app/auth.py", 12) in pos_map, "Line 12 should be mapped"
    assert ("app/auth.py", 13) in pos_map, "Line 13 should be mapped"
    print(f"  Added line 12 at diff position {pos_map[('app/auth.py', 12)]}")
    print(f"  Added line 13 at diff position {pos_map[('app/auth.py', 13)]}")
    print("  PASS\n")


def test_enrich_diff():
    """Test diff enrichment with a diff pointing to this file."""
    print("=== enrich_diff_with_context Test ===")

    # Create a fake diff pointing to this file
    fake_diff = f"+++ b/{__file__}\n@@ -1,3 +1,4 @@\n+# new line"
    enriched = enrich_diff_with_context(fake_diff)
    assert "Repository Context" in enriched
    assert "Functions:" in enriched
    print(f"  Enriched diff length: {len(enriched)} chars (vs {len(fake_diff)} original)")
    print("  PASS\n")


if __name__ == "__main__":
    test_extract_file_context()
    test_map_line_to_diff_position()
    test_enrich_diff()
    print("All context extractor tests passed!")
