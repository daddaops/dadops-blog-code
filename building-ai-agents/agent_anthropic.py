"""
Anthropic ClaudeAgent class with tool calling loop.

Blog post: https://dadops.dev/blog/building-ai-agents/
Code Blocks 3, 4, 13: Anthropic tool calling + ClaudeAgent class.

REQUIRES: Anthropic API key (set ANTHROPIC_API_KEY environment variable)
"""
import json
import os
from tool_functions import search_files, read_file, calculate


# ── Code Block 13: The ClaudeAgent Class ──

class ClaudeAgent:
    def __init__(self, system_prompt, tools, tool_functions, max_turns=10):
        import anthropic
        self.client = anthropic.Anthropic()
        self.system_prompt = system_prompt
        self.tools = tools
        self.tool_functions = tool_functions
        self.max_turns = max_turns

    def run(self, user_message):
        messages = [{"role": "user", "content": user_message}]

        for turn in range(self.max_turns):
            response = self.client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=4096,
                system=self.system_prompt,
                tools=self.tools,
                messages=messages,
            )

            # Append the full assistant response
            messages.append({"role": "assistant", "content": response.content})

            # end_turn means the model is done — extract text and return
            if response.stop_reason == "end_turn":
                return "".join(
                    block.text for block in response.content
                    if hasattr(block, "text")
                )

            # Execute tools and collect results
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    try:
                        result = self.tool_functions[block.name](**block.input)
                    except Exception as e:
                        result = {"error": str(e)}

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result)
                            if not isinstance(result, str) else result,
                    })

            # Tool results go in a "user" message (not a "tool" role)
            messages.append({"role": "user", "content": tool_results})

        return "Agent reached maximum turns without finishing."


# ── Anthropic tool schemas (use input_schema instead of parameters) ──

CLAUDE_TOOLS = [
    {
        "name": "search_files",
        "description": "Search for files matching a glob pattern",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern, e.g. '**/*.py'"
                },
                "directory": {
                    "type": "string",
                    "description": "Directory to search in (default: current)"
                }
            },
            "required": ["pattern"]
        }
    },
    {
        "name": "read_file",
        "description": "Read the contents of a file",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path to read"},
                "max_lines": {"type": "integer", "description": "Max lines (default 50)"}
            },
            "required": ["path"]
        }
    },
    {
        "name": "calculate",
        "description": "Evaluate a mathematical expression",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression"}
            },
            "required": ["expression"]
        }
    }
]


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("SKIP: Set ANTHROPIC_API_KEY to run this example")
        print("ClaudeAgent class and tool schemas loaded successfully.")
    else:
        agent = ClaudeAgent(
            system_prompt="You are a helpful research assistant. Use your tools to answer questions accurately.",
            tools=CLAUDE_TOOLS,
            tool_functions={
                "search_files": search_files,
                "read_file": read_file,
                "calculate": calculate,
            },
        )
        answer = agent.run("How many Python files are in this directory?")
        print(f"Agent answer: {answer}")
