"""
OpenAI Agent class with tool calling loop.

Blog post: https://dadops.dev/blog/building-ai-agents/
Code Blocks 1, 2, 5, 7: Weather tool demo + Agent class + research assistant.

REQUIRES: OpenAI API key (set OPENAI_API_KEY environment variable)
"""
import json
import os
from tool_functions import search_files, read_file, calculate


# ── Code Block 5: The Agent Class ──

class Agent:
    def __init__(self, system_prompt, tools, tool_functions, max_turns=10):
        from openai import OpenAI
        self.client = OpenAI()
        self.system_prompt = system_prompt
        self.tools = tools                     # tool schemas for the API
        self.tool_functions = tool_functions    # {"name": callable} mapping
        self.max_turns = max_turns

    def run(self, user_message):
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]

        for turn in range(self.max_turns):
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=self.tools,
            )

            msg = response.choices[0].message
            messages.append(msg)

            # No tool calls? The model is done — return the answer
            if not msg.tool_calls:
                return msg.content

            # Execute every tool the model requested
            for tool_call in msg.tool_calls:
                name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)

                try:
                    result = self.tool_functions[name](**args)
                except KeyError:
                    result = f"Unknown tool '{name}'. Available tools: {list(self.tool_functions)}"
                except TypeError as e:
                    result = f"Wrong arguments for {name}: {e}. Check the schema."
                except Exception as e:
                    result = f"Tool '{name}' failed: {type(e).__name__}: {e}"

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result) if not isinstance(result, str) else result,
                })

        return "Agent reached maximum turns without finishing."


# ── Code Block 7: Research Assistant Tool Schemas ──

RESEARCH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for files matching a glob pattern",
            "parameters": {
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
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"},
                    "max_lines": {"type": "integer", "description": "Max lines (default 50)"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a mathematical expression",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Math expression, e.g. '(10 + 20) / 3'"}
                },
                "required": ["expression"]
            }
        }
    }
]


def run_weather_demo():
    """Code Blocks 1 & 2: Weather tool calling demo."""
    from openai import OpenAI
    client = OpenAI()

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City name, e.g. 'San Francisco'"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "What's the weather in Tokyo?"}],
        tools=tools,
    )

    msg = response.choices[0].message
    print(f"Tool called: {msg.tool_calls[0].function.name}")
    print(f"Arguments: {msg.tool_calls[0].function.arguments}")

    # Execute the tool
    def get_weather(location):
        return {"temp": 22, "condition": "partly cloudy", "city": location}

    tool_call = msg.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    result = get_weather(**args)

    messages = [
        {"role": "user", "content": "What's the weather in Tokyo?"},
        msg,
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(result),
        }
    ]

    final = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        tools=tools,
    )

    print(f"Final answer: {final.choices[0].message.content}")


def run_research_agent():
    """Code Block 7: Research assistant agent."""
    agent = Agent(
        system_prompt="You are a helpful research assistant. Use your tools to answer questions accurately.",
        tools=RESEARCH_TOOLS,
        tool_functions={
            "search_files": search_files,
            "read_file": read_file,
            "calculate": calculate,
        },
    )

    answer = agent.run("How many Python files are in this directory, and what's the total line count?")
    print(f"\nAgent answer: {answer}")


if __name__ == "__main__":
    if not os.environ.get("OPENAI_API_KEY"):
        print("SKIP: Set OPENAI_API_KEY to run this example")
        print("Agent class and tool schemas loaded successfully.")
    else:
        print("=== Weather Tool Demo ===")
        run_weather_demo()
        print("\n=== Research Agent Demo ===")
        run_research_agent()
