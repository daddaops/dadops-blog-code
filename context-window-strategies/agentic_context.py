"""Agentic context management with tool-based document navigation.

Gives the LLM search and read_section tools to selectively navigate
a document, reading only the sections needed to answer a question.

Requires: ANTHROPIC_API_KEY environment variable.
"""

from anthropic import Anthropic

client = Anthropic()

TOOLS = [
    {
        "name": "search",
        "description": "Search the document for relevant sections.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "read_section",
        "description": "Read the full text of a section by its heading.",
        "input_schema": {
            "type": "object",
            "properties": {
                "heading": {"type": "string", "description": "Section heading"}
            },
            "required": ["heading"]
        }
    }
]


def context_agent(doc_index, search_fn, read_fn, question, max_reads=5):
    """Navigate a document with tools to answer a question."""
    messages = [{
        "role": "user",
        "content": (
            f"Document outline:\n{doc_index}\n\n"
            f"Question: {question}\n\n"
            f"Search and read sections to answer this. "
            f"You have {max_reads} reads -- be selective."
        )
    }]
    reads_used = 0

    while reads_used < max_reads:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1500,
            tools=TOOLS,
            messages=messages
        )

        # If the model stopped generating (no more tool calls), return
        if response.stop_reason == "end_turn":
            return response.content[0].text

        # Append assistant turn once, then collect all tool results
        messages.append({"role": "assistant", "content": response.content})
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                if block.name == "search":
                    result = search_fn(block.input["query"])
                else:
                    result = read_fn(block.input["heading"])
                    reads_used += 1
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })
        messages.append({"role": "user", "content": tool_results})

    return "Reached read limit without a final answer."


if __name__ == "__main__":
    print("=== Agentic Context Management Demo ===\n")

    # Sample document structure
    doc_sections = {
        "Introduction": "This contract governs software consulting services between Party A and Party B.",
        "Payment Terms": "Party A shall pay within 30 days. Late payments accrue 1.5% monthly interest.",
        "Intellectual Property": "All deliverables become property of Party A upon full payment.",
        "Confidentiality": "Both parties keep proprietary information confidential for 3 years.",
        "Liability": "Total liability capped at fees paid in preceding 12 months.",
        "Termination": "Either party may terminate with 30 days written notice.",
    }

    doc_index = "\n".join(f"- {heading}" for heading in doc_sections)
    print(f"Document outline:\n{doc_index}\n")

    # Define local search and read functions
    def search_fn(query):
        query_words = set(query.lower().split())
        results = []
        for heading, text in doc_sections.items():
            text_words = set(text.lower().split())
            overlap = len(query_words & text_words)
            if overlap > 0:
                results.append(f"- {heading} ({overlap} keyword matches)")
        return "\n".join(results) if results else "No matching sections found."

    def read_fn(heading):
        for h, text in doc_sections.items():
            if heading.lower() in h.lower():
                return text
        return f"Section '{heading}' not found."

    print("To run the full agentic pipeline with the Anthropic API:")
    print("  Set ANTHROPIC_API_KEY and call:")
    print("  context_agent(doc_index, search_fn, read_fn, 'your question')")

    # Show what the local tools produce
    print(f"\nLocal search('payment'):")
    print(f"  {search_fn('payment')}")
    print(f"\nLocal read('Payment Terms'):")
    print(f"  {read_fn('Payment Terms')}")
    print("\nDone.")
