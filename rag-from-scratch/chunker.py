"""
Recursive text chunker for RAG pipelines.

Splits text into overlapping chunks using a hierarchy of separators,
trying the coarsest separator first and recursing to finer ones when
chunks exceed the size limit.

From: https://dadops.dev/blog/rag-from-scratch/
"""


def chunk_text(text, max_chars=1000, overlap=200):
    """Split text into overlapping chunks using recursive separators."""
    separators = ["\n\n", "\n", ". ", " "]
    chunks = []

    def split_recursive(text, sep_idx=0):
        # Base case: text fits in one chunk
        if len(text) <= max_chars:
            if text.strip():
                chunks.append(text.strip())
            return

        # Try current separator
        sep = separators[sep_idx]
        parts = text.split(sep)

        current_chunk = ""
        for part in parts:
            candidate = current_chunk + sep + part if current_chunk else part
            if len(candidate) > max_chars and current_chunk:
                chunks.append(current_chunk.strip())
                # Start next chunk with overlap from end of current
                overlap_text = current_chunk[-overlap:] if overlap else ""
                current_chunk = overlap_text + sep + part
            else:
                current_chunk = candidate

        if current_chunk.strip():
            # If this chunk is still too big, try finer separator
            if len(current_chunk) > max_chars and sep_idx + 1 < len(separators):
                split_recursive(current_chunk, sep_idx + 1)
            else:
                chunks.append(current_chunk.strip())

    split_recursive(text)
    return chunks


if __name__ == "__main__":
    sample = """Solar panels convert sunlight into electricity through the
photovoltaic effect. When photons hit silicon cells, they knock electrons
loose, creating an electrical current.

Installation requires careful roof assessment. South-facing roofs with
15-40 degree pitch are ideal in the northern hemisphere. Shading from
trees or neighboring buildings can reduce output by 10-25%.

A typical residential system is 6-10 kW, requiring 15-25 panels. At
average US electricity rates, payback period is 6-10 years. Federal tax
credits currently cover 30% of installation costs."""

    chunks = chunk_text(sample, max_chars=300, overlap=60)
    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i} ({len(chunk)} chars) ---")
        print(chunk[:80] + "..." if len(chunk) > 80 else chunk)
        print()
