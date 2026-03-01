"""
Full RAG pipeline: chunk → embed → retrieve → generate.

Ties together all components into a single RAGPipeline class.
Requires an Anthropic API key (ANTHROPIC_API_KEY env var) for the
generation step.

From: https://dadops.dev/blog/rag-from-scratch/
"""

from sentence_transformers import SentenceTransformer
import numpy as np
import anthropic  # pip install anthropic
from chunker import chunk_text


def build_rag_prompt(question, retrieved_chunks):
    """Assemble retrieved context into a grounded prompt."""
    context_block = "\n\n".join(
        f"[Document {i+1}]\n{chunk}"
        for i, (chunk, score) in enumerate(retrieved_chunks)
    )

    return f"""You are a helpful assistant that answers questions using
ONLY the provided context documents. Follow these rules strictly:

1. Answer based ONLY on the context below. Do not use prior knowledge.
2. If the context doesn't contain enough information, say:
   "I don't have enough information to answer this question."
3. Cite which document(s) you used, e.g. [Document 1].
4. Keep answers concise and direct.

Context:
{context_block}

Question: {question}

Answer:"""


class RAGPipeline:
    def __init__(self, embed_model="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(embed_model)
        self.chunks = []
        self.embeddings = None
        self.client = anthropic.Anthropic()  # uses ANTHROPIC_API_KEY env var

    def ingest(self, text, max_chars=1000, overlap=200):
        """Chunk and embed a document."""
        self.chunks = chunk_text(text, max_chars, overlap)
        self.embeddings = self.model.encode(
            self.chunks, normalize_embeddings=True
        )
        print(f"Ingested {len(self.chunks)} chunks")

    def query(self, question, k=3):
        """Retrieve relevant chunks and generate a grounded answer."""
        # Retrieve
        query_vec = self.model.encode(
            [question], normalize_embeddings=True
        )
        scores = (self.embeddings @ query_vec.T).squeeze()
        top_k = np.argsort(-scores)[:k]
        retrieved = [(self.chunks[i], float(scores[i])) for i in top_k]

        # Build prompt
        prompt = build_rag_prompt(question, retrieved)

        # Generate
        response = self.client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}]
        )

        return {
            "answer": response.content[0].text,
            "sources": retrieved,
        }


if __name__ == "__main__":
    solar_panel_guide = """Solar panels convert sunlight into electricity through the
photovoltaic effect. When photons hit silicon cells, they knock electrons
loose, creating an electrical current.

Installation requires careful roof assessment. South-facing roofs with
15-40 degree pitch are ideal in the northern hemisphere. Shading from
trees or neighboring buildings can reduce output by 10-25%.

A typical residential system is 6-10 kW, requiring 15-25 panels. At
average US electricity rates, payback period is 6-10 years. Federal tax
credits currently cover 30% of installation costs."""

    rag = RAGPipeline()
    rag.ingest(solar_panel_guide)

    # Query 1: Should find relevant context
    print("=" * 60)
    print("Query: What tax incentives are available for solar panels?")
    print("=" * 60)
    result = rag.query("What tax incentives are available for solar panels?")
    print(result["answer"])
    print()

    # Query 2: Should refuse (no brand info in context)
    print("=" * 60)
    print("Query: What's the best brand of solar panel?")
    print("=" * 60)
    result = rag.query("What's the best brand of solar panel?")
    print(result["answer"])
