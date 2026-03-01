"""
Search quality evaluation: P@K, R@K, NDCG@K metrics with pipeline comparison.

Blog post: https://dadops.dev/blog/building-ai-search-engine/
Code Block 8.

Runs a synthetic evaluation to test BM25-only vs Hybrid vs Reranked pipeline stages.
Blog claims: P@5 0.52 → 0.64 → 0.78 on 200 docs / 20 queries.
We test with a smaller synthetic corpus to verify the directional improvement.
"""
import sqlite3
import struct
import time

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder


# ── Import pipeline functions (duplicated here for standalone execution) ──

def create_search_db(db_path=":memory:"):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks
        USING fts5(title, content, source)
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            chunk_id INTEGER PRIMARY KEY,
            vector BLOB
        )
    """)
    return conn


def chunk_document(text, chunk_size=256, overlap=64):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        start += chunk_size - overlap
    return chunks


def ingest(conn, documents, model):
    for doc in documents:
        chunks = chunk_document(doc["content"])
        for chunk in chunks:
            cursor = conn.execute(
                "INSERT INTO chunks(title, content, source) VALUES (?, ?, ?)",
                (doc["title"], chunk, doc["source"])
            )
            chunk_id = cursor.lastrowid
            embedding = model.encode(chunk)
            blob = struct.pack(f"{len(embedding)}f", *embedding)
            conn.execute(
                "INSERT INTO embeddings(chunk_id, vector) VALUES (?, ?)",
                (chunk_id, blob)
            )
    conn.commit()


def bm25_search(conn, query, top_k=20):
    results = conn.execute(
        """SELECT rowid, title, content, rank
           FROM chunks WHERE chunks MATCH ?
           ORDER BY rank LIMIT ?""",
        (query, top_k)
    ).fetchall()
    return [(r[0], r[1], r[2], -r[3]) for r in results]


def vector_search(conn, query, model, top_k=20):
    q_vec = model.encode(query)
    rows = conn.execute("SELECT chunk_id, vector FROM embeddings").fetchall()
    scores = []
    for chunk_id, blob in rows:
        dim = len(blob) // 4
        doc_vec = np.array(struct.unpack(f"{dim}f", blob))
        similarity = np.dot(q_vec, doc_vec) / (
            np.linalg.norm(q_vec) * np.linalg.norm(doc_vec) + 1e-8
        )
        scores.append((chunk_id, similarity))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def hybrid_search(conn, query, model, top_k=20, k=60):
    bm25_results = bm25_search(conn, query, top_k)
    vec_results = vector_search(conn, query, model, top_k)
    rrf_scores = {}
    for rank, (chunk_id, *_) in enumerate(bm25_results):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (k + rank + 1)
    for rank, (chunk_id, _) in enumerate(vec_results):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1 / (k + rank + 1)
    fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return fused[:top_k]


def search_pipeline(conn, query, model, reranker, top_k=10):
    candidates = hybrid_search(conn, query, model, top_k=20)
    docs = []
    for chunk_id, rrf_score in candidates:
        row = conn.execute(
            "SELECT title, content, source FROM chunks WHERE rowid = ?",
            (chunk_id,)
        ).fetchone()
        if row:
            docs.append({
                "id": chunk_id, "title": row[0],
                "content": row[1], "source": row[2],
                "retrieval_score": rrf_score
            })
    pairs = [(query, doc["content"]) for doc in docs]
    if not pairs:
        return []
    scores = reranker.predict(pairs)
    scored = sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
    return [{**doc, "rerank_score": float(score)} for doc, score in scored[:top_k]]


# ── Code Block 8: Search Quality Metrics ──

def precision_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    return len(set(retrieved_k) & set(relevant)) / k


def recall_at_k(retrieved, relevant, k):
    retrieved_k = retrieved[:k]
    return len(set(retrieved_k) & set(relevant)) / len(relevant)


def ndcg_at_k(retrieved, relevance_scores, k):
    dcg = sum(
        relevance_scores.get(doc_id, 0) / np.log2(i + 2)
        for i, doc_id in enumerate(retrieved[:k])
    )
    ideal = sorted(relevance_scores.values(), reverse=True)[:k]
    idcg = sum(
        score / np.log2(i + 2)
        for i, score in enumerate(ideal)
    )
    return dcg / idcg if idcg > 0 else 0


def evaluate_pipeline(conn, model, reranker, eval_set):
    """Evaluate BM25-only, Hybrid, and Reranked pipelines."""
    stages = {"BM25 only": [], "Hybrid": [], "Reranked": []}

    for query, relevant_ids in eval_set:
        bm25 = [r[0] for r in bm25_search(conn, query)]
        hybrid = [cid for cid, _ in hybrid_search(conn, query, model)]
        reranked = [r["id"] for r in search_pipeline(conn, query, model, reranker)]

        for name, results in [
            ("BM25 only", bm25), ("Hybrid", hybrid), ("Reranked", reranked)
        ]:
            stages[name].append(precision_at_k(results, relevant_ids, k=5))

    for name, scores in stages.items():
        print(f"{name:12s}  P@5 = {np.mean(scores):.3f}")

    return {name: np.mean(scores) for name, scores in stages.items()}


# ── Synthetic test corpus ──
# We build a corpus of 30 topically-diverse documents with labeled relevant IDs per query.
# This tests whether hybrid > BM25 and reranked > hybrid directionally.

CORPUS = [
    # Topic: Neural Networks / Deep Learning
    {"title": "Neural Network Fundamentals",
     "content": "Neural networks consist of layers of artificial neurons that process information. Each neuron applies a weighted sum followed by a nonlinear activation function. Deep networks with many hidden layers can learn hierarchical features. Training uses backpropagation to compute gradients and gradient descent to update weights. The universal approximation theorem proves a sufficiently wide network can approximate any continuous function.",
     "source": "neural-nets-101.html"},
    {"title": "Backpropagation Algorithm",
     "content": "Backpropagation computes the gradient of the loss function with respect to each weight by applying the chain rule. The forward pass computes activations layer by layer. The backward pass propagates error signals from the output layer back through the network. Automatic differentiation frameworks like PyTorch implement backpropagation efficiently using computational graphs. Gradient clipping prevents exploding gradients in deep networks.",
     "source": "backprop.html"},
    {"title": "Convolutional Neural Networks",
     "content": "CNNs use convolutional filters to detect local patterns in images. Pooling layers reduce spatial dimensions while retaining important features. Modern architectures like ResNet use skip connections to train very deep networks. Data augmentation improves generalization by creating transformed versions of training images. Transfer learning allows fine-tuning pre-trained CNNs for new tasks with limited data.",
     "source": "cnns.html"},
    {"title": "Recurrent Neural Networks and LSTMs",
     "content": "RNNs process sequential data by maintaining a hidden state across timesteps. Vanilla RNNs suffer from vanishing gradients on long sequences. LSTMs solve this with gating mechanisms: input gate, forget gate, and output gate. GRUs simplify the LSTM architecture while achieving similar performance. Bidirectional RNNs process sequences in both directions for better context understanding.",
     "source": "rnns.html"},
    {"title": "Training Deep Networks Effectively",
     "content": "Learning rate scheduling reduces the step size during training for better convergence. Batch normalization stabilizes training by normalizing activations. Dropout regularization prevents overfitting by randomly zeroing neurons. Weight initialization schemes like Xavier and He initialization help gradients flow. Mixed precision training uses FP16 to speed up computation while maintaining accuracy.",
     "source": "training-tricks.html"},

    # Topic: Natural Language Processing
    {"title": "Transformer Architecture Deep Dive",
     "content": "The transformer architecture uses self-attention to process all tokens in parallel. Scaled dot-product attention computes QKV matrices from the input embeddings. Multi-head attention attends to different representation subspaces simultaneously. Positional encodings inject sequence order information. The encoder processes the input and the decoder generates the output autoregressively. Layer normalization and residual connections stabilize training.",
     "source": "transformers.html"},
    {"title": "Word Embeddings and Representations",
     "content": "Word embeddings map words to dense vector representations capturing semantic relationships. Word2Vec learns embeddings through skip-gram or CBOW objectives. GloVe combines global matrix factorization with local context windows. Subword tokenization like BPE handles out-of-vocabulary words. Contextual embeddings from BERT and GPT produce different vectors for the same word based on context.",
     "source": "embeddings.html"},
    {"title": "Text Classification with Transformers",
     "content": "Fine-tuning pre-trained transformers achieves state-of-the-art text classification. BERT uses a classification token for sentence-level predictions. Sequence length limits require truncation or chunking strategies. Few-shot learning with large language models can classify text without fine-tuning. Prompt engineering designs input formats that elicit desired classification behavior from LLMs.",
     "source": "text-classification.html"},
    {"title": "Named Entity Recognition",
     "content": "NER identifies and classifies named entities like persons, organizations, and locations in text. Token classification with BERT assigns entity labels to each subword token. BIO tagging scheme marks begin, inside, and outside tokens for each entity type. CRF layers on top of transformers improve sequence labeling consistency. Domain-specific NER requires annotated training data from the target domain.",
     "source": "ner.html"},
    {"title": "Machine Translation Systems",
     "content": "Neural machine translation uses encoder-decoder architectures to translate between languages. Attention mechanisms allow the decoder to focus on relevant source tokens. Beam search explores multiple translation hypotheses to find better outputs. Back-translation augments training data by translating monolingual text. Multilingual models like mBART handle dozens of language pairs with a single model.",
     "source": "translation.html"},

    # Topic: Search and Information Retrieval
    {"title": "BM25 and Classic Information Retrieval",
     "content": "BM25 ranks documents by term frequency and inverse document frequency with length normalization. The algorithm extends TF-IDF with saturation of term frequency to prevent long documents from dominating. Query expansion adds related terms to improve recall. Inverted indices enable fast lookup of documents containing specific terms. Relevance feedback uses user clicks to refine search results.",
     "source": "bm25.html"},
    {"title": "Semantic Search with Dense Retrieval",
     "content": "Dense retrieval encodes queries and documents into embedding vectors for semantic matching. Bi-encoder models like DPR encode queries and documents independently for fast indexing. Contrastive learning trains retrievers to place relevant pairs close in embedding space. Hard negative mining improves retriever training by using challenging non-relevant examples. Approximate nearest neighbor search with FAISS enables efficient similarity lookup.",
     "source": "dense-retrieval.html"},
    {"title": "Hybrid Search Combining BM25 and Vectors",
     "content": "Hybrid search merges keyword-based BM25 results with semantic vector search results. Reciprocal Rank Fusion combines ranked lists by summing inverse ranks across methods. BM25 excels at exact keyword matching while vectors capture semantic similarity. The fusion constant k controls the balance between early and late ranked documents. Hybrid search consistently outperforms either method alone on diverse query types.",
     "source": "hybrid-search.html"},
    {"title": "Cross-Encoder Reranking for Search",
     "content": "Cross-encoders jointly encode the query and document for more accurate relevance scoring. Unlike bi-encoders, cross-encoders can capture fine-grained interactions between query and document tokens. The tradeoff is speed: cross-encoders are too slow for first-stage retrieval but excel at reranking top candidates. Models like MS MARCO MiniLM achieve high accuracy on passage ranking benchmarks. Two-stage retrieve-then-rerank pipelines combine the speed of bi-encoders with the accuracy of cross-encoders.",
     "source": "reranking.html"},
    {"title": "Building Production Search Systems",
     "content": "Production search systems require indexing, caching, and load balancing for scalability. Elasticsearch and Vespa provide distributed search infrastructure. Query understanding parses user intent and extracts filters from natural language queries. A/B testing measures the impact of search quality changes on user behavior. Monitoring search metrics like click-through rate and mean reciprocal rank tracks system health.",
     "source": "production-search.html"},

    # Topic: Python Programming
    {"title": "Python Concurrency and Parallelism",
     "content": "Python threading handles I/O-bound tasks but the GIL prevents true CPU parallelism. The multiprocessing module creates separate processes with their own memory space. Asyncio provides cooperative multitasking for high-concurrency I/O operations. Process pools distribute CPU-intensive work across multiple cores. Concurrent.futures provides a unified interface for both thread and process pools.",
     "source": "python-concurrency.html"},
    {"title": "Python Type Hints and Static Analysis",
     "content": "Type hints improve code readability and enable static analysis with tools like mypy. Generic types parameterize containers with element types. Protocol classes define structural subtyping for duck-typed interfaces. TypeGuard narrows types in conditional branches. Runtime type checking with Pydantic validates data at system boundaries. Gradual typing allows mixing typed and untyped code in the same project.",
     "source": "python-types.html"},
    {"title": "Testing Python Applications",
     "content": "Pytest provides a flexible framework for writing and running Python tests. Fixtures manage test setup and teardown with dependency injection. Parametrize decorates tests to run with multiple input combinations. Mock objects replace dependencies for isolated unit testing. Coverage tools measure which code paths are exercised by tests. Property-based testing with Hypothesis generates random test inputs to find edge cases.",
     "source": "python-testing.html"},
    {"title": "Python Performance Optimization",
     "content": "Profiling with cProfile identifies performance bottlenecks in Python code. NumPy vectorization replaces Python loops with optimized C operations. Caching with functools.lru_cache avoids recomputing expensive results. Generators reduce memory usage by yielding values lazily. Cython compiles Python-like code to C for significant speedups. Line profiler shows time spent on each line of code.",
     "source": "python-perf.html"},
    {"title": "Python Data Structures and Algorithms",
     "content": "Lists provide O(1) append and O(n) insert. Dictionaries use hash tables for O(1) average lookup. Sets enable fast membership testing and intersection operations. Heaps from the heapq module implement priority queues. Collections module provides specialized containers like defaultdict, Counter, and deque. Sorting with custom key functions enables flexible ordering of complex objects.",
     "source": "python-ds.html"},

    # Topic: Machine Learning Operations
    {"title": "ML Model Deployment Strategies",
     "content": "Model serving frameworks like TorchServe and TF Serving handle inference requests. Containerization with Docker ensures reproducible deployment environments. Batch inference processes large datasets offline for efficiency. Online inference handles real-time prediction requests with low latency. Model versioning tracks which model version is serving in production. Canary deployments gradually roll out new models to detect issues early.",
     "source": "ml-deployment.html"},
    {"title": "Feature Engineering for Machine Learning",
     "content": "Feature scaling normalizes input ranges for algorithms sensitive to magnitude. One-hot encoding converts categorical variables to binary vectors. Feature interaction terms capture nonlinear relationships between variables. Time-based features extract day, month, and seasonal patterns from timestamps. Text features include TF-IDF vectors, embedding averages, and character n-grams. Feature selection removes irrelevant variables to reduce overfitting and improve training speed.",
     "source": "feature-engineering.html"},
    {"title": "Experiment Tracking with MLflow",
     "content": "MLflow tracks experiments by logging parameters, metrics, and artifacts. The tracking server stores experiment data in a backend database. Autologging captures model parameters and training metrics automatically. Model registry manages model lifecycle from staging to production. MLflow projects package code for reproducible runs across environments. Comparison views show metric trends across experiment runs.",
     "source": "mlflow.html"},
    {"title": "Data Pipeline Orchestration",
     "content": "Apache Airflow orchestrates complex data workflows as directed acyclic graphs. Tasks define individual pipeline steps with dependencies. Sensors wait for external conditions before triggering downstream tasks. XCom enables communication between tasks in the same DAG run. Scheduling expressions control pipeline execution frequency. Error handling with retries and alerts ensures pipeline reliability.",
     "source": "data-pipelines.html"},
    {"title": "Monitoring ML Models in Production",
     "content": "Data drift detection compares incoming feature distributions to training data. Model performance monitoring tracks prediction accuracy over time. Concept drift occurs when the relationship between features and targets changes. Statistical tests like KS test and PSI quantify distribution shifts. Alert thresholds trigger retraining when performance degrades. Shadow mode compares new model predictions against the current production model.",
     "source": "ml-monitoring.html"},

    # Topic: Miscellaneous / Distractor docs
    {"title": "Database Indexing Strategies",
     "content": "B-tree indices accelerate range queries and ordered access patterns. Hash indices provide O(1) lookup for equality queries. Composite indices cover multiple columns for complex query patterns. Partial indices only index rows matching a filter condition. Index maintenance adds overhead to write operations. Query planners choose which indices to use based on estimated selectivity.",
     "source": "db-indexing.html"},
    {"title": "REST API Design Best Practices",
     "content": "REST APIs use HTTP methods to express CRUD operations on resources. Consistent naming conventions improve API discoverability. Pagination handles large result sets with cursor or offset-based approaches. Rate limiting protects services from excessive request volumes. Versioning strategies include URL path, header, and query parameter approaches. OpenAPI specifications document endpoints, parameters, and response schemas.",
     "source": "rest-api.html"},
    {"title": "Container Orchestration with Kubernetes",
     "content": "Kubernetes manages containerized application deployment and scaling. Pods are the smallest deployable units containing one or more containers. Services provide stable network endpoints for pod groups. Deployments manage rolling updates and rollback capabilities. Horizontal pod autoscaler adjusts replica count based on metrics. ConfigMaps and Secrets separate configuration from application code.",
     "source": "kubernetes.html"},
    {"title": "Git Workflow Best Practices",
     "content": "Feature branches isolate development work from the main branch. Pull requests enable code review before merging changes. Rebasing creates a linear history by replaying commits. Conventional commits standardize commit message formats. Git hooks automate checks before commits and pushes. Cherry-picking applies specific commits to other branches.",
     "source": "git-workflow.html"},
    {"title": "Web Security Fundamentals",
     "content": "Cross-site scripting XSS injects malicious scripts into web pages. Content security policy headers restrict which scripts can execute. SQL injection exploits unsanitized database queries. Parameterized queries prevent injection by separating code from data. CSRF tokens validate that form submissions originate from the application. HTTPS encrypts data in transit between client and server.",
     "source": "web-security.html"},
]


# Queries with relevant document IDs (hand-labeled for evaluation).
# IDs are 1-indexed since SQLite rowid starts at 1.
EVAL_SET = [
    # Neural network queries — should match NN docs
    ("how do neural networks learn through backpropagation",
     [1, 2, 5]),  # NN fundamentals, backprop, training tricks
    ("deep learning training techniques and optimization",
     [2, 5, 1]),  # backprop, training tricks, NN fundamentals
    ("convolutional neural networks for image recognition",
     [3, 1]),  # CNNs, NN fundamentals
    ("recurrent networks and sequence modeling",
     [4, 6]),  # RNNs, transformers

    # NLP queries — should match NLP docs
    ("transformer self-attention mechanism",
     [6, 7]),  # transformers, embeddings
    ("word embeddings and semantic representations",
     [7, 12]),  # embeddings, dense retrieval
    ("text classification using pre-trained language models",
     [8, 6]),  # text classification, transformers
    ("named entity recognition with BERT",
     [9, 8]),  # NER, text classification

    # Search/IR queries — BM25 should struggle with semantic queries
    ("how to build a search engine with ranking",
     [11, 13, 14, 15]),  # BM25, hybrid, reranking, production search
    ("semantic similarity matching for documents",
     [12, 13, 7]),  # dense retrieval, hybrid search, embeddings
    ("combining keyword and vector search",
     [13, 11, 12]),  # hybrid search, BM25, dense retrieval
    ("improving search result quality with reranking",
     [14, 13, 15]),  # reranking, hybrid, production search

    # Python queries
    ("python async concurrent programming patterns",
     [16, 19]),  # concurrency, performance
    ("testing python code with pytest",
     [18, 17]),  # testing, type hints

    # MLOps queries — semantic matching important here
    ("deploying machine learning models to production",
     [21, 25]),  # deployment, monitoring
    ("tracking machine learning experiments",
     [23, 24]),  # mlflow, data pipelines

    # Cross-topic queries where vector search should help
    ("optimizing model training speed",
     [5, 19, 2]),  # training tricks, python perf, backprop
    ("securing web applications against injection attacks",
     [30, 27]),  # web security, REST API

    # Broad queries
    ("machine learning model lifecycle management",
     [21, 23, 25]),  # deployment, mlflow, monitoring
    ("python best practices for data science",
     [16, 17, 18, 19, 20]),  # all python docs
]


if __name__ == "__main__":
    print("=== Search Quality Evaluation ===\n")

    print("Loading models...")
    t0 = time.perf_counter()
    bi_encoder = SentenceTransformer("all-MiniLM-L6-v2")
    cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    t1 = time.perf_counter()
    print(f"Models loaded in {t1-t0:.1f}s\n")

    print(f"Ingesting {len(CORPUS)} documents...")
    conn = create_search_db()
    ingest(conn, CORPUS, bi_encoder)
    t2 = time.perf_counter()
    print(f"Ingestion complete in {t2-t1:.1f}s\n")

    print(f"Running {len(EVAL_SET)} evaluation queries...\n")
    results = evaluate_pipeline(conn, bi_encoder, cross_encoder, EVAL_SET)

    print()
    print("Blog claims: BM25 P@5=0.52, Hybrid P@5=0.64, Reranked P@5=0.78")
    print("(Blog used 200 docs / 20 queries; we used 30 docs / 20 queries)")
    print()

    # Check directional correctness
    bm25_p5 = results["BM25 only"]
    hybrid_p5 = results["Hybrid"]
    reranked_p5 = results["Reranked"]

    print("Directional checks:")
    print(f"  Hybrid > BM25:    {'PASS' if hybrid_p5 > bm25_p5 else 'FAIL'} ({hybrid_p5:.3f} vs {bm25_p5:.3f})")
    print(f"  Reranked > Hybrid: {'PASS' if reranked_p5 > hybrid_p5 else 'FAIL'} ({reranked_p5:.3f} vs {hybrid_p5:.3f})")

    print("\nAll evaluation tests completed!")
