*This project has been created as part of the 42 curriculum by mawelsch*

# Description

**A retrieval-augmented generation (RAG) system that combines lexical search with large language model integration.** This project implements a complete RAG pipeline using **Qwen/Qwen0.5B** as the LLM backbone and **BM25** for semantic document retrieval. The system enables users to index documents, retrieve relevant passages, and generate contextually grounded answers to natural language queries.

---

## Installation

### Prerequisites
- **Python 3.11+**
- **uv** (optional, but recommended for dependency management)

### Setup

Choose one of the following installation methods:

Using the Makefile:
```
make install
```

Or, if uv is already installed:
```
uv sync
```

---

## Running the Project

Execute the application using either method below:

Using the Makefile:
```
make run {arguments}
```

Or directly with uv and Python:
```
uv run python3 -m src {arguments}
```

---

## Commands

The system provides **six core commands** for indexing, searching, and answering queries:

| Command | Purpose | Input | Output |
|---------|---------|-------|--------|
| **index** | Ingest documents from `data/raw/` and subdirectories, chunk them, and build a searchable index | optional `--max_chunk_size` parameter | Index saved to `data/processed/*.pkl`; chunk metadata to `data/chunks/` |
| **search** | Retrieve top-k document chunks matching a query using BM25 lexical search | Query string; optional `--k` parameter | Ranked list of relevant chunks with metadata |
| **search_dataset** | Run batch search on a dataset of queries | Dataset file path; optional `--k`, `--save_directory` | Search results saved to file |
| **answer** | Generate an LLM-powered answer by retrieving relevant context and conditioning the model | Query string; optional `--k` parameter | Generated answer with source attribution |
| **answer_dataset** | Run batch answer generation on a dataset of queries | Dataset file path; optional `--k`, `--save_directory` | Answers saved to file with retrieval sources |
| **evaluate** | Compute recall@k by comparing system results against ground truth sources | Two dataset paths; optional `--k` parameter | Recall@k score with intersection-over-union (IoU ≥ 5%) validation |

---

## System Architecture

The system is organized into three primary components:

### RagCLI
**User-facing interface** that handles command parsing and orchestrates the data pipeline. Translates high-level operations into lower-level system calls.

### RagAgainstTheMachine
**Core orchestrator** responsible for end-to-end functionality. Coordinates document indexing, retrieval, and answer generation across all subsystems.

### Chunker
**Document processing engine** that implements multiple chunking strategies to decompose documents into semantically coherent units suitable for indexing and retrieval.

---

## Chunking Strategy

The system implements **two distinct chunking approaches** to preserve semantic coherence:

**Python Abstract Syntax Tree (AST) Chunking**  
Analyzes Python source code structurally to keep functions, classes, and logical code blocks intact. This ensures that related code remains unified during retrieval, improving answer relevance for code-based queries.

**Recursive Text Chunking**  
Decomposes natural language documents hierarchically by progressively splitting on sentence boundaries, paragraph breaks, and structural markers. This approach maintains narrative flow and context across chunks.

Both strategies are configurable via the `--max_chunk_size` parameter (default: **2000 tokens**).

---

## Retrieval Method

**BM25 Lexical Search** is the primary retrieval mechanism. BM25 (Best Matching 25) is a probabilistic ranking function that scores document chunks based on term frequency and inverse document frequency (IDF). This approach was selected for its **simplicity of implementation**, **robust performance on diverse query types**, and **minimal computational overhead** compared to dense embedding-based methods.

The implementation leverages the **bm25s** library for efficient indexing and retrieval operations.

---

## Performance Analysis

Measured on standard hardware (CPU-based inference):

| Operation | Single Query | Notes |
|-----------|--------------|-------|
| **Search** | ~**1 second** | BM25 index lookup and ranking |
| **Answer** | ~**37 seconds** | Includes search + LLM inference (CPU) |

Performance scales linearly with corpus size and query complexity. GPU acceleration can significantly reduce answer generation time.

---

## Design Decisions

**Chunking Strategy**  
Python code is chunked using AST analysis to preserve function and class boundaries, ensuring code fragments remain semantically cohesive. Text documents use recursive splitting with the same goal—keeping logical units together improves downstream retrieval quality.

**Lexical Search Over Dense Embeddings**  
BM25 was selected over embedding-based methods because it is **lightweight**, **deterministic**, and **easy to implement and debug**. It performs well on keyword-heavy queries without requiring embedding models or vector databases.

**Tokenization Decisions**  
Initial experiments applied stemming to both the corpus and queries; however, **stemming was removed** after revealing poor recall performance. The mismatch arose from stemming the indexed corpus but not the user queries, causing retrieval failures. Removing stemming entirely improved recall and simplified the pipeline.

---

## Challenges Faced

**Monotony and Motivation**  
Once the core RAG concept was understood and initial prototyping was complete, the project became repetitive—primarily an exercise in gluing together existing libraries rather than solving novel technical problems.

**Recall Optimization**  
A critical recall bottleneck emerged from inconsistent stemming: the corpus was stemmed during indexing, but user queries were not stemmed before retrieval. This mismatch caused relevant documents to be missed. The solution—removing stemming entirely—was simple but required careful debugging to identify the root cause.

---

## Example Usage

**Index documents with a 2000-token chunk size:**
```
uv run python3 -m src index --max_chunk_size 2000
```

**Search for a query and retrieve the top 5 results:**
```
uv run python3 -m src search "What is machine learning?" --k 5
```

**Run batch search on a dataset and save results:**
```
uv run python3 -m src search_dataset data/queries.json --k 5 --save_directory output/search_results
```

**Generate an answer to a single query:**
```
uv run python3 -m src answer "Explain neural networks" --k 5
```

**Batch answer generation from a dataset:**
```
uv run python3 -m src answer_dataset data/queries.json --k 5 --save_directory output/answers
```

**Evaluate system performance against ground truth:**
```
uv run python3 -m src evaluate output/search_results data/ground_truth.json --k 5
```

**Note:** All options prefixed with `--` are optional and override default values.

---

## Resources

**Retrieval-Augmented Generation (Concept Learning)**  
[Retrieval-Augmented Generation — Wikipedia](https://en.wikipedia.org/wiki/Retrieval-augmented_generation)  
Foundational understanding of RAG architectures and use cases.

**Text Chunking Implementation**  
[LangChain Documentation](https://docs.langchain.com/)  
Extensively used for text splitting and document chunking utilities.

**BM25 Search Algorithm**  
[BM25S GitHub Repository](https://github.com/xhluca/bm25s)  
Core implementation library for the BM25 ranking function.

---

## AI Usage

Artificial intelligence was used **minimally** during development:

- **Point-level debugging:** AI assisted in resolving specific technical issues and edge cases.
- **Pre-project prototyping:** A proof-of-concept RAG system was constructed with AI assistance to validate the core approach before full implementation.
- **Documentation:** AI was used to structure and refine this README for clarity and completeness.

The majority of the system was built through manual implementation and iterative refinement.