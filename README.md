# RAG-Scope

**A hybrid Retrieval-Augmented Generation (RAG) pipeline for PDF question answering and retrieval evaluation.**

RAG-Scope combines dense vector search and keyword-based retrieval to find relevant document content, generate answers using Groq, and evaluate retrieval performance with a benchmark.

## Features

- PDF loading and text chunking
- Dense retrieval using Sentence Transformers and FAISS
- BM25 keyword retrieval with Reciprocal Rank Fusion (RRF)
- LLM-based question answering with source citations
- Benchmark evaluation for answer matching, answerability, and page-level retrieval

## Tech Stack

Python · Streamlit · FAISS · BM25 · Sentence Transformers · Groq · Pytest

## Setup

```bash
git clone https://github.com/bmanasareddy05/rag-scope.git
cd rag-scope

python -m venv .venv
```

Activate the environment on Windows:

```cmd
.venv\Scripts\activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Create a `.env` file in the project root with your Groq credentials:

```dotenv
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
```

Place your PDF at `data/documents/sec_fill.pdf`. The source PDF is not included in this repository.

## Run

Start the application:

```bash
streamlit run app.py
```

Run the tests:

```bash
python -m pytest -q
```

Run the evaluation benchmark:

```bash
python -m src.evaluation.benchmark_runner
```

## Baseline Evaluation

Evaluated on a 30-question benchmark using an SEC filing:

| Metric | Result |
|---|---:|
| Page Hit@5 | 84.62% |
| Page Recall@5 | 76.92% |
| Exact match (answerable questions) | 38.46% |
| Answer containment (answerable generations) | 50.00% |
| Generation success rate | 100% |

These are baseline results on one document and a small benchmark, not general-purpose accuracy claims. Page-level retrieval metrics measure whether relevant pages appear in the retrieved results; exact match and answer containment are limited text-overlap measures.

## Tests

**17 tests passed** with `python -m pytest -q`.

## Repository

[GitHub: bmanasareddy05/rag-scope](https://github.com/bmanasareddy05/rag-scope)
