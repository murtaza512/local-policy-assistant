# RAG Project

A small Retrieval-Augmented Generation (RAG) demo that answers questions about company policies using local embeddings and a local language model.

## How it works

1. **Load documents** — Policy text is read from `data/policies.txt`.
2. **Chunk by heading** — Sections are split on `#` headings (e.g. "Enterprise Refund Policy").
3. **Embed chunks** — Each section is encoded with `all-MiniLM-L6-v2` via [sentence-transformers](https://www.sbert.net/).
4. **Retrieve** — A user question is embedded and compared to chunk embeddings using cosine similarity. The top `k` matches are returned.
5. **Filter** — Results below a similarity threshold (`min_score=0.5`) are dropped to reduce hallucinations.
6. **Generate** — Retrieved context is passed to `HuggingFaceTB/SmolLM2-1.7B-Instruct`, which produces a grounded answer or falls back to: *"I don't know based on the provided company policies."*

## Project structure

```
local-policy-assistant/
├── data/
│   └── policies.txt      # Sample company policy document
├── rag/
│   ├── v1.py             # Basic RAG (in-memory retrieval + generation)
│   ├── v2.py             # RAG with heading-based chunks and file storage
│   └── v3_chroma.py      # RAG with ChromaDB vector store and metadata filters
├── storage/              # Cached chunks and embeddings (v2)
├── chroma_db/            # ChromaDB persistence (v3)
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.9+
- ~4 GB disk space for model downloads (first run)
- Apple Silicon Mac, CUDA GPU, or CPU (slower on CPU)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On first run, Hugging Face will download:

- `sentence-transformers/all-MiniLM-L6-v2` (embeddings)
- `HuggingFaceTB/SmolLM2-1.7B-Instruct` (text generation)

## Usage

Run a version from the project root:

```bash
python rag/v1.py          # basic in-memory RAG
python rag/v2.py          # file-backed chunks + embeddings
python rag/v3_chroma.py   # ChromaDB with metadata filtering
```

The script demonstrates retrieval, prompt building, score filtering, and final answers for sample questions such as:

- *What is the refund processing period for Enterprise customers in Germany?*
- *What is Standard Refund Policy?*
- *What is company's maternity policy?*

The main entry point for Q&A is `answer_question()`:

```python
result = answer_question("What is the refund processing period for Enterprise customers in Germany?")
print(result["answer"])
print(result["sources"])
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `k` | `2` | Number of chunks to retrieve |
| `min_score` | `0.5` | Minimum cosine similarity to include a chunk |
| `chunk_size` / `overlap` | `30` / `5` | Used by `chunk_text()` (word-based splitting; not used in the main flow) |

To use your own documents, replace or extend `data/policies.txt`. Use `# Heading` lines to define sections.

## Models

| Component | Model |
|-----------|-------|
| Embeddings | `all-MiniLM-L6-v2` |
| Generation | `HuggingFaceTB/SmolLM2-1.7B-Instruct` |
