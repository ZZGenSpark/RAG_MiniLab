# Mini RAG Lab

Grounded Q&A over company policy documents. Policy text is chunked by numbered rule, embedded with MiniLM, stored in Chroma, and answered only from retrieved excerpts.

## Prerequisites

- Python 3.12 and the packages in `requirements.txt`
- Ollama running with `qwen3:8b-q4_K_M` for answer generation
- `cp .env.example .env` if `.env` is missing

Embeddings and reranking run locally with sentence-transformers. In the Docker/devcontainer setup, Ollama is reached at `http://host.docker.internal:11434` and Chroma persists at `/app/chroma_db`.

## Ingest

```bash
python ingest.py
```

Reads only the markdown files in `source/policies/`, chunks each numbered rule, embeds them with MiniLM, and upserts text + vector + metadata into the `company_policies` Chroma collection (cosine space). Re-running upserts the same stable IDs and drops sections that are no longer in those files.

Optional: `python ingest.py --policies source/policies --chroma-path /path/to/chroma`

## Minimal loop

```bash
python scripts/minimal_loop.py
```

Embeds two known texts with MiniLM, stores them in a temporary Chroma directory, and prints which text the question retrieved. This runs before the full corpus ingest. `pytest` covers the same loop with a fake embedder.

## Ask

```bash
python ask.py "How long can an employee play foosball each day?"
```

Prints JSON: `answer`, `citations` (empty when the policy does not answer), and up to three `retrieved_chunks` with numeric cosine distances, smallest first.

## Tests and saved output

```bash
ruff check .
ruff format --check .
mypy
pytest
python -m rag.eval --retrieval-only
python -m rag.eval
```

GitHub Actions has two jobs on Python 3.12. `check` runs ruff, mypy, pytest, and `python -m rag.eval --retrieval-only`. That job does not call Ollama or TypeSafe. The retrieval command ingests the policies with MiniLM, retrieves every evaluation question with the section-code router and the cross-encoder, and fails if recall is below 1 or a question returns no chunks. It writes `outputs/retrieval_eval.json`. `pytest` covers the same recall check, plus chunking, Chroma storage, hybrid retrieval, reranking, and grounded generation.

`eval` starts after `check` passes. It installs Ollama, pulls `qwen3:8b-q4_K_M`, ingests the policies, and runs `python -m rag.eval`. That command asks Ollama, writes `outputs/eval_report.json`, and fails unless recall and answer accuracy are both 1.0. The first `eval` run is slow because it downloads the chat model; later runs restore `~/.ollama/models` from cache. Local `python -m rag.eval` uses an already-ingested Chroma directory and applies the same score gate.

## Chroma persistence

Chunks live on disk under `CHROMA_PATH` (default `<repo>/chroma_db`; `/app/chroma_db` in Docker). The directory is gitignored. Compose mounts it as the `chroma_data` volume. Schema: [rag/schema.py](rag/schema.py) (`company_policies`, `hnsw:space=cosine`).
