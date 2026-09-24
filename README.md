# Mini RAG Lab

Grounded expense-policy assistant. Policy text is split into six section chunks, embedded with Ollama, stored in Chroma, and answered only from retrieved excerpts.

## Prerequisites

- Python 3.12 and the packages in `requirements.txt`
- Ollama running with `nomic-embed-text:v1.5` and `qwen3:8b-q4_K_M`
- `cp .env.example .env` if `.env` is missing

In the Docker/devcontainer setup, Ollama is reached at `http://host.docker.internal:11434` and Chroma persists at `/app/chroma_db`.

## Ingest

```bash
python ingest.py
```

Reads the markdown files in `source/policies/`, chunks each numbered section, and upserts text + vector + metadata into the `expense_policy` Chroma collection (cosine space). Re-running upserts the same stable IDs and drops sections that are no longer in those files.

Optional: `python ingest.py --policies source/policies --chroma-path /path/to/chroma`

## Minimal loop

```bash
python scripts/minimal_loop.py
```

Embeds two known texts with Ollama, stores them in a temporary Chroma directory, and prints which text the question retrieved. This runs before the full corpus ingest. `pytest` covers the same loop with a fake embedder.

## Ask

```bash
python ask.py "How much can I spend on food each day?"
```

Prints JSON: `answer`, `citation` (or `null` when the policy does not answer), and up to three `retrieved_chunks` with numeric cosine distances, smallest first.

## Tests and saved output

```bash
ruff check .
ruff format --check .
mypy
pytest
python -m rag.eval
```

GitHub Actions runs ruff, mypy, and pytest on Python 3.12. That job does not call Ollama or TypeSafe. `pytest` covers chunking, Chroma storage, retrieval, grounded generation, and the six required questions. `python -m rag.eval` writes `outputs/required_questions.json`.

## Chroma persistence

Chunks live on disk under `CHROMA_PATH` (default `<repo>/chroma_db`; `/app/chroma_db` in Docker). The directory is gitignored. Compose mounts it as the `chroma_data` volume. Schema: [rag/schema.py](rag/schema.py) (`expense_policy`, `hnsw:space=cosine`).

## Practice updates

See [docs/practice-updates.md](docs/practice-updates.md).
