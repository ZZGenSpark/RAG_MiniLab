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

Reads `source/policy.md`, creates six structural chunks, embeds each section, and upserts text + vector + metadata into the `expense_policy` Chroma collection (cosine space). Re-running upserts the same stable IDs and drops sections that are no longer in the file.

Source documents live in `source/`. Ingest currently loads the expense policy at `source/policy.md`.

Optional: `python ingest.py --policy source/policy.md --chroma-path /path/to/chroma`

## Ask

```bash
python ask.py "How much can I spend on food each day?"
```

Prints JSON: `answer`, `citation` (or `null` when the policy does not answer), and up to three `retrieved_chunks` with numeric cosine distances, smallest first.

## Tests and saved output

```bash
pytest
python -m rag.eval
```

`pytest` covers chunking, Chroma storage, retrieval, grounded generation, and the six required questions. `python -m rag.eval` writes `outputs/required_questions.json`.

## Chroma persistence

Chunks live on disk under `CHROMA_PATH` (default `<repo>/chroma_db`; `/app/chroma_db` in Docker). The directory is gitignored. Compose mounts it as the `chroma_data` volume. Schema: [rag/schema.py](rag/schema.py) (`expense_policy`, `hnsw:space=cosine`).

## Practice updates

See [docs/practice-updates.md](docs/practice-updates.md).
