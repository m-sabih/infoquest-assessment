# InfoQuest Expert Search Copilot — Backend API

## Prerequisites

- Python 3.11+ (project tested with 3.12 in conda env `infoquest`)
- Network access to the provided PostgreSQL host and to `https://openrouter.ai`

## Setup

```bash
conda activate infoquest
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env`:

- `DATABASE_URL` — PostgreSQL connection string (see credentials provided separately).
- `OPENROUTER_API_KEY` — required for `POST /ingest` unless a key file is used (see below).

Optional: `CHROMA_PATH`, `EMBEDDING_MODEL`, `CANDIDATE_PAGE_SIZE`, `EMBEDDING_BATCH_SIZE`, `MAX_PROFILE_CHARS`.

## Run the API

```bash
conda activate infoquest
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Open interactive docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

**Production note:** Use **one** Uvicorn worker with the default Chroma persistent client so only one process writes the on-disk index (avoid file contention).

## Project layout

| Path | Role |
|------|------|
| `app/main.py` | FastAPI app factory, lifespan (DB pool + vector store) |
| `app/config.py` | `pydantic-settings` from environment |
| `app/api/routes/` | `health`, `ingest` routers |
| `app/schemas/` | Pydantic request/response models |
| `app/services/candidate_repository.py` | Paginated Postgres reads + denormalized profile fields |
| `app/services/profile_builder.py` | Text document + Chroma metadata per candidate |
| `app/services/embeddings_client.py` | OpenRouter `POST /embeddings` |
| `app/services/vector_store.py` | Chroma persistent collection wrapper |
| `app/services/ingestion.py` | Orchestrates fetch → embed → upsert |

## Design (Part 1)

### Profile text and chunking

- **One embedding per candidate.** The database is relational (candidates, work experience, education, skills, languages). A SQL query aggregates related rows into text fields, then the app builds a **single ordered document** per candidate: identity and headline, location/nationality, overall years of experience, skills, languages, work history (title, company, industry, dates, truncated role descriptions), then education.
- **Rationale:** Expert search queries are usually holistic (“regulatory affairs in pharma in the Middle East”). A whole-profile vector typically matches better than many tiny chunks per person, and it keeps storage at ~1× the number of candidates (~25k vectors). Role-level chunks would multiply vectors and complicate deduplication in results.
- **Length cap:** `MAX_PROFILE_CHARS` (default 24k) truncates the document if needed; metadata includes `text_truncated` so consumers know the tail was cut (rare given per-role description limits in SQL).

### Embeddings provider and model

- Embeddings go through **OpenRouter** (`POST {OPENROUTER_BASE_URL}/embeddings`) using the API key from the environment, default model **`openai/text-embedding-3-small`** (override with `EMBEDDING_MODEL`).
- **Rationale:** Strong general-purpose quality, 1536 dimensions, widely documented, and easy to swap to another OpenRouter embedding model via configuration without code changes.

### Vector database choice

- **ChromaDB** with `PersistentClient` under `CHROMA_PATH` (default `./data/chroma`).
- **Rationale:** No separate server to run for local or minimal deployments, simple Python API, metadata stored alongside vectors for rich hits in Part 2, and cosine space configured on the collection for semantic search.

### Ingestion API

- `POST /ingest` accepts `replace_collection` (default `true`) and optional `limit` for testing. It processes candidates in pages (`CANDIDATE_PAGE_SIZE`), embeds in batches (`EMBEDDING_BATCH_SIZE`), and upserts into Chroma by candidate id.
