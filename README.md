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
- `OPENROUTER_API_KEY` — required for `POST /ingest` and `POST /chat`.

Optional: `CHROMA_PATH`, `EMBEDDING_MODEL`, `CHAT_MODEL`, `CANDIDATE_PAGE_SIZE`, `EMBEDDING_BATCH_SIZE`, `MAX_PROFILE_CHARS`.
Optional: `CHAT_CHECKPOINT_PATH` (LangGraph SQLite checkpointer file).

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
| `app/main.py` | FastAPI app factory, lifespan (DB pool + vector store + LangGraph) |
| `app/config.py` | `pydantic-settings` from environment |
| `app/api/routes/` | `health`, `ingest`, `chat`, `candidates` routers |
| `app/schemas/` | Pydantic request/response models |
| `app/services/candidate_repository.py` | Paginated Postgres reads + denormalized profile fields |
| `app/services/profile_builder.py` | Text document + Chroma metadata per candidate |
| `app/services/embeddings_client.py` | OpenRouter `POST /embeddings` |
| `app/services/chat_client.py` | OpenRouter `POST /chat/completions` (JSON completions) |
| `app/services/vector_store.py` | Chroma persistent collection wrapper |
| `app/services/ingestion.py` | Orchestrates fetch → embed → upsert |
| `app/services/chat_logic.py` | Pure functions: query rewrite, filter extraction, vector search, LLM re-rank |
| `app/services/chat_graph.py` | LangGraph pipeline wiring all chat_logic steps together |

---

## Design decisions

### Part 1 — Ingestion

#### One embedding per candidate

The database is relational (candidates, work experience, education, skills, languages). A SQL query denormalises all related rows into text fields. From those, `profile_builder.py` assembles a **single ordered document** per candidate — one vector per person, not many chunks.

**Rationale:** Expert search queries are holistic. A whole-profile vector captures that holistic intent far better than isolated role-level chunks, and it keeps the index at exactly ~1× the number of candidates (~25k vectors).

#### Profile document structure and field ordering

- We build the profile text in a fixed order so embeddings stay consistent.
- We put **role + skills early** so they influence similarity the most:

1. Headline (candidate's own summary — highest signal)
2. Primary roles (derived from work history titles)
3. Industries (derived from company industry fields)
4. Seniority (rule-based, derived from titles)
5. Years of experience
6. Skills
7. Full work experience with descriptions
8. Education
9. Languages
10. Geographic background

- **No name**: names don’t help role/skill matching and can add noise.
- **Geo at the end**: still searchable, but doesn’t drown out skills/roles.
- **Derived fields at the top**: makes the key signals easier to match.

#### Embedding model: `openai/text-embedding-3-large`

- **Model:** `openai/text-embedding-3-large` (configurable via `EMBEDDING_MODEL`), accessed through OpenRouter.
- **Dimensions:** 3072 — measurably higher retrieval quality than `-small` (1536d) on MTEB benchmarks.
- **Considered alternative:** `cohere/embed-multilingual-v3.0` (great multilingual retrieval), but it **was not available as an OpenRouter embeddings model** for this service at the time, so we stayed with `text-embedding-3-large`.

#### Vector database: ChromaDB with `PersistentClient`

- **Why Chroma:** No separate server process needed for a local deployment; simple Python API; stores metadata alongside vectors (used for `where` filtering); supports cosine space natively.
- **Cosine similarity:** Cosine similarity is orientation-based, making it robust to document length variation (a short profile and a long profile can still be very similar if they share the same semantic direction).

---

### Part 2 — Conversational search

#### Deterministic LangGraph pipeline instead of an LLM-with-tools agent

The `/chat` endpoint is implemented as a **fixed, deterministic node graph** (LangGraph `StateGraph`) rather than a free-form ReAct agent that picks tools autonomously.

The graph has five fixed steps:

```
START → rewrite → retrieve → [retry?] → format_base → explain → END
```

**Why**:
- **Predictable**: same steps every time
- **Bounded cost/latency**: no agent loops
- **Easier to debug**: clear logs per step

#### Query rewrite (`rewrite_node`)

- We rewrite the message into a short “search query” before embedding.
- **Why**: removes filler words and keeps only what matters (role, skills, industry, location).
- For follow-ups, we merge the previous query + the new message into one clear query.

#### Metadata filter extraction (`extract_filters_llm`)

After rewrite, a second LLM call parses the rewritten query for hard constraints (country, city, nationality, years of experience, region expansions like "Middle East") and converts them into a `where` filter.

**Why hybrid (filter + vector) instead of pure vector search?**

A vector search for "regulatory affairs in Saudi Arabia" may return candidates from Egypt or Jordan simply because their profiles are semantically similar. A `where: {"country": {"$in": ["Saudi Arabia"]}}` filter applies the geographic constraint exactly, allowing the vector score to focus entirely on the functional match.

**Fallback:** If the LLM call fails or returns an invalid `where` shape, the system falls back to `where=None` and runs a pure vector search. No request ever fails due to the filter step.

#### Over-retrieve + LLM re-rank

The vector search retrieves `top_k × rerank_multiplier` candidates (default: 3×). A separate LLM call (`explain_matches_llm`) then scores each candidate 0–10 against the query, writes a human-readable `why_match` explanation, and produces `highlights` bullet points. Results below `min_rerank_score` are filtered out; the remainder are sorted by LLM score and truncated to `top_k`.

**Why over-retrieve?**

Vector cosine similarity is a proxy for relevance, not a perfect ranker. A candidate ranked #8 by embedding distance may be a better functional match than #2. Giving the LLM a wider pool (e.g. 30 candidates to judge from when `top_k=10`) reduces the probability of the best matches being excluded before the LLM sees them.

**Why not a dedicated cross-encoder re-ranker?**

A cross-encoder (e.g. `cross-encoder/ms-marco-MiniLM`) would be cheaper and faster, but it only produces a relevance score — it cannot write the `why_match` explanation and highlights that the API response requires. Combining scoring + explanation in a single LLM call is simpler and sufficient at this scale.

#### Retry node for low-recall scenarios

If the retrieved hit count after the rewrite is below a threshold, the graph takes a `retry` edge and re-runs retrieval with the original (unrewritten) user query, skipping filter extraction. This handles cases where the rewriter over-specifies and reduces recall (e.g. a complex multi-part query collapsed into something too narrow).


**Why SQLite over an in-memory dict?**

An in-memory dict is lost on every process restart. SQLite is a zero-dependency, durable store for the development and assessment context. In production this would be replaced with a Redis or Postgres-backed checkpointer (LangGraph supports both) to survive restarts and scale across multiple workers.


---

## What could be improved

### Design improvements

**1. Hybrid BM25 + vector search**
Pure vector search misses exact-match signals (e.g., a rare skill name like "REACH regulatory compliance"). A hybrid retriever combining sparse BM25 (keyword match) with dense vector similarity would improve recall for specific, technical queries. 

**2. Document-level chunking with candidate-level deduplication**
The current one-vector-per-candidate approach means role-specific queries ("Python engineer at a fintech") compete with the candidate's entire career. Role-level chunks would improve precision for tenure-specific searches, at the cost of requiring result deduplication (multiple chunks → one candidate). 

**3. LLM-assisted seniority and industry normalisation at ingest time**
The current `_derive_seniority` function is a keyword lookup over title strings. It misclassifies unconventional titles (e.g. "Growth Lead", "Technical Fellow"). An LLM pass during ingestion to normalise titles into canonical seniority bands and standardise industry labels (e.g. mapping "Pharmaceuticals", "Pharma", and "Life Sciences" to one label) would improve both vector quality and filter precision.

**4. Re-ranker model (cross-encoder) for scoring, LLM for explanation only**
Using a full LLM to score 30 candidates per request is the most expensive step. A dedicated cross-encoder could score candidates in ~50ms at near-zero cost, while the LLM only generates `why_match` explanations for the top-N results that survive re-ranking.

**5. Structured skill taxonomy and normalisation**
Skills are currently stored as free-text strings ("Python", "python 3", "Python 3.x"). A normalisation step at ingest (stemming, synonym mapping, or an LLM taxonomy pass) would prevent vocabulary mismatch between query and indexed skills.

**6. Agent-based retrieval (tool-calling)**
Instead of a fixed pipeline, an agent could decide when to rewrite, when to add filters, when to broaden/narrow the search, and when to ask clarifying questions. This can improve hard queries, but it needs strong guardrails for cost, latency, and correctness.

---

### Production-readiness improvements

**0. Use proper design patterns**
As the codebase grows, introduce clearer boundaries like **Repository** (DB access), **Service** (business logic), and **Adapter/Client** (OpenRouter/Chroma). This makes testing easier and keeps the API layer thin.

**1. Replace ChromaDB with a scalable vector store**
`chromadb.PersistentClient` is a single-process, file-backed store. It cannot be shared across multiple API workers, does not support horizontal scaling, and has no built-in replication or backup. For production, replace with:
- **Qdrant / Weaviate / Pinecone** — purpose-built vector DBs with server mode, clustering, and managed cloud options.

**2. Replace SQLite checkpointer with Redis or Postgres**
The `AsyncSqliteSaver` is a single-file store that breaks under multiple Uvicorn workers and is lost if the container's filesystem is ephemeral. LangGraph ships Postgres checkpointer for production use.

**3. Incremental ingestion**
The current `POST /ingest` is a full re-index. At 25k profiles with 3072-dimensional embeddings, this costs time and API credits on every run. Incremental ingestion (track `updated_at` timestamps, only re-embed changed or new candidates) would make ingestion a lightweight background job that can run on a schedule.

**4. Authentication and authorisation**
The API has no authentication. In production, every endpoint should require at minimum a bearer token validated against an identity provider, with role-based access control to gate the `POST /ingest` endpoint.

**5. Rate limiting**
Both the `/ingest` and `/chat` endpoints make outbound LLM API calls with real cost implications. Without rate limiting, a single misbehaving client can exhaust the OpenRouter budget. 

**6. Retry and backoff in API clients**
`EmbeddingsClient` and `ChatClient` make raw `httpx` calls with no retry logic. In production, transient 429 / 503 errors from OpenRouter would silently fail entire embedding batches. 

**7. Observability: structured logging, metrics, and tracing**
The app emits structured JSON logs (`app/json_logging.py`), but there is no metrics collection or distributed tracing. For production:
- **Metrics:** Expose a Prometheus `/metrics` endpoint (request latency, error rates, vector index size, LLM call latency).
- **Tracing:** Instrument with OpenTelemetry spans across the LangGraph nodes so each step of a `/chat` request is visible in a trace.
- **LLM observability:** Integrate LangSmith or a similar LLM observability platform to track prompt versions, token usage, and output quality over time.

**8. Secrets management**
`OPENROUTER_API_KEY` and `DATABASE_URL` are loaded from `.env`. In production, secrets should come from a secrets manager (AWS Secrets Manager, HashiCorp Vault, GCP Secret Manager) with automatic rotation, not from static environment files or committed config.

**9. Containerisation and orchestration**
The service has no `Dockerfile`. For reproducible deployments:
- Add a `Dockerfile` with a locked Python version, non-root user, and health-check `CMD`.
- Add a `docker-compose.yml` for local development that wires up the API, the Postgres tunnel proxy, and a Chroma server container.

**10. Test coverage**
There are no automated tests. Before production, add:
- Unit tests
- Integration tests
- End-to-end etc

**11. Async Chroma client or connection pooling**
All Chroma operations run inside `asyncio.to_thread`, which occupies a thread-pool thread for the duration of each query. Under high concurrency this could exhaust the default `ThreadPoolExecutor`.
