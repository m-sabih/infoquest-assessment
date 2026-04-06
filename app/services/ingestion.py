from __future__ import annotations

import asyncio
import logging
import time
import asyncpg

from app.config import Settings
from app.schemas.ingest import IngestRequest, IngestResponse
from app.services.candidate_repository import iter_candidate_pages
from app.services.embeddings_client import EmbeddingsClient
from app.services.profile_builder import build_profile_text, row_to_chroma_metadata
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


async def run_ingestion(
    *,
    settings: Settings,
    pool: asyncpg.Pool,
    store: VectorStore,
    body: IngestRequest,
) -> IngestResponse:
    if not settings.openrouter_api_key.strip():
        return IngestResponse(
            status="failed",
            candidates_fetched=0,
            vectors_upserted=0,
            errors=0,
            duration_seconds=0.0,
            message="OPENROUTER_API_KEY is not set. Add it to your environment or .env file.",
        )

    t0 = time.perf_counter()
    candidates_fetched = 0
    vectors_upserted = 0
    errors = 0

    logger.info(
        "Ingestion started (collection=%s replace=%s limit=%s page_size=%s embed_batch=%s max_chars=%s)",
        store.collection_name,
        body.replace_collection,
        body.limit,
        settings.candidate_page_size,
        settings.embedding_batch_size,
        settings.max_profile_chars,
    )

    if body.replace_collection:
        logger.info("Resetting Chroma collection=%s", store.collection_name)
        await asyncio.to_thread(store.reset_collection)

    client = EmbeddingsClient(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        model=settings.embedding_model,
    )

    page_idx = 0
    async for page in iter_candidate_pages(
        pool,
        page_size=settings.candidate_page_size,
        limit=body.limit,
    ):
        page_idx += 1
        logger.info("Fetched candidate page=%s size=%s", page_idx, len(page))
        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict] = []

        for row in page:
            candidates_fetched += 1
            text, truncated = build_profile_text(row, settings.max_profile_chars)
            cid = str(row["id"])
            ids.append(cid)
            documents.append(text)
            metadatas.append(row_to_chroma_metadata(row, truncated))

        bsize = settings.embedding_batch_size
        for start in range(0, len(documents), bsize):
            batch_ids = ids[start : start + bsize]
            batch_docs = documents[start : start + bsize]
            batch_meta = metadatas[start : start + bsize]
            try:
                t_embed0 = time.perf_counter()
                logger.info(
                    "Embedding batch page=%s offset=%s size=%s (candidates_fetched=%s vectors_upserted=%s errors=%s)",
                    page_idx,
                    start,
                    len(batch_docs),
                    candidates_fetched,
                    vectors_upserted,
                    errors,
                )
                embeddings = await client.embed_texts(batch_docs)
                logger.info(
                    "Embedded batch page=%s offset=%s size=%s in %.3fs",
                    page_idx,
                    start,
                    len(batch_docs),
                    time.perf_counter() - t_embed0,
                )
            except Exception:
                logger.exception("Embedding batch failed (size=%s)", len(batch_docs))
                errors += len(batch_docs)
                continue
            if len(embeddings) != len(batch_ids):
                logger.error(
                    "Embedding count mismatch: got %s expected %s",
                    len(embeddings),
                    len(batch_ids),
                )
                errors += len(batch_ids)
                continue
            try:
                t_upsert0 = time.perf_counter()
                await asyncio.to_thread(
                    store.upsert_batch,
                    ids=batch_ids,
                    embeddings=embeddings,
                    documents=batch_docs,
                    metadatas=batch_meta,
                )
                vectors_upserted += len(batch_ids)
                logger.info(
                    "Upserted batch page=%s offset=%s size=%s in %.3fs (vectors_upserted=%s)",
                    page_idx,
                    start,
                    len(batch_ids),
                    time.perf_counter() - t_upsert0,
                    vectors_upserted,
                )
            except Exception:
                logger.exception("Vector upsert failed (size=%s)", len(batch_ids))
                errors += len(batch_ids)

    elapsed = time.perf_counter() - t0
    status = "ok" if errors == 0 else "partial"
    if vectors_upserted == 0 and candidates_fetched > 0:
        status = "failed"

    logger.info(
        "Ingestion finished (status=%s candidates_fetched=%s vectors_upserted=%s errors=%s duration=%.3fs)",
        status,
        candidates_fetched,
        vectors_upserted,
        errors,
        elapsed,
    )

    return IngestResponse(
        status=status,
        candidates_fetched=candidates_fetched,
        vectors_upserted=vectors_upserted,
        errors=errors,
        duration_seconds=round(elapsed, 3),
    )
