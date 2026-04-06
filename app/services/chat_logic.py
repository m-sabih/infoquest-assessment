from dataclasses import dataclass
import json
import logging
from uuid import UUID

from app.config import Settings
from app.services.embeddings_client import EmbeddingsClient
from app.services.chat_client import ChatClient
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RewriteResult:
    rewritten_query: str
    use_previous_results: bool

def _score_from_distance(distance: float | None) -> float:
    # Chroma distances are smaller=better. Convert to a similarity-like score.
    if distance is None:
        return 0.0
    return float(1.0 / (1.0 + float(distance)))

async def rewrite_query(
    *,
    settings: Settings,
    query: str,
    previous_query: str | None,
    previous_result_ids: list[str] | None,
) -> RewriteResult:
    prev_q = (previous_query or "").strip()
    if not settings.openrouter_api_key.strip() or not prev_q:
        logger.info("rewrite_query: skip (no api key or no previous_query)")
        return RewriteResult(rewritten_query=query, use_previous_results=False)

    logger.info(
        "rewrite_query: start (prev_query_len=%s query_len=%s prev_ids=%s model=%s)",
        len(prev_q),
        len(query),
        len(previous_result_ids or []),
        settings.chat_model,
    )
    client = ChatClient(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        model=settings.chat_model,
    )
    system = (
        "You rewrite follow-up expert-search queries into a standalone query. "
        "Return JSON only with keys: rewritten_query (string), use_previous_results (boolean)."
    )
    user = (
        f"Previous query:\n{prev_q}\n\n"
        f"User query:\n{query}\n\n"
        f"Previous result ids count: {len(previous_result_ids or [])}\n\n"
        "Rules:\n"
        "- If the user references prior results (e.g., 'those', 'them', 'filter those'), set use_previous_results=true.\n"
        "- rewritten_query must be standalone, incorporating the intent from the previous query when needed.\n"
        "- Output JSON only."
    )
    try:
        obj = await client.json_completion(system=system, user=user)
    except Exception:
        logger.exception("rewrite_query: LLM call failed")
        return RewriteResult(rewritten_query=query, use_previous_results=False)

    rq = str(obj.get("rewritten_query") or "").strip() or query
    upr = bool(obj.get("use_previous_results")) if "use_previous_results" in obj else False
    logger.info(
        "rewrite_query: done (rewritten_len=%s use_previous_results=%s)",
        len(rq),
        upr,
    )
    return RewriteResult(rewritten_query=rq, use_previous_results=upr)


async def search_experts(
    *,
    settings: Settings,
    store: VectorStore,
    query: str,
    top_k: int,
) -> list[dict]:
    logger.info("search_experts: start (query_len=%s top_k=%s)", len(query), top_k)
    embed = EmbeddingsClient(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        model=settings.embedding_model,
    )
    [qemb] = await embed.embed_texts([query])
    res = store.query(query_embedding=qemb, top_k=top_k)

    ids = (res.get("ids") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]
    docs = (res.get("documents") or [[]])[0]

    out: list[dict] = []
    for i, cid in enumerate(ids):
        meta = metas[i] if i < len(metas) else {}
        dist = dists[i] if i < len(dists) else None
        doc = docs[i] if i < len(docs) else ""
        out.append(
            {
                "candidate_id": cid,
                "metadata": meta or {},
                "distance": dist,
                "score": _score_from_distance(dist),
                "document": doc or "",
            }
        )
    logger.info("search_experts: done (hits=%s)", len(out))
    return out


def build_matches_base(*, hits: list[dict]) -> list[dict]:
    results: list[dict] = []
    for h in hits:
        meta = h.get("metadata") or {}
        cid = h.get("candidate_id")
        try:
            uuid = UUID(str(cid))
        except Exception:
            continue

        first = str(meta.get("first_name") or "").strip()
        last = str(meta.get("last_name") or "").strip()
        name = (first + " " + last).strip() or str(uuid)

        city = str(meta.get("city") or "").strip()
        country = str(meta.get("country") or "").strip()
        loc = ", ".join([b for b in [city, country] if b]) or None

        yoe = int(meta.get("years_of_experience"))
        results.append(
            {
                "candidate_id": uuid,
                "name": name,
                "headline": (meta.get("headline") or None),
                "location": loc,
                "nationality": (meta.get("nationality") or None),
                "years_of_experience": yoe,
                "score": float(h.get("score") or 0.0),
                "why_match": "Relevant profile based on semantic similarity.",
                "highlights": [],
                "metadata": meta,
            }
        )
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


async def explain_matches_llm(
    *,
    settings: Settings,
    query: str,
    hits: list[dict],
) -> dict[str, dict]:
    """
    Ask an LLM to produce why_match + highlights per candidate_id.
    Returns mapping: candidate_id(str) -> {"why_match": str, "highlights": [str, ...]}
    """
    if not settings.openrouter_api_key.strip():
        logger.info("explain_matches_llm: skip (no api key)")
        return {}
    if not hits:
        logger.info("explain_matches_llm: skip (no hits)")
        return {}

    expected_ids = [str(h.get("candidate_id")) for h in hits]
    logger.info(
        "explain_matches_llm: start (query_len=%s hits=%s model=%s)",
        len(query),
        len(hits),
        settings.chat_model,
    )
    client = ChatClient(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        model=settings.chat_model,
    )

    candidates: list[dict] = []
    for h in hits[:50]:
        meta = h.get("metadata") or {}
        candidates.append(
            {
                "candidate_id": str(h.get("candidate_id")),
                "headline": meta.get("headline") or "",
                "location": ", ".join([b for b in [meta.get("city") or "", meta.get("country") or ""] if b]).strip(),
                "nationality": meta.get("nationality") or "",
                "years_of_experience": meta.get("years_of_experience"),
                "profile_snippet": str(h.get("document") or "").strip(),
            }
        )

    system = (
        "You are an expert recruiter assistant. "
        "Given a user search query and candidate profile snippets, produce concise explanations. "
        "You MUST return one item for EVERY candidate_id provided (no omissions). "
        "Return JSON only with shape:\n"
        "{\"items\": [{\"candidate_id\": \"<id>\", \"why_match\": \"<text>\", \"highlights\": [\"<h1>\", \"<h2>\"]}]}.\n"
        "Example:\n"
        "{\"items\": [{\"candidate_id\": \"9c2f...\", \"why_match\": \"Matches due to regulatory affairs experience in pharma and MENA exposure.\", "
        "\"highlights\": [\"Regulatory Affairs Lead — pharma\", \"Based in Dubai, UAE\", \"12 years experience\"]}]}\n"
        "Rules:\n"
        "- why_match: 1-2 short sentences, specific to the query.\n"
        "- highlights: 2-5 bullet-like strings, no markdown.\n"
        "- Do not invent facts not present in the snippet/fields.\n"
        "- Output JSON only."
    )
    user = {
        "query": query,
        "candidates": candidates,
    }
    try:
        obj = await client.json_completion(system=system, user=json.dumps(user, ensure_ascii=False))
    except Exception:
        logger.exception("LLM explanation call failed")
        return {}

    items = obj.get("items") if isinstance(obj, dict) else None
    if not isinstance(items, list):
        logger.warning("LLM explanation returned unexpected JSON shape: %s", obj)
        return {}

    out: dict[str, dict] = {}
    for it in items:
        if not isinstance(it, dict):
            continue

        cid = str(it.get("candidate_id") or "").strip()
        why = str(it.get("why_match") or "").strip()
        highs = it.get("highlights")

        if not cid or not why:
            continue

        if not isinstance(highs, list):
            highs = []
        highlights = [str(h).strip() for h in highs if str(h).strip()][:5]
        out[cid] = {"why_match": why, "highlights": highlights}

    logger.info("explain_matches_llm: done (explained=%s of %s)", len(out), len(expected_ids))
    return out
