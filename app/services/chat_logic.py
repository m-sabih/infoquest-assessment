from dataclasses import dataclass
import json
import logging
from typing import Any
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
    if not settings.openrouter_api_key.strip():
        logger.info("rewrite_query: skip (no api key)")
        return RewriteResult(rewritten_query=query, use_previous_results=False)

    prev_q = (previous_query or "").strip()
    is_followup = bool(prev_q)

    logger.info(
        "rewrite_query: start (model=%s is_followup=%s prev_ids=%s)\nprevious_query=%r\nquery=%r",
        settings.chat_model,
        is_followup,
        len(previous_result_ids or []),
        prev_q,
        query,
    )
    client = ChatClient(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        model=settings.chat_model,
    )
    system = (
        "You rewrite expert-search queries into clean, embedding-optimized standalone queries.\n"
        "Always:\n"
        "- Strip conversational filler: 'Find me', 'I need', 'Can you find', 'I am looking for',\n"
        "  'Show me', 'Search for', 'Look for', 'Get me', 'I want', 'Please find', etc.\n"
        "- Produce a concise, keyword-dense query that captures role, skills, industry, seniority,\n"
        "  and location intent — the kind of text that embeds well for semantic search.\n"
        "- If a previous query exists and the new query is a follow-up, merge the full intent from\n"
        "  both into a single standalone query.\n"
        "- Set use_previous_results=true ONLY if the user explicitly references prior results\n"
        "  (e.g., 'those', 'them', 'from those results', 'filter those').\n"
        "Return JSON only: {\"rewritten_query\": \"<clean query>\", \"use_previous_results\": false}"
    )

    if is_followup:
        user = (
            f"Previous query:\n{prev_q}\n\n"
            f"User query:\n{query}\n\n"
            f"Previous result ids count: {len(previous_result_ids or [])}\n\n"
            "Output JSON only."
        )
    else:
        user = f"User query:\n{query}\n\nOutput JSON only."

    try:
        obj = await client.json_completion(system=system, user=user)
    except Exception:
        logger.exception("rewrite_query: LLM call failed")
        return RewriteResult(rewritten_query=query, use_previous_results=False)

    rq = str(obj.get("rewritten_query") or "").strip() or query
    upr = bool(obj.get("use_previous_results")) if "use_previous_results" in obj else False
    logger.info(
        "rewrite_query: done (is_followup=%s use_previous_results=%s)\nrewritten_query=%r",
        is_followup,
        upr,
        rq,
    )
    return RewriteResult(rewritten_query=rq, use_previous_results=upr)


async def search_experts(
    *,
    settings: Settings,
    store: VectorStore,
    query: str,
    top_k: int,
    where: dict[str, Any] | None = None,
) -> list[dict]:
    """
    Search for experts in the vector store.
    Args:
        settings: The settings object.
        store: The vector store object.
        query: The query to search for.
        top_k: The number of results to return.
        where: The where clause to filter the results by.
    Returns:
        A list of dictionaries, each containing the expert's metadata, distance, and score.
    """
    logger.info("search_experts: start (top_k=%s where=%s)\nquery=%r", top_k, where, query)
    embed = EmbeddingsClient(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        model=settings.embedding_model,
    )
    [qemb] = await embed.embed_texts([query])
    res = store.query(query_embedding=qemb, top_k=top_k, where=where)

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


async def extract_filters_llm(
    *,
    settings: Settings,
    query: str,
) -> tuple[str, dict[str, Any] | None]:
    """
    Extract structured Chroma metadata filters from the query.

    Returns:
    - query_text: the semantic query to embed
    - where: Chroma `where` filter dict (or None)
    """
    if not settings.openrouter_api_key.strip():
        logger.info("extract_filters_llm: skip (no api key)")
        return query, None

    client = ChatClient(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        model=settings.chat_model,
    )
    system = (
        "You extract structured search filters from an expert-search query.\n"
        "Return JSON only with keys:\n"
        "- query_text: string (the semantic query to embed, keep ALL intent including regions/industries)\n"
        "- where: object|null (Chroma metadata filters)\n\n"
        "Allowed where fields (ONLY these, must match stored metadata):\n"
        "- candidate_id\n"
        "- headline\n"
        "- city\n"
        "- country\n"
        "- nationality\n"
        "- years_of_experience\n"
        "- text_truncated\n"
        "Rules:\n"
        "- If the user specifies ANY filterable constraint, you SHOULD express it as a hard filter in where.\n"
        "  Filterable constraints include: city, country, nationality, years_of_experience ranges, candidate_id, and regions.\n"
        "- Use exact string values for country/city/nationality ONLY when the user names a specific city or country (e.g. 'in Dubai', 'from Germany').\n"
        "- For REGIONS (Middle East, APAC, Europe, Latin America, etc.), create a HARD filter using $or + $in across BOTH country and nationality.\n"
        "  Example shape:\n"
        "  {\"$or\": [{\"country\": {\"$in\": [\"...\"]}}, {\"nationality\": {\"$in\": [\"...\"]}}]}\n"
        "  Region expansions you can use:\n"
        "  - Middle East countries: Saudi Arabia, United Arab Emirates, Qatar, Kuwait, Bahrain, Oman, Yemen, Iraq, Jordan, Lebanon, Syria, Palestine, Egypt\n"
        "  - Middle East nationalities: Saudi Arabian, Emirati, Qatari, Kuwaiti, Bahraini, Omani, Yemeni, Iraqi, Jordanian, Lebanese, Syrian, Palestinian, Egyptian\n"        
        "- For years_of_experience, use operators like {\"$gte\": 10} when the user says '10+ years'.\n"
        "- Only use headline as a filter when the user provides an exact phrase to match; otherwise keep role/function terms in query_text.\n"
        "- Prefer where=null over a filter that is likely to produce zero results.\n"
        "- Do not invent filters.\n"
        "Output JSON only."
    )
    user = {"query": query}
    try:
        obj = await client.json_completion(system=system, user=json.dumps(user, ensure_ascii=False))
    except Exception:
        logger.exception("extract_filters_llm: LLM call failed")
        return query, None

    query_text = str((obj or {}).get("query_text") or "").strip() or query
    where = (obj or {}).get("where")
    if where is None:
        logger.info("extract_filters_llm: done (where=None)\nquery_text=%r", query_text)
        return query_text, None
    if not isinstance(where, dict):
        logger.warning("extract_filters_llm: invalid where type: %s", type(where))
        return query_text, None

    # Pass Chroma logical operators ($or, $and) through unchanged; only strip
    # unrecognised plain field names to avoid Chroma validation errors.
    allowed_fields = {
        "candidate_id",
        "headline",
        "city",
        "country",
        "nationality",
        "years_of_experience",
        "text_truncated",
    }
    logical_ops = {"$or", "$and", "$not"}
    cleaned: dict[str, Any] = {}
    for k, v in where.items():
        if k in logical_ops:
            cleaned[k] = v
        elif k in allowed_fields and v is not None:
            cleaned[k] = v
    logger.info("extract_filters_llm: done (where=%s)\nquery_text=%r", cleaned, query_text)
    return query_text, cleaned or None


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
        name = (first + " " + last).strip()

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
        "explain_matches_llm: start (model=%s hits=%s)\nquery=%r\ncandidate_ids=%s",
        settings.chat_model,
        len(hits),
        query,
        expected_ids,
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
        "You are an expert recruiter assistant and relevance judge. "
        "Given a user search query and candidate profile snippets:\n"
        "1. Score each candidate's relevance to the query on a scale of 0–10 (10 = perfect match on ALL stated requirements, "
        "0 = completely irrelevant). Be strict: missing a core requirement (wrong role, wrong industry, wrong region) should "
        "bring the score below 5.\n"
        "2. Write a concise explanation and key highlights.\n"
        "You MUST return one item for EVERY candidate_id provided (no omissions). "
        "Return JSON only with shape:\n"
        "{\"items\": [{\"candidate_id\": \"<id>\", \"relevance_score\": 8, \"why_match\": \"<text>\", \"highlights\": [\"<h1>\", \"<h2>\"]}]}.\n"
        "Example:\n"
        "{\"items\": [{\"candidate_id\": \"9c2f...\", \"relevance_score\": 9, "
        "\"why_match\": \"Strong match — Regulatory Affairs Lead at a pharma company in Dubai with 12 years MENA experience.\", "
        "\"highlights\": [\"Regulatory Affairs Lead — pharma\", \"Based in Dubai, UAE\", \"12 years experience\"]}]}\n"
        "Rules:\n"
        "- relevance_score: integer 0–10. Missing ANY core requirement lowers the score significantly.\n"
        "- why_match: 1-2 short sentences, specific to the query. If weak, state the gaps clearly.\n"
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

        raw_score = it.get("relevance_score")
        try:
            relevance_score = float(raw_score) if raw_score is not None else None
        except (TypeError, ValueError):
            relevance_score = None

        out[cid] = {"why_match": why, "highlights": highlights, "relevance_score": relevance_score}

    logger.info("explain_matches_llm: done (explained=%s of %s)", len(out), len(expected_ids))
    return out
