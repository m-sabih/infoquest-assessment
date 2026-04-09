import logging
import re
from typing import Any
from uuid import UUID

from langchain_core.tools import tool
from langchain_openrouter import ChatOpenRouter
from langgraph.prebuilt import create_react_agent

from app.config import Settings
from app.services.chat_logic import extract_filters_llm, search_experts
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)

# Each ReAct iteration costs ~2 recursion steps (LLM pick + tool run).
# 3 searches + initial reasoning + final response  ≈  10 steps.
_MAX_RECURSION = 10

def _normalize_years(value: object) -> int | None:
    if value is None:
        return None
    try:
        v = int(value)
    except Exception:
        return None
    return v if v >= 0 else None


def _simple_highlights(meta: dict) -> list[str]:
    out: list[str] = []
    headline = (meta.get("headline") or "").strip()
    if headline:
        out.append(headline)
    loc_parts = [
        str(meta.get("city") or "").strip(),
        str(meta.get("country") or "").strip(),
    ]
    loc = ", ".join(p for p in loc_parts if p)
    if loc:
        out.append(f"Location: {loc}")
    yoe = _normalize_years(meta.get("years_of_experience"))
    if yoe is not None:
        out.append(f"Years of experience: {yoe}")
    nat = (meta.get("nationality") or "").strip()
    if nat:
        out.append(f"Nationality: {nat}")
    return out[:5]


def _heuristic_why(query: str, meta: dict) -> str:
    q = query.lower()
    hints: list[str] = []
    for key in ("headline", "country", "city", "nationality"):
        val = str(meta.get(key) or "").strip()
        if not val:
            continue
        if any(tok in q for tok in re.findall(r"[a-zA-Z]{3,}", val.lower())):
            hints.append(val)
    if hints:
        return ("Matches on: " + "; ".join(dict.fromkeys(hints)))[:220]
    headline = (meta.get("headline") or "").strip()
    if headline:
        return f"Relevant background: {headline}"[:240]
    return "Semantically similar profile to the query."


def _build_matches(*, query: str, hits: list[dict]) -> list[dict]:
    """Convert raw vector-store hits into ExpertMatch-compatible dicts."""
    results: list[dict] = []
    for h in hits:
        meta = h.get("metadata") or {}
        cid = h.get("candidate_id")
        try:
            uuid_val = UUID(str(cid))
        except Exception:
            continue

        first = str(meta.get("first_name") or "").strip()
        last = str(meta.get("last_name") or "").strip()
        name = (first + " " + last).strip() or str(uuid_val)

        city = str(meta.get("city") or "").strip()
        country = str(meta.get("country") or "").strip()
        loc = ", ".join(b for b in [city, country] if b) or None

        results.append(
            {
                "candidate_id": uuid_val,
                "name": name,
                "headline": meta.get("headline") or None,
                "location": loc,
                "nationality": meta.get("nationality") or None,
                "years_of_experience": _normalize_years(meta.get("years_of_experience")),
                "score": float(h.get("score") or 0.0),
                "why_match": _heuristic_why(query, meta),
                "highlights": _simple_highlights(meta),
                "metadata": meta,
            }
        )
    results.sort(key=lambda r: r["score"], reverse=True)
    return results


def _summarise_hits(hits: list[dict]) -> str:
    """Compact text summary of search results for the LLM to review."""
    if not hits:
        return "No results returned."

    lines = [f"Found {len(hits)} expert(s):"]
    for i, h in enumerate(hits[:15], 1):
        meta = h.get("metadata") or {}
        score = h.get("score", 0.0)
        first = str(meta.get("first_name") or "").strip()
        last = str(meta.get("last_name") or "").strip()
        name = (first + " " + last).strip() or str(h.get("candidate_id", "unknown"))
        headline = str(meta.get("headline") or "").strip()
        country = str(meta.get("country") or "").strip()
        city = str(meta.get("city") or "").strip()
        location = ", ".join(b for b in [city, country] if b)
        doc_snippet = (h.get("document") or "")[:200].replace("\n", " ").strip()
        lines.append(
            f"  [{i}] {name} | score={score:.3f} | {headline} | {location}\n"
            f"      Profile: {doc_snippet}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an expert-search agent. Your sole responsibility is to find the most \
relevant experts from a vector database for the user's query.

Location and experience constraints (city, country, nationality, \
years_of_experience) are applied automatically as hard metadata filters — \
you do NOT need to handle them. Focus entirely on crafting the best \
semantic query for role, skills, industry, and seniority.

## Search strategy

1. **First search** — craft a precise semantic query using all key criteria \
from the user's request (role, skills, industry, certifications, seniority, etc.).

2. **Review results** — examine each result's score, headline, location, \
and profile snippet:
   - score ≥ 0.50  →  excellent match
   - score 0.35–0.49  →  moderate; inspect the profile carefully
   - score < 0.35  →  weak; the query likely needs refinement

3. **Decide whether to refine** (up to 3 searches total):
   - Results already look relevant and cover the request → STOP and confirm.
   - Too few or irrelevant results → refine with ONE of these tactics:
     • **Narrow**: add missing specifics (exact certification, tool, sub-role).
     • **Broaden**: relax one constraint (wider region, adjacent role, \
related industry).
     • **Reframe**: rephrase using synonyms or alternative terminology.

4. **Stop early** as soon as you are satisfied with result quality.

## Cost / latency guardrails
- Never search more than 3 times.
- Keep search queries concise (1–2 sentences max).
- Do NOT attempt to format or list the final results — the system will do that.

When you are done searching, respond with exactly:
  SEARCH_COMPLETE
"""


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

async def run_search_agent(
    *,
    settings: Settings,
    store: VectorStore,
    query: str,
    top_k: int,
    thread_id: str,
) -> dict[str, Any]:
    """Run the iterative search agent and return ``{"matches": [...]}``.

    The agent calls ``search_experts_tool`` up to 3 times, reviewing and
    refining its query after each round.  The last set of raw hits is
    transformed into ``ExpertMatch``-compatible dicts before returning.
    """
    # Pre-extract hard metadata filters (city/country/nationality/yoe) from the
    # query using a dedicated LLM call.  This is more reliable than asking the
    # ReAct agent to produce them inline — models consistently ignore optional
    # dict parameters in tool schemas.  The agent then only controls the
    # semantic query text; the where filter is injected via closure.
    semantic_query, base_where = await extract_filters_llm(settings=settings, query=query)
    logger.info(
        "run_search_agent: pre-extracted where=%s\nsemantic_query=%r",
        base_where,
        semantic_query,
    )

    # Shared mutable state between the closure and the caller.
    _state: dict[str, list[dict]] = {"hits": []}

    # -----------------------------------------------------------------
    # Tool: wraps search_experts, closes over settings / store / top_k /
    # base_where.  The agent only controls search_query.
    # -----------------------------------------------------------------
    @tool
    async def search_experts_tool(search_query: str) -> str:
        """Search the expert vector database with a natural-language query.

        Args:
            search_query: Semantic description of the expert profile to find.
                Include role, skills, industry, and seniority.
                Keep it to 1-2 sentences.
                Do NOT include location or geo constraints — those are applied
                automatically as hard filters.
        """
        logger.info("search_agent [tool]: query=%r top_k=%d where=%s", search_query, top_k, base_where)
        hits = await search_experts(
            settings=settings,
            store=store,
            query=search_query,
            top_k=top_k,
            where=base_where,
        )
        _state["hits"] = hits
        summary = _summarise_hits(hits)
        logger.info("search_agent [tool]: returned %d hits", len(hits))
        return summary

    # -----------------------------------------------------------------
    # LLM + agent
    # -----------------------------------------------------------------
    model = ChatOpenRouter(
        model=settings.chat_model,
        temperature=0,
        max_tokens=512,
        max_retries=2,
    )

    agent = create_react_agent(
        model=model,
        tools=[search_experts_tool],
        prompt=_SYSTEM_PROMPT,
    )

    try:
        await agent.ainvoke(
            {"messages": [{"role": "user", "content": semantic_query}]},
            config={
                "recursion_limit": _MAX_RECURSION,
                "configurable": {"thread_id": thread_id},
            },
        )
    except Exception as exc:
        # GraphRecursionError (or any other limit) — use whatever we have.
        logger.warning(
            "search_agent: agent stopped early (%s: %s) — using last results",
            type(exc).__name__,
            exc,
        )

    hits = _state["hits"]
    logger.info("search_agent: building matches from %d raw hits", len(hits))
    matches = _build_matches(query=query, hits=hits)
    logger.info("search_agent: done — %d matches returned", len(matches))
    return {"matches": matches}
