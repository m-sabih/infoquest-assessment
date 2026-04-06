from __future__ import annotations

import logging
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.config import Settings
from app.services.chat_logic import (
    build_matches_base,
    explain_matches_llm,
    extract_filters_llm,
    rewrite_query,
    search_experts,
)
from app.services.vector_store import VectorStore

logger = logging.getLogger(__name__)


class ChatGraphState(TypedDict, total=False):
    # inputs
    query: str
    top_k: int

    # persisted conversational context (stored via checkpointer)
    previous_query: str | None
    previous_result_ids: list[str] | None

    # intermediate
    rewritten_query: str
    use_previous_results: bool
    query_text: str
    where: dict
    hits: list[dict]
    attempt: int

    # outputs
    result_ids: list[str]
    matches: list[dict]
    retry_reason: str | None


def _should_retry(state: ChatGraphState) -> bool:
    """
    Single simple quality loop:
    - If we got too few hits, retry once using the original user query
      (sometimes rewrite overspecifies and hurts recall).
    """
    attempt = int(state.get("attempt") or 0)
    hits = state.get("hits") or []
    if attempt >= 1:
        return False
    threshold = max(3, int(state.get("top_k") or 10) // 3)
    decision = len(hits) < threshold
    logger.info(
        "chat_graph: retry_decision=%s (attempt=%s hits=%s threshold=%s)",
        decision,
        attempt,
        len(hits),
        threshold,
    )
    return decision


async def _rewrite_node(state: ChatGraphState, *, settings: Settings) -> ChatGraphState:
    logger.info(
        "chat_graph: rewrite start\nquery=%r\nprevious_query=%r\nprevious_result_ids_count=%s",
        state.get("query") or "",
        state.get("previous_query"),
        len(state.get("previous_result_ids") or []),
    )
    rr = await rewrite_query(
        settings=settings,
        query=state["query"],
        previous_query=state.get("previous_query"),
        previous_result_ids=state.get("previous_result_ids"),
    )
    logger.info(
        "chat_graph: rewrite done (rewritten_len=%s use_previous_results=%s)",
        len(rr.rewritten_query),
        rr.use_previous_results,
    )
    return {
        "rewritten_query": rr.rewritten_query,
        "use_previous_results": rr.use_previous_results,
        "attempt": 0,
    }


async def _retrieve_node(
    state: ChatGraphState,
    *,
    settings: Settings,
    store: VectorStore,
) -> ChatGraphState:
    q = state.get("rewritten_query") or state["query"]
    top_k = state["top_k"]
    # Over-retrieve so the LLM re-ranker has a larger candidate pool to judge from.
    retrieve_k = top_k * settings.rerank_multiplier
    logger.info(
        "chat_graph: retrieve start (top_k=%s retrieve_k=%s)\nquery=%r",
        top_k,
        retrieve_k,
        q,
    )

    query_text, where = await extract_filters_llm(settings=settings, query=q)
    hits = await search_experts(
        settings=settings,
        store=store,
        query=query_text,
        top_k=retrieve_k,
        where=where,
    )

    # If the user asked to filter prior results, restrict to those ids.
    if state.get("use_previous_results") and state.get("previous_result_ids"):
        keep = set(state["previous_result_ids"])
        before = len(hits)
        hits = [h for h in hits if str(h.get("candidate_id")) in keep]
        logger.info("chat_graph: retrieve filtered_to_previous (before=%s after=%s)", before, len(hits))

    logger.info("chat_graph: retrieve done (hits=%s where=%s)", len(hits), where)
    return {"hits": hits, "query_text": query_text, "where": where or {}}


async def _retry_node(
    state: ChatGraphState,
    *,
    settings: Settings,
    store: VectorStore,
) -> ChatGraphState:
    # Retry with the raw user query (skip rewrite), preserving prior-result filtering if requested.
    logger.info("chat_graph: retry start (top_k=%s)\nquery=%r", state.get("top_k"), state.get("query") or "")
    hits = await search_experts(settings=settings, store=store, query=state["query"], top_k=state["top_k"])
    if state.get("use_previous_results") and state.get("previous_result_ids"):
        keep = set(state["previous_result_ids"])
        hits = [h for h in hits if str(h.get("candidate_id")) in keep]
    logger.info("chat_graph: retry done (hits=%s)", len(hits))
    return {"hits": hits, "attempt": int(state.get("attempt") or 0) + 1, "retry_reason": "low_recall_after_rewrite"}


async def _format_base_node(state: ChatGraphState) -> ChatGraphState:
    logger.info("chat_graph: format_base start (hits=%s)", len(state.get("hits") or []))
    base = build_matches_base(hits=state.get("hits") or [])
    logger.info("chat_graph: format_base done (matches=%s)", len(base))
    return {"matches": base}


async def _explain_node(state: ChatGraphState, *, settings: Settings) -> ChatGraphState:
    q = state.get("rewritten_query") or state["query"]
    top_k = int(state.get("top_k") or 10)
    hits = state.get("hits") or []
    logger.info("chat_graph: explain start (hits=%s top_k=%s)", len(hits), top_k)
    explain_map = await explain_matches_llm(settings=settings, query=q, hits=hits)
    logger.info("chat_graph: explain llm_done (explained=%s)", len(explain_map))

    match_dicts = state.get("matches") or []
    patched = 0
    for m in match_dicts:
        cid = str(m.get("candidate_id"))
        info = explain_map.get(cid)
        if not info:
            continue
        m["why_match"] = info.get("why_match") or m.get("why_match")
        m["highlights"] = info.get("highlights") or m.get("highlights") or []
        rs = info.get("relevance_score")
        if rs is not None:
            m["relevance_score"] = float(rs)
        patched += 1

    # Re-rank by LLM relevance score and filter out poor matches.
    min_score = settings.min_rerank_score
    good = [m for m in match_dicts if m.get("relevance_score") is not None and m["relevance_score"] >= min_score]
    good.sort(key=lambda m: m.get("relevance_score", 0.0), reverse=True)

    if good:
        final = good[:top_k]
        logger.info(
            "chat_graph: explain rerank kept=%s/%s (min_score=%.1f top_k=%s)",
            len(final),
            len(match_dicts),
            min_score,
            top_k,
        )
    else:
        # No candidate cleared the threshold — return the best available so the
        # caller still gets something (with honest why_match scores already set).
        all_scored = sorted(
            match_dicts,
            key=lambda m: m.get("relevance_score") or 0.0,
            reverse=True,
        )
        final = all_scored[:top_k]
        logger.warning(
            "chat_graph: explain no candidate cleared min_score=%.1f, returning top %s by score",
            min_score,
            len(final),
        )

    result_ids = [str(m["candidate_id"]) for m in final]
    logger.info("chat_graph: explain done (patched=%s final_matches=%s)", patched, len(final))
    return {
        "matches": final,
        "result_ids": result_ids,
        # persist context for follow-ups
        "previous_query": q,
        "previous_result_ids": result_ids,
    }


class ChatNodes:
    def __init__(self, *, settings: Settings, store: VectorStore) -> None:
        self._settings = settings
        self._store = store

    async def rewrite(self, state: ChatGraphState) -> ChatGraphState:
        return await _rewrite_node(state, settings=self._settings)

    async def retrieve(self, state: ChatGraphState) -> ChatGraphState:
        return await _retrieve_node(state, settings=self._settings, store=self._store)

    async def retry(self, state: ChatGraphState) -> ChatGraphState:
        return await _retry_node(state, settings=self._settings, store=self._store)

    async def explain(self, state: ChatGraphState) -> ChatGraphState:
        return await _explain_node(state, settings=self._settings)

    async def format_base(self, state: ChatGraphState) -> ChatGraphState:
        return await _format_base_node(state)


def build_chat_graph(*, settings: Settings, store: VectorStore, checkpointer=None):
    g: StateGraph[ChatGraphState] = StateGraph(ChatGraphState)
    nodes = ChatNodes(settings=settings, store=store)

    g.add_node("rewrite", nodes.rewrite)
    g.add_node("retrieve", nodes.retrieve)
    g.add_node("retry", nodes.retry)
    g.add_node("format_base", _format_base_node)
    g.add_node("explain", nodes.explain)

    g.add_edge(START, "rewrite")
    g.add_edge("rewrite", "retrieve")
    g.add_conditional_edges("retrieve", _should_retry, {True: "retry", False: "format_base"})
    g.add_edge("retry", "format_base")
    g.add_edge("format_base", "explain")
    g.add_edge("explain", END)

    return g.compile(checkpointer=checkpointer)


async def run_chat_graph(*, graph, query: str, top_k: int, thread_id: str) -> dict[str, Any]:
    state: ChatGraphState = {
        "query": query,
        "top_k": top_k,
    }
    out = await graph.ainvoke(state, {"configurable": {"thread_id": thread_id}})
    return dict(out)

