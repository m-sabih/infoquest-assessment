from __future__ import annotations

from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from app.config import Settings
from app.services.chat_logic import build_matches_base, explain_matches_llm, rewrite_query, search_experts
from app.services.vector_store import VectorStore


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
    return len(hits) < max(3, int(state.get("top_k") or 10) // 3)


async def _rewrite_node(state: ChatGraphState, *, settings: Settings) -> ChatGraphState:
    rr = await rewrite_query(
        settings=settings,
        query=state["query"],
        previous_query=state.get("previous_query"),
        previous_result_ids=state.get("previous_result_ids"),
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
    hits = await search_experts(settings=settings, store=store, query=q, top_k=state["top_k"])

    # If the user asked to filter prior results, restrict to those ids.
    if state.get("use_previous_results") and state.get("previous_result_ids"):
        keep = set(state["previous_result_ids"])
        hits = [h for h in hits if str(h.get("candidate_id")) in keep]

    return {"hits": hits}


async def _retry_node(
    state: ChatGraphState,
    *,
    settings: Settings,
    store: VectorStore,
) -> ChatGraphState:
    # Retry with the raw user query (skip rewrite), preserving prior-result filtering if requested.
    hits = await search_experts(settings=settings, store=store, query=state["query"], top_k=state["top_k"])
    if state.get("use_previous_results") and state.get("previous_result_ids"):
        keep = set(state["previous_result_ids"])
        hits = [h for h in hits if str(h.get("candidate_id")) in keep]
    return {"hits": hits, "attempt": int(state.get("attempt") or 0) + 1, "retry_reason": "low_recall_after_rewrite"}


async def _format_base_node(state: ChatGraphState) -> ChatGraphState:
    base = build_matches_base(hits=state.get("hits") or [])
    return {"matches": base}


async def _explain_node(state: ChatGraphState, *, settings: Settings) -> ChatGraphState:
    q = state.get("rewritten_query") or state["query"]
    hits = state.get("hits") or []
    explain_map = await explain_matches_llm(settings=settings, query=q, hits=hits)

    match_dicts = state.get("matches") or []
    for m in match_dicts:
        cid = str(m.get("candidate_id"))
        info = explain_map.get(cid)
        if not info:
            continue
        m["why_match"] = info.get("why_match") or m.get("why_match")
        m["highlights"] = info.get("highlights") or m.get("highlights") or []

    result_ids = [str(m["candidate_id"]) for m in match_dicts]
    return {
        "matches": match_dicts,
        "result_ids": result_ids,
        # persist context for follow-ups
        "previous_query": q,
        "previous_result_ids": result_ids,
    }


def build_chat_graph(*, settings: Settings, store: VectorStore, checkpointer=None):
    g: StateGraph[ChatGraphState] = StateGraph(ChatGraphState)

    g.add_node("rewrite", lambda s: _rewrite_node(s, settings=settings))
    g.add_node("retrieve", lambda s: _retrieve_node(s, settings=settings, store=store))
    g.add_node("retry", lambda s: _retry_node(s, settings=settings, store=store))
    g.add_node("format_base", _format_base_node)
    g.add_node("explain", lambda s: _explain_node(s, settings=settings))

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

