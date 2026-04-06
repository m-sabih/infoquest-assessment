from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request, status

from app.schemas.chat import ChatRequest, ChatResponse, ExpertMatch
from app.services.chat_graph import run_chat_graph


router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, body: ChatRequest) -> ChatResponse:
    settings = request.app.state.settings
    graph = request.app.state.chat_graph
    conversation_id = body.conversation_id or uuid4().hex

    if not settings.openrouter_api_key.strip():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OPENROUTER_API_KEY is not set. It is required for embeddings during /chat.",
        )

    out = await run_chat_graph(
        graph=graph,
        query=body.query,
        top_k=body.top_k,
        thread_id=conversation_id,
    )
    rewritten_query = out.get("rewritten_query")
    match_dicts = out.get("matches") or []
    matches = [ExpertMatch.model_validate(m) for m in match_dicts]

    return ChatResponse(
        conversation_id=conversation_id,
        query=body.query,
        rewritten_query=rewritten_query if rewritten_query and rewritten_query != body.query else None,
        results=matches,
    )

