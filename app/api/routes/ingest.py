from fastapi import APIRouter, HTTPException, Request, status

from app.schemas.ingest import IngestRequest, IngestResponse
from app.services.ingestion import run_ingestion

router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest(request: Request, body: IngestRequest | None = None) -> IngestResponse:
    if body is None:
        body = IngestRequest()
    settings = request.app.state.settings
    pool = request.app.state.db_pool
    store = request.app.state.vector_store

    result = await run_ingestion(settings=settings, pool=pool, store=store, body=body)
    if result.status == "failed" and result.message:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=result.message)
    return result
