from typing import Any

from fastapi import APIRouter, Query, Request

from app.schemas.candidates import CandidateRecord, CandidatesListResponse
from app.services import candidate_repository

router = APIRouter()


@router.get("/candidates", response_model=CandidatesListResponse)
async def list_candidates(
    request: Request,
    limit: int = Query(default=200, ge=1, le=2000),
    offset: int = Query(default=0, ge=0),
) -> CandidatesListResponse:
    pool = request.app.state.db_pool
    rows: list[dict[str, Any]] = []
    async for page in candidate_repository.iter_candidate_pages(
        pool,
        page_size=limit,
        limit=limit,
        start_offset=offset,
    ):
        rows = page
        break
    return CandidatesListResponse(
        candidates=[CandidateRecord.model_validate(r) for r in rows],
        limit=limit,
        offset=offset,
        count=len(rows),
    )
