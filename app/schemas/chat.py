from uuid import UUID
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, description="Natural language expert search query.")
    conversation_id: str | None = Field(
        default=None,
        description="Optional conversation/session id for follow-up queries.",
    )
    top_k: int = Field(default=10, ge=1, le=50, description="Number of experts to return.")


class ExpertMatch(BaseModel):
    candidate_id: UUID
    name: str
    headline: str | None = None
    location: str | None = None
    nationality: str | None = None
    years_of_experience: int | None = None
    score: float = Field(description="Similarity score (higher is better).")
    why_match: str = Field(description="Short explanation of why this expert matches.")
    highlights: list[str] = Field(default_factory=list, description="Key profile highlights.")
    metadata: dict[str, str | int | float | bool] = Field(
        default_factory=dict,
        description="Raw stored metadata from the vector DB (scalar-only).",
    )


class ChatResponse(BaseModel):
    conversation_id: str
    query: str
    rewritten_query: str | None = None
    results: list[ExpertMatch]

