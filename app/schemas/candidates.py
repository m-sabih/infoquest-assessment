from uuid import UUID

from pydantic import BaseModel, Field


class CandidateRecord(BaseModel):
    """One denormalized candidate row from the repository query."""

    model_config = {"extra": "ignore"}

    id: UUID
    first_name: str | None = None
    last_name: str | None = None
    email: str | None = None
    headline: str | None = None
    years_of_experience: float | None = None
    nationality: str | None = None
    city_name: str | None = None
    country_name: str | None = None
    skills_text: str | None = None
    languages_text: str | None = None
    work_text: str | None = None
    education_text: str | None = None


class CandidatesListResponse(BaseModel):
    candidates: list[CandidateRecord]
    limit: int
    offset: int
    count: int = Field(description="Number of candidates returned in this response.")
