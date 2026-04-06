from pydantic import BaseModel, Field


class IngestRequest(BaseModel):
    """Optional controls for the ingestion run."""

    replace_collection: bool = Field(
        default=True,
        description="If true, drops and recreates the vector collection before indexing.",
    )

    limit: int | None = Field(
        default=None,
        ge=1,
        description="Optional maximum number of candidates to ingest (None means no limit).",
    )


class IngestResponse(BaseModel):
    status: str
    candidates_fetched: int
    vectors_upserted: int
    errors: int
    duration_seconds: float
    message: str | None = None
