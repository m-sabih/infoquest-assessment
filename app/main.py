from fastapi import FastAPI

from app.schemas.health import HealthResponse


def create_app() -> FastAPI:
    application = FastAPI(title="InfoQuest Expert Search API")

    @application.get("/health", response_model=HealthResponse, tags=["health"])
    def health() -> HealthResponse:
        return HealthResponse()

    return application


app = create_app()
