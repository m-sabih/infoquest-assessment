from contextlib import asynccontextmanager

import asyncpg
from fastapi import FastAPI

from app.api.routes import candidates, health
from app.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    dsn = str(settings.database_url)
    if "+asyncpg" in dsn:
        dsn = dsn.replace("postgresql+asyncpg://", "postgresql://", 1)
    app.state.db_pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=8)    
    yield
    await app.state.db_pool.close()    


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    app.include_router(health.router, tags=["health"])
    app.include_router(candidates.router, tags=["candidates"])
    return app


app = create_app()
