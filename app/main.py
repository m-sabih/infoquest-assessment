from contextlib import asynccontextmanager
import logging
import asyncpg
from fastapi import FastAPI

from app.api.routes import candidates, chat, health, ingest
from app.config import get_settings
from app.services.chat_graph import build_chat_graph
from app.services.vector_store import VectorStore
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    dsn = str(settings.database_url)
    if "+asyncpg" in dsn:
        dsn = dsn.replace("postgresql+asyncpg://", "postgresql://", 1)
    app.state.db_pool = await asyncpg.create_pool(dsn=dsn, min_size=1, max_size=8)    
    app.state.vector_store = VectorStore(
        persist_path=settings.chroma_path,
        collection_name=settings.collection_name,
    )

    # LangGraph checkpointer (SQLite) for conversation state.
    saver_cm = AsyncSqliteSaver.from_conn_string(settings.chat_checkpoint_path)
    checkpointer = await saver_cm.__aenter__()
    app.state._chat_checkpointer_cm = saver_cm
    app.state.chat_graph = build_chat_graph(
        settings=settings,
        store=app.state.vector_store,
        checkpointer=checkpointer,
    )

    logger.info("Startup complete (db pool + vector store + chat graph ready).")
    yield    
    await app.state.db_pool.close()    
    await app.state._chat_checkpointer_cm.__aexit__(None, None, None)
    logger.info("Shutdown complete.")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name, lifespan=lifespan)
    app.include_router(health.router, tags=["health"])
    app.include_router(candidates.router, tags=["candidates"])
    app.include_router(ingest.router, tags=["ingest"])
    app.include_router(chat.router, tags=["chat"])
    logger.info("API routes mounted.")
    return app


app = create_app()
