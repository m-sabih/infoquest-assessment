from app.main import logger
from langgraph.prebuilt import create_react_agent
from app.config import Settings
from app.services.chat_client import ChatClient
from typing import Any
from langchain_openrouter import ChatOpenRouter
from app.services.chat_logic import search_experts


async def run_search_agent(*, settings: Settings, query: str, top_k: int, thread_id: str) -> dict[str, Any]:
    model = ChatOpenRouter(
        model=settings.chat_model,
        temperature=0,
        max_tokens=1024,
        max_retries=2,
    )
    
    agent = create_react_agent(
        model=model,
        tools=[search_experts],
        prompt="""You're responsible for preparing a query to search for experts in the vector store. Once the result is returned, you need to review the results
        and compare it against the provided query. If the results are not relevant, you need to rewrite the query and search again.
        For the next search, you need to broaden the horizon of the search For example:
        **Example Query Progression:**
            *User query: “Find me regulatory affairs experts with SFDA experience in the pharmaceutical industry based in Saudi Arabia.”*
            **Iteration 1 (Narrow):**
            Search for experts matching ALL criteria: regulatory affairs role + pharmaceutical industry + Saudi Arabia location + SFDA mentioned in biography or expertise.
            **Iteration 2 (Slightly broader):**
            Relax one constraint — perhaps search for regulatory affairs in pharma across the broader Middle East region, or include adjacent regulatory bodies.
            **Iteration 3 (Broader still):**
            Expand to related roles (e.g., quality assurance, compliance) within the same industry and region.
        """
    )    
    
    out = await agent.ainvoke(
    {
        "messages": [{"role": "user", "content": query}]
    },
    config={
        "configurable": {
            "thread_id": thread_id
            }
        }
    )
    logger.info("search_agent: done (out=%s)", out)
    return out