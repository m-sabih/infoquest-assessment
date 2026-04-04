from fastapi import FastAPI


def create_app() -> FastAPI:
    return FastAPI(title="InfoQuest Expert Search API")


app = create_app()
