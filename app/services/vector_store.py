from __future__ import annotations

import chromadb
from chromadb.api import Collection


class VectorStore:
    def __init__(self, *, persist_path: str, collection_name: str) -> None:
        self._persist_path = persist_path
        self._collection_name = collection_name
        self._client = chromadb.PersistentClient(path=persist_path)

    @property
    def collection_name(self) -> str:
        return self._collection_name

    @property
    def persist_path(self) -> str:
        return self._persist_path

    def _collection(self) -> Collection:
        return self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def reset_collection(self) -> None:
        try:
            self._client.delete_collection(self._collection_name)
        except Exception:
            pass
        self._client.create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def count(self) -> int:
        col = self._collection()
        return col.count()

    def upsert_batch(
        self,
        *,
        ids: list[str],
        embeddings: list[list[float]],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        col = self._collection()
        col.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )

    def query(
        self,
        *,
        query_embedding: list[float],
        top_k: int = 10,
    ) -> dict:
        col = self._collection()
        return col.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["metadatas", "documents", "distances"],
        )
