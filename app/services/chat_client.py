from __future__ import annotations

import json

import httpx


class ChatClient:
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        timeout_seconds: float = 120.0,
    ) -> None:
        self._api_key = api_key
        self._base = base_url.rstrip("/")
        self._model = model
        self._timeout = timeout_seconds

    async def json_completion(self, *, system: str, user: str) -> dict:
        """
        Ask for a JSON object and parse it.

        This uses OpenRouter's chat completions. We keep the response-formatting
        robust by (a) forcing JSON-only instructions and (b) extracting/parsing
        the first JSON object we can find.
        """
        if not self._api_key:
            raise ValueError("API_KEY is not set")

        url = f"{self._base}/chat/completions"
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self._model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        async with httpx.AsyncClient(timeout=self._timeout) as client:
            resp = await client.post(url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

        content = (
            (((data.get("choices") or [{}])[0]).get("message") or {}).get("content")
            or ""
        ).strip()
        if not content:
            return {}

        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1 or end <= start:
            return {}
        return json.loads(content[start : end + 1])

