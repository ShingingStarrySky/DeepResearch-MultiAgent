from __future__ import annotations

import time
from typing import Optional, Any

import httpx
import requests


class DeepSeekClient:
    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.deepseek.com/v1",
        max_retries: int = 3,
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.max_retries = max_retries
        self._session: Optional[requests.Session] = None

    @property
    def session(self) -> requests.Session:
        if self._session is None:
            self._session = requests.Session()
            self._session.headers.update(
                {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                }
            )
        return self._session

    def _make_request(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> dict:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        for attempt in range(self.max_retries):
            try:
                resp = self.session.post(
                    f"{self.base_url}/chat/completions",
                    json=payload,
                    timeout=180,
                )

                if resp.status_code == 429:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    continue

                resp.raise_for_status()
                data = resp.json()
                return {
                    "content": data["choices"][0]["message"]["content"],
                    "prompt_tokens": data.get("usage", {}).get("prompt_tokens", 0),
                    "completion_tokens": data.get("usage", {}).get("completion_tokens", 0),
                    "model": model,
                }
            except Exception as e:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(2 ** attempt)

        raise RuntimeError("Max retries exceeded for DeepSeek API call")

    async def _make_request_async(
        self,
        messages: list[dict],
        model: str,
        temperature: float,
        max_tokens: int,
    ) -> dict:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        async with httpx.AsyncClient(timeout=180) as client:
            for attempt in range(self.max_retries):
                try:
                    resp = await client.post(
                        f"{self.base_url}/chat/completions",
                        headers=headers,
                        json=payload,
                    )

                    if resp.status_code == 429:
                        await __import__("asyncio").sleep(2 ** attempt)
                        continue

                    resp.raise_for_status()
                    data = resp.json()
                    return {
                        "content": data["choices"][0]["message"]["content"],
                        "prompt_tokens": data.get("usage", {}).get("prompt_tokens", 0),
                        "completion_tokens": data.get("usage", {}).get("completion_tokens", 0),
                        "model": model,
                    }
                except Exception as e:
                    if attempt == self.max_retries - 1:
                        raise
                    await __import__("asyncio").sleep(2 ** attempt)

        raise RuntimeError("Max retries exceeded for DeepSeek API call (async)")

    def chat_sync(
        self,
        messages: list[dict],
        model: str = "deepseek-chat",
        temperature: float = 0.3,
        max_tokens: int = 8192,
    ) -> dict:
        return self._make_request(messages, model, temperature, max_tokens)

    async def chat(
        self,
        messages: list[dict],
        model: str = "deepseek-chat",
        temperature: float = 0.3,
        max_tokens: int = 8192,
    ) -> dict:
        return await self._make_request_async(messages, model, temperature, max_tokens)

    def close(self):
        if self._session:
            self._session.close()
            self._session = None

    def __del__(self):
        self.close()
