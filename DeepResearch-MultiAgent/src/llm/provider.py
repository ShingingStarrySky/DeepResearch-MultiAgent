from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, Any

from config.settings import DeepSeekSettings, GLMSettings, OpenAISettings

from src.llm.token_tracker import TokenTracker


class ProviderType(str, Enum):
    DEEPSEEK = "deepseek"
    GLM = "glm"
    OPENAI = "openai"


class LLMProvider(ABC):
    def __init__(self, token_tracker: Optional[TokenTracker] = None):
        self.token_tracker = token_tracker

    @abstractmethod
    async def chat(
        self,
        messages: list[dict],
        model: str = "",
        temperature: float = 0.3,
        max_tokens: int = 8192,
    ) -> str:
        pass

    @abstractmethod
    def chat_sync(
        self,
        messages: list[dict],
        model: str = "",
        temperature: float = 0.3,
        max_tokens: int = 8192,
    ) -> str:
        pass

    def _track_tokens(self, prompt_tokens: int, completion_tokens: int, model: str):
        if self.token_tracker:
            self.token_tracker.add(prompt_tokens, completion_tokens, model)


class DeepSeekProvider(LLMProvider):
    def __init__(
        self,
        settings: DeepSeekSettings,
        token_tracker: Optional[TokenTracker] = None,
    ):
        super().__init__(token_tracker)
        from src.llm.deepseek_client import DeepSeekClient

        self.client = DeepSeekClient(
            api_key=settings.api_key,
            base_url=settings.base_url,
        )
        self.pro_model = settings.pro_model
        self.flash_model = settings.flash_model

    async def chat(
        self,
        messages: list[dict],
        model: str = "",
        temperature: float = 0.3,
        max_tokens: int = 8192,
    ) -> str:
        model = model or self.pro_model
        response = await self.client.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self._track_tokens(
            response.get("prompt_tokens", 0),
            response.get("completion_tokens", 0),
            model,
        )
        return response["content"]

    def chat_sync(
        self,
        messages: list[dict],
        model: str = "",
        temperature: float = 0.3,
        max_tokens: int = 8192,
    ) -> str:
        model = model or self.pro_model
        response = self.client.chat_sync(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        self._track_tokens(
            response.get("prompt_tokens", 0),
            response.get("completion_tokens", 0),
            model,
        )
        return response["content"]


class GLMProvider(LLMProvider):
    def __init__(
        self,
        settings: GLMSettings,
        token_tracker: Optional[TokenTracker] = None,
    ):
        super().__init__(token_tracker)
        self.settings = settings

    async def chat(
        self,
        messages: list[dict],
        model: str = "",
        temperature: float = 0.5,
        max_tokens: int = 4096,
    ) -> str:
        import httpx

        model = model or self.settings.model
        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{self.settings.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            data = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        self._track_tokens(
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
            model,
        )
        return content

    def chat_sync(
        self,
        messages: list[dict],
        model: str = "",
        temperature: float = 0.5,
        max_tokens: int = 4096,
    ) -> str:
        import requests

        model = model or self.settings.model
        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = requests.post(
            f"{self.settings.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=120,
        )
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        self._track_tokens(
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
            model,
        )
        return content


class OpenAIProvider(LLMProvider):
    def __init__(
        self,
        settings: OpenAISettings,
        token_tracker: Optional[TokenTracker] = None,
    ):
        super().__init__(token_tracker)
        self.settings = settings

    async def chat(
        self,
        messages: list[dict],
        model: str = "",
        temperature: float = 0.3,
        max_tokens: int = 8192,
    ) -> str:
        import httpx

        model = model or self.settings.model
        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{self.settings.base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
            data = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        self._track_tokens(
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
            model,
        )
        return content

    def chat_sync(
        self,
        messages: list[dict],
        model: str = "",
        temperature: float = 0.3,
        max_tokens: int = 8192,
    ) -> str:
        import requests

        model = model or self.settings.model
        headers = {
            "Authorization": f"Bearer {self.settings.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        resp = requests.post(
            f"{self.settings.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=120,
        )
        data = resp.json()
        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        self._track_tokens(
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
            model,
        )
        return content


def create_provider(
    provider_type: ProviderType,
    token_tracker: Optional[TokenTracker] = None,
) -> LLMProvider:
    from config.settings import get_settings

    settings = get_settings()

    if provider_type == ProviderType.DEEPSEEK:
        return DeepSeekProvider(settings.deepseek, token_tracker)
    elif provider_type == ProviderType.GLM:
        return GLMProvider(settings.glm, token_tracker)
    elif provider_type == ProviderType.OPENAI:
        return OpenAIProvider(settings.openai, token_tracker)
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")
