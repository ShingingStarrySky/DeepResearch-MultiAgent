from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from config.settings import get_settings
from src.llm.provider import LLMProvider, ProviderType, create_provider
from src.llm.token_tracker import TokenTracker
from src.utils.logger import get_logger


class BaseAgent(ABC):
    def __init__(
        self,
        name: str,
        provider_type: ProviderType = ProviderType.DEEPSEEK,
        token_tracker: Optional[TokenTracker] = None,
    ):
        self.name = name
        self.settings = get_settings()
        self.token_tracker = token_tracker or TokenTracker()
        self.provider: LLMProvider = create_provider(provider_type, self.token_tracker)
        self.logger = get_logger(f"Agent.{name}")

    def log(self, message: str, level: str = "INFO"):
        if level == "WARNING":
            self.logger.warning(message)
        elif level == "ERROR":
            self.logger.error(message)
        else:
            self.logger.info(message)

    def _build_system_message(self, system_prompt: str) -> dict:
        return {"role": "system", "content": system_prompt}

    def _build_user_message(self, content: str) -> dict:
        return {"role": "user", "content": content}

    def _call_llm(
        self,
        messages: list[dict],
        model: str = "",
        temperature: float = 0.3,
        max_tokens: int = 8192,
    ) -> str:
        return self.provider.chat_sync(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    async def _call_llm_async(
        self,
        messages: list[dict],
        model: str = "",
        temperature: float = 0.3,
        max_tokens: int = 8192,
    ) -> str:
        return await self.provider.chat(
            messages=messages,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    @abstractmethod
    def execute(self, *args, **kwargs):
        pass

    def get_status(self) -> dict:
        return {
            "agent_name": self.name,
            "total_tokens": self.token_tracker.total_tokens,
            "total_calls": self.token_tracker.total_calls,
        }

    def get_token_report(self) -> dict:
        return self.token_tracker.generate_report()
