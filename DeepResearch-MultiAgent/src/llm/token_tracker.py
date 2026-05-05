from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict


class TokenUsage:
    def __init__(self):
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0
        self.call_count: int = 0

    def add(self, prompt: int, completion: int):
        self.prompt_tokens += prompt
        self.completion_tokens += completion
        self.total_tokens += prompt + completion
        self.call_count += 1


class TokenTracker:
    def __init__(self, session_id: str = ""):
        self.session_id = session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._global: TokenUsage = TokenUsage()
        self._by_model: Dict[str, TokenUsage] = defaultdict(TokenUsage)
        self._by_agent: Dict[str, TokenUsage] = defaultdict(TokenUsage)
        self._by_agent_model: Dict[str, Dict[str, TokenUsage]] = defaultdict(
            lambda: defaultdict(TokenUsage)
        )
        self._history: List[dict] = []

    def add(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        model: str = "unknown",
        agent: str = "",
    ):
        self._global.add(prompt_tokens, completion_tokens)
        self._by_model[model].add(prompt_tokens, completion_tokens)

        if agent:
            self._by_agent[agent].add(prompt_tokens, completion_tokens)
            self._by_agent_model[agent][model].add(prompt_tokens, completion_tokens)

        self._history.append(
            {
                "timestamp": datetime.now().isoformat(),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "model": model,
                "agent": agent,
            }
        )

    @property
    def total_tokens(self) -> int:
        return self._global.total_tokens

    @property
    def total_calls(self) -> int:
        return self._global.call_count

    def get_by_model(self, model: str) -> TokenUsage:
        return self._by_model.get(model, TokenUsage())

    def get_by_agent(self, agent: str) -> TokenUsage:
        return self._by_agent.get(agent, TokenUsage())

    def generate_report(self) -> dict:
        report = {
            "session_id": self.session_id,
            "total_tokens": self.total_tokens,
            "total_calls": self.total_calls,
            "prompt_tokens": self._global.prompt_tokens,
            "completion_tokens": self._global.completion_tokens,
            "by_model": {
                model: {
                    "total": usage.total_tokens,
                    "prompt": usage.prompt_tokens,
                    "completion": usage.completion_tokens,
                    "calls": usage.call_count,
                }
                for model, usage in self._by_model.items()
            },
            "by_agent": {
                agent: {
                    "total": usage.total_tokens,
                    "prompt": usage.prompt_tokens,
                    "completion": usage.completion_tokens,
                    "calls": usage.call_count,
                    "models": {
                        m: {
                            "total": u.total_tokens,
                            "calls": u.call_count,
                        }
                        for m, u in self._by_agent_model[agent].items()
                    },
                }
                for agent, usage in self._by_agent.items()
            },
        }
        return report

    def reset(self):
        self._global = TokenUsage()
        self._by_model.clear()
        self._by_agent.clear()
        self._by_agent_model.clear()
        self._history.clear()
