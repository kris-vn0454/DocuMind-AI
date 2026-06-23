import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    content: str
    model: str
    input_tokens: int
    output_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


class GrokClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = "grok-3",
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ):
        self.api_key = api_key or os.getenv("XAI_API_KEY")
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._client = None

    @property
    def client(self):
        if self._client is None:
            if not self.api_key:
                raise ValueError(
                    "XAI_API_KEY not set. Add it to your .env file or set the environment variable."
                )
            try:
                from openai import OpenAI
                self._client = OpenAI(
                    api_key=self.api_key,
                    base_url="https://api.x.ai/v1",
                )
            except ImportError:
                raise ImportError("Install openai: pip install openai")
        return self._client

    def generate(self, prompt: str, system_prompt: str | None = None) -> LLMResponse:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        logger.debug(f"Calling {self.model} with {len(prompt)} chars prompt...")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )

    def is_available(self) -> bool:
        return bool(self.api_key or os.getenv("XAI_API_KEY"))


# Alias for backward compatibility
ClaudeClient = GrokClient
