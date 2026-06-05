import logging
import httpx
import json
from typing import AsyncIterator
from groq import AsyncGroq
from backend.config import settings

logger = logging.getLogger(__name__)

class GroqProvider:
    def __init__(self):
        self.client = AsyncGroq(api_key=settings.groq_api_key)
        self.model = settings.llm_model

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        async for chunk in response:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta


class OllamaProvider:
    def __init__(self, model="qwen3:8b", base_url="http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    async def stream(self, prompt: str) -> AsyncIterator[str]:
        async with httpx.AsyncClient() as client:
            async with client.stream(
                "POST", 
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                    "stream": True
                },
                timeout=120.0
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if line:
                        data = json.loads(line)
                        if "message" in data and "content" in data["message"]:
                            yield data["message"]["content"]


class LLMRouter:
    def __init__(self):
        self.primary = GroqProvider()
        self.fallback = OllamaProvider(model="qwen3:8b")

    async def generate(self, prompt: str) -> AsyncIterator[str]:
        stream_iter = self.primary.stream(prompt)
        try:
            # Try to fetch the first token from Groq
            first_token = await anext(stream_iter)
            yield first_token
            # If successful, continue with Groq
            async for token in stream_iter:
                yield token
        except Exception as e:
            logger.error(f"Groq primary provider failed: {e}. Falling back to Ollama (qwen3:8b).")
            # Fallback to local Ollama if Groq fails
            async for token in self.fallback.stream(prompt):
                yield token
