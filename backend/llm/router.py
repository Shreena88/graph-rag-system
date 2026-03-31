from typing import AsyncIterator
from groq import AsyncGroq
from backend.config import settings


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


class LLMRouter:
    def __init__(self):
        self.provider = GroqProvider()

    async def generate(self, prompt: str) -> AsyncIterator[str]:
        async for token in self.provider.stream(prompt):
            yield token
