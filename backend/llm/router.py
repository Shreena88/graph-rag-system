import logging
from typing import AsyncIterator

from groq import AsyncGroq

from backend.config import settings

logger = logging.getLogger(__name__)


class LLMRouter:
    def __init__(self):
        self.client = AsyncGroq(api_key=settings.groq_api_key)
        self.model = settings.llm_model

    async def generate(self, prompt: str) -> AsyncIterator[str]:
        """Stream tokens directly from Groq."""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
            )
            async for chunk in response:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as e:
            logger.error("[LLMRouter] Groq request failed: %s", e)
            raise RuntimeError(
                f"Groq LLM unavailable: {e}. Check your API key and network connection."
            )
