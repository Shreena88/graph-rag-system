from typing import List
from backend.models.query import ParsedQuery, EntityRef
from backend.nlp.entity_extractor import EntityExtractor


class QueryParser:
    INTENT_PATTERNS = {
        "causal": ["why", "because", "cause", "reason", "effect", "lead to"],
        "procedural": ["how", "steps", "process", "procedure", "implement"],
        "comparative": ["compare", "difference", "versus", "vs", "better", "worse"],
    }

    def __init__(self, entity_extractor: EntityExtractor):
        self.extractor = entity_extractor

    def parse(self, query: str) -> ParsedQuery:
        intent = self._classify_intent(query)
        extraction = self.extractor.extract(query)
        entities = [
            EntityRef(name=e.name, type=e.type, confidence=e.confidence)
            for e in extraction.entities
        ]
        return ParsedQuery(
            original=query,
            intent=intent,
            entities=entities,
            keywords=self._extract_keywords(query),
            sub_questions=self._decompose(query, intent, entities),
        )

    def _classify_intent(self, query: str) -> str:
        q_lower = query.lower()
        for intent, patterns in self.INTENT_PATTERNS.items():
            if any(p in q_lower for p in patterns):
                return intent
        return "factual"

    def _extract_keywords(self, query: str) -> List[str]:
        stop_words = {"what", "is", "the", "a", "an", "of", "in", "and", "or", "how", "why"}
        return [w for w in query.lower().split() if w not in stop_words and len(w) > 2]

    def _decompose(self, query: str, intent: str, entities: List[EntityRef]) -> List[str]:
        if intent == "causal" and entities:
            return [f"What is {e.name}?" for e in entities] + [query]
        if intent == "comparative" and len(entities) >= 2:
            return [
                f"What are the properties of {entities[0].name}?",
                f"What are the properties of {entities[1].name}?",
                query,
            ]
        return [query]
