from typing import List
from backend.graph.neo4j_client import Neo4jClient
from backend.nlp.chunker import Chunk
from backend.nlp.entity_extractor import ExtractionResult


class GraphBuilder:
    def __init__(self, client: Neo4jClient):
        self.client = client

    async def build_from_extraction(
        self,
        doc_id: str,
        chunks: List[Chunk],
        extractions: List[ExtractionResult],
    ) -> dict:
        chunk_count = 0
        entity_count = 0

        for chunk, extraction in zip(chunks, extractions):
            # Upsert chunk
            await self.client.run_write(
                """
                MERGE (c:Chunk {id: $id})
                SET c.text = $text, c.page_number = $page, c.token_count = $tokens
                WITH c MATCH (d:Document {id: $doc_id})
                MERGE (d)-[:HAS_CHUNK]->(c)
                """,
                id=chunk.chunk_id, text=chunk.text,
                page=chunk.page_number, tokens=chunk.token_count,
                doc_id=doc_id,
            )
            chunk_count += 1

            # Upsert entities + MENTIONS edges
            for entity in extraction.entities:
                await self.client.run_write(
                    """
                    MERGE (e:Entity {canonical_name: $name})
                    SET e.type = $type
                    WITH e MATCH (c:Chunk {id: $chunk_id})
                    MERGE (c)-[:MENTIONS {confidence: $conf}]->(e)
                    """,
                    name=entity.canonical_name, type=entity.type,
                    chunk_id=chunk.chunk_id, conf=entity.confidence,
                )
                entity_count += 1

            # Upsert relations as edges between entities
            for rel in extraction.relations:
                await self.client.run_write(
                    """
                    MERGE (a:Entity {canonical_name: $subj})
                    MERGE (b:Entity {canonical_name: $obj})
                    MERGE (a)-[:RELATED_TO {predicate: $pred}]->(b)
                    """,
                    subj=rel.subject.lower(), obj=rel.object.lower(), pred=rel.predicate,
                )

        return {"chunk_count": chunk_count, "entity_count": entity_count}
