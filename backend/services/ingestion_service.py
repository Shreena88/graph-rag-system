import uuid
import logging
from datetime import datetime
from backend.nlp.ocr import DocumentParser
from backend.nlp.chunker import SemanticChunker
from backend.nlp.entity_extractor import EntityExtractor
from backend.graph.neo4j_client import neo4j_client
from backend.vector.store import vector_store
from backend.models.document import DocumentResult
from backend.services import redis_client
from backend.config import settings

logger = logging.getLogger(__name__)

parser = DocumentParser(use_gpu=settings.use_gpu)
chunker = SemanticChunker()
extractor = EntityExtractor()


async def register_document(filename: str) -> str:
    """Register a document and return its ID immediately (before ingestion starts)."""
    doc_id = str(uuid.uuid4())
    await redis_client.set_status(doc_id, DocumentResult(
        doc_id=doc_id, filename=filename, status="processing"
    ))
    return doc_id


async def ingest_document(file_path: str, filename: str, doc_id: str = None) -> str:
    if doc_id is None:
        doc_id = str(uuid.uuid4())
        await redis_client.set_status(doc_id, DocumentResult(
            doc_id=doc_id, filename=filename, status="processing"
        ))
    try:
        # 1. Parse
        parsed = parser.parse(file_path, filename)

        # 2. Chunk
        chunks = chunker.chunk(parsed.pages)

        # 3. Store document node
        await neo4j_client.run_write(
            "MERGE (d:Document {id: $id}) SET d.filename = $filename, d.created_at = $ts",
            id=doc_id, filename=filename, ts=datetime.utcnow().isoformat()
        )

        # 4. Extract entities in batch
        entity_count = 0
        texts = [chunk.text for chunk in chunks]
        extractions = list(extractor.extract_batch(texts))

        for chunk, extraction in zip(chunks, extractions):
            entity_count += len(extraction.entities)

            await neo4j_client.run_write(
                """
                MERGE (c:Chunk {id: $id})
                SET c.text = $text, c.page_number = $page, c.token_count = $tokens
                WITH c
                MATCH (d:Document {id: $doc_id})
                MERGE (d)-[:HAS_CHUNK]->(c)
                """,
                id=chunk.chunk_id, text=chunk.text,
                page=chunk.page_number, tokens=chunk.token_count,
                doc_id=doc_id,
            )

            for entity in extraction.entities:
                await neo4j_client.run_write(
                    """
                    MERGE (e:Entity {canonical_name: $name})
                    SET e.type = $type
                    WITH e
                    MATCH (c:Chunk {id: $chunk_id})
                    MERGE (c)-[:MENTIONS]->(e)
                    """,
                    name=entity.canonical_name, type=entity.type, chunk_id=chunk.chunk_id,
                )

            # Persist entity-to-entity relations (SVO + co-occurrence)
            for rel in extraction.relations:
                if rel.subject and rel.object and rel.subject != rel.object:
                    await neo4j_client.run_write(
                        """
                        MERGE (a:Entity {canonical_name: $subj})
                        MERGE (b:Entity {canonical_name: $obj})
                        MERGE (a)-[r:RELATED_TO {predicate: $pred}]->(b)
                        ON CREATE SET r.weight = 1
                        ON MATCH SET r.weight = r.weight + 1
                        """,
                        subj=rel.subject, obj=rel.object, pred=rel.predicate,
                    )

            vector_store.add(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                doc_id=doc_id,
                page_number=chunk.page_number,
            )

        await redis_client.set_status(doc_id, DocumentResult(
            doc_id=doc_id, filename=filename, status="indexed",
            chunk_count=len(chunks), entity_count=entity_count,
        ))
    except Exception as e:
        logger.exception("Ingestion failed for %s: %s", filename, e)
        await redis_client.set_status(doc_id, DocumentResult(
            doc_id=doc_id, filename=filename, status="failed", error=str(e)
        ))

    return doc_id


async def get_status(doc_id: str) -> DocumentResult | None:
    return await redis_client.get_status(doc_id)
