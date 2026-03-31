from fastapi import APIRouter, Query
from backend.graph.neo4j_client import neo4j_client

router = APIRouter()


@router.get("/entities")
async def get_entities(doc_id: str = Query(None), limit: int = 100):
    if neo4j_client._driver is None:
        return []
    try:
        cypher = """
        MATCH (e:Entity)<-[:MENTIONS]-(c:Chunk)<-[:HAS_CHUNK]-(d:Document)
        WHERE $doc_id IS NULL OR d.id = $doc_id
        RETURN e.canonical_name AS name, e.type AS type, count(c) AS mentions
        ORDER BY mentions DESC LIMIT $limit
        """
        return await neo4j_client.run(cypher, doc_id=doc_id, limit=limit)
    except Exception:
        return []


@router.get("/edges")
async def get_edges(doc_id: str = Query(None), limit: int = 200):
    if neo4j_client._driver is None:
        return []
    try:
        # Direct RELATED_TO edges (SVO + co-occurrence stored during ingestion)
        cypher = """
        MATCH (a:Entity)-[r:RELATED_TO]->(b:Entity)
        WHERE $doc_id IS NULL OR EXISTS {
            MATCH (a)<-[:MENTIONS]-(c:Chunk)<-[:HAS_CHUNK]-(d:Document {id: $doc_id})
        }
        RETURN a.canonical_name AS source, b.canonical_name AS target,
               r.predicate AS predicate, r.weight AS weight
        ORDER BY r.weight DESC LIMIT $limit
        """
        return await neo4j_client.run(cypher, doc_id=doc_id, limit=limit)
    except Exception:
        return []


@router.get("/neighbors")
async def get_neighbors(entity: str, depth: int = Query(default=2, le=4)):
    if neo4j_client._driver is None:
        return []
    try:
        cypher = f"""
        MATCH path = (e:Entity {{canonical_name: $entity}})-[*1..{depth}]-(n:Entity)
        RETURN DISTINCT n.canonical_name AS name, n.type AS type
        LIMIT 50
        """
        return await neo4j_client.run(cypher, entity=entity.lower())
    except Exception:
        return []
