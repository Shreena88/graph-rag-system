from typing import List, Optional
from backend.graph.neo4j_client import Neo4jClient
from backend.models.query import ParsedQuery


class TraversalNode:
    def __init__(self, chunk_id: str, text: str, hop_distance: int):
        self.chunk_id = chunk_id
        self.text = text
        self.hop_distance = hop_distance


class TraversalPath:
    def __init__(self, nodes: list, rel_types: list, depth: int):
        self.nodes = nodes
        self.rel_types = rel_types
        self.depth = depth


class GraphTraversal:
    def __init__(self, client: Neo4jClient):
        self.client = client

    async def bfs(
        self,
        seed_entities: List[str],
        max_depth: int = 3,
        relationship_filter: Optional[List[str]] = None,
    ) -> List[TraversalNode]:
        rel_clause = f":{('|'.join(relationship_filter))}" if relationship_filter else ""
        cypher = f"""
        MATCH path = (e:Entity)-[r{rel_clause}*1..{max_depth}]-(c:Chunk)
        WHERE e.canonical_name IN $seeds
        WITH c, min(length(path)) AS hop_distance
        RETURN c.id AS chunk_id, c.text AS text, hop_distance
        ORDER BY hop_distance ASC
        LIMIT 50
        """
        results = await self.client.run(cypher, seeds=seed_entities)
        return [TraversalNode(**r) for r in results]

    async def dfs_path(
        self,
        start_entity: str,
        end_entity: str,
        max_depth: int = 5,
    ) -> List[TraversalPath]:
        cypher = f"""
        MATCH path = shortestPath(
            (a:Entity {{canonical_name: $start}})-[*1..{max_depth}]-(b:Entity {{canonical_name: $end}})
        )
        RETURN [node IN nodes(path) | node] AS nodes,
               [rel IN relationships(path) | type(rel)] AS rel_types,
               length(path) AS depth
        ORDER BY depth ASC
        LIMIT 5
        """
        results = await self.client.run(cypher, start=start_entity, end=end_entity)
        return [TraversalPath(**r) for r in results]

    async def multi_hop(self, query: ParsedQuery) -> dict:
        bfs_results = await self.bfs(
            seed_entities=[e.name.lower().strip() for e in query.entities],
            max_depth=query.max_hops if hasattr(query, "max_hops") else 3,
        )
        dfs_results = []
        if query.intent in ("causal", "procedural") and len(query.entities) >= 2:
            for i in range(len(query.entities) - 1):
                paths = await self.dfs_path(
                    query.entities[i].name.lower().strip(),
                    query.entities[i + 1].name.lower().strip(),
                )
                dfs_results.extend(paths)

        return {"bfs_chunks": bfs_results, "dfs_paths": dfs_results}
