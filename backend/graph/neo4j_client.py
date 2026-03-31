import logging
from neo4j import AsyncGraphDatabase, AsyncDriver
from typing import List, Dict, Any
from backend.config import settings

# Suppress Neo4j notification warnings about missing labels/properties
logging.getLogger("neo4j").setLevel(logging.ERROR)


class Neo4jClient:
    _driver: AsyncDriver = None

    async def connect(self):
        self._driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
            max_connection_pool_size=50,
        )

    async def close(self):
        if self._driver:
            await self._driver.close()

    async def run(self, cypher: str, **params) -> List[Dict[str, Any]]:
        if self._driver is None:
            return []
        for attempt in range(3):
            try:
                async with self._driver.session() as session:
                    result = await session.run(cypher, **params)
                    return [dict(record) async for record in result]
            except Exception as e:
                if attempt < 2:
                    import asyncio
                    await asyncio.sleep(1)
                else:
                    logging.getLogger(__name__).error("Neo4j query failed: %s", e)
                    return []

    async def run_write(self, cypher: str, **params):
        if self._driver is None:
            logging.getLogger(__name__).warning("Neo4j not connected — skipping write")
            return
        for attempt in range(3):
            try:
                async with self._driver.session() as session:
                    await session.run(cypher, **params)
                return
            except Exception as e:
                if attempt < 2:
                    import asyncio
                    await asyncio.sleep(1)
                else:
                    raise

    async def init_schema(self):
        statements = [
            "CREATE CONSTRAINT doc_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.canonical_name IS UNIQUE",
            "CREATE INDEX chunk_section IF NOT EXISTS FOR (c:Chunk) ON (c.section_id)",
            "CREATE INDEX entity_type IF NOT EXISTS FOR (e:Entity) ON (e.type)",
        ]
        for stmt in statements:
            try:
                await self.run_write(stmt)
            except Exception as e:
                # Log but don't fail startup on schema conflicts
                logging.getLogger(__name__).warning("Schema statement skipped: %s", e)


neo4j_client = Neo4jClient()
