from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.routes import documents, query, graph
from backend.config import settings
from backend.graph.neo4j_client import neo4j_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: connect to Neo4j with retry
    import asyncio
    connected = False
    for attempt in range(5):
        try:
            await neo4j_client.connect()
            connected = True
            print("[Neo4j] Connected successfully")
            break
        except Exception as e:
            if attempt < 4:
                print(f"[Neo4j] Connection attempt {attempt+1} failed, retrying in 3s...")
                await asyncio.sleep(3)
            else:
                print(f"[Neo4j] Not available: {e}. Graph features disabled.")
    if connected:
        # Schema init errors (e.g. index already exists) should not kill startup
        try:
            await neo4j_client.init_schema()
        except Exception as e:
            print(f"[Neo4j] Schema init warning: {e}")
    yield
    try:
        await neo4j_client.close()
    except Exception:
        pass


app = FastAPI(title="Graph RAG System", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router, prefix="/api/documents", tags=["documents"])
app.include_router(query.router, prefix="/api/query", tags=["query"])
app.include_router(graph.router, prefix="/api/graph", tags=["graph"])


@app.get("/health")
async def health():
    neo4j_ok = False
    neo4j_error = None
    if neo4j_client._driver is not None:
        try:
            result = await neo4j_client.run("RETURN 1 AS ping")
            neo4j_ok = len(result) > 0
        except Exception as e:
            neo4j_error = str(e)
    else:
        neo4j_error = "Driver not initialized"

    return {
        "status": "ok",
        "neo4j": {
            "connected": neo4j_ok,
            "error": neo4j_error,
        }
    }
