import asyncio
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.graph.neo4j_client import neo4j_client
from backend.graph.traversal import GraphTraversal
from backend.nlp.entity_extractor import EntityExtractor
from backend.nlp.query_parser import QueryParser

async def test_neo4j_traversal():
    print("Connecting to Neo4j...")
    await neo4j_client.connect()
    
    try:
        extractor = EntityExtractor()
        parser = QueryParser(extractor)
        traversal = GraphTraversal(neo4j_client)
        
        # Ask for input or use a test query
        query = input("\nEnter a question to test: ")
        
        print(f"\n[1] Parsing query: '{query}'")
        parsed = parser.parse(query)
        print(f"    Intent: {parsed.intent}")
        print(f"    Entities Extracted: {[e.name for e in parsed.entities]}")
        
        if not parsed.entities:
            print("\nWARNING: No entities found in the question! Graph traversal will likely fail or return empty.")
        else:
            print("\n[2] Running Neo4j Multi-Hop Traversal...")
            results = await traversal.multi_hop(parsed)
            
            bfs_chunks = results.get("bfs_chunks", [])
            dfs_paths = results.get("dfs_paths", [])
            
            print(f"\n--- Traversal Results ---")
            print(f"Total Chunks Found: {len(bfs_chunks)}")
            for i, chunk in enumerate(bfs_chunks):
                # chunk is a TraversalNode object
                print(f"  Chunk {i+1} [Hop Distance: {chunk.hop_distance}]: {chunk.text[:100]}...")
                
            if dfs_paths:
                print(f"\nTotal Relationship Paths Found (DFS): {len(dfs_paths)}")
                for i, path in enumerate(dfs_paths):
                    print(f"  Path {i+1} [Depth: {path.depth}]: {path.rel_types}")
            else:
                print("\nNo DFS paths found (usually requires 2+ entities and causal/procedural intent).")
                
    except Exception as e:
        print(f"Error during traversal: {e}")
        
    finally:
        print("\nClosing connection...")
        await neo4j_client.close()

if __name__ == "__main__":
    # Ensure this script is run from the root of graph-rag-system
    asyncio.run(test_neo4j_traversal())
