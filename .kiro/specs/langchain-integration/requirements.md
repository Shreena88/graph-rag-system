# Requirements Document

## Introduction

This feature integrates LangChain into the existing Graph RAG system without replacing its
current architecture. The integration targets three specific layers where LangChain adds
measurable value: (1) wrapping the Neo4j BFS/DFS traversal as a LangChain Tool so it can
participate in tool-calling workflows, (2) wrapping the FAISS VectorStore as a LangChain
Retriever so it conforms to the standard retriever interface, and (3) re-expressing the
existing `query_service.answer_query` pipeline as a LangChain LCEL (Runnable) chain so the
pipeline becomes composable, observable, and upgradeable to LangGraph without rewriting
business logic.

All existing components — spaCy entity extraction, Neo4j graph storage, FAISS vector store,
Hybrid RRF ranker, Groq streaming LLM, FastAPI backend, and Redis cache — are preserved
unchanged. LangChain is added as a thin orchestration layer on top of them.

---

## Glossary

- **LCEL_Pipeline**: The LangChain Expression Language chain that replaces the imperative
  `answer_query` function in `query_service.py`.
- **GraphTraversalTool**: A `langchain_core.tools.BaseTool` subclass that wraps
  `GraphTraversal.multi_hop`.
- **FAISSRetriever**: A `langchain_core.retrievers.BaseRetriever` subclass that wraps
  `VectorStore.search`.
- **HybridRankerRunnable**: A `langchain_core.runnables.Runnable` that wraps `HybridRanker.rank`.
- **GroqLCELProvider**: A `langchain_core.language_models.BaseChatModel` or
  `langchain_core.runnables.Runnable` wrapper around the existing `GroqProvider` that emits
  streaming tokens through LCEL.
- **QueryParser**: The existing `backend.nlp.query_parser.QueryParser` — unchanged.
- **VectorStore**: The existing `backend.vector.store.VectorStore` backed by FAISS — unchanged.
- **GraphTraversal**: The existing `backend.graph.traversal.GraphTraversal` — unchanged.
- **HybridRanker**: The existing `backend.vector.hybrid_ranker.HybridRanker` — unchanged.
- **GroqProvider**: The existing `backend.llm.router.GroqProvider` — unchanged.
- **RankedChunk**: The existing `backend.vector.hybrid_ranker.RankedChunk` dataclass.
- **ParsedQuery**: The existing `backend.models.query.ParsedQuery` dataclass.
- **LangGraph**: The optional future upgrade path using `langgraph` for agent-based reasoning.

---

## Requirements

### Requirement 1: LCEL Pipeline Assembly

**User Story:** As a backend engineer, I want the query pipeline expressed as an LCEL chain,
so that I can compose, trace, and extend it without rewriting business logic.

#### Acceptance Criteria

1. THE LCEL_Pipeline SHALL accept a `QueryRequest` as input and produce an async token stream
   as output, preserving the same observable behaviour as the current `answer_query` function.
2. THE LCEL_Pipeline SHALL be composed of the following sequential steps in order:
   `QueryParser` → parallel(`GraphTraversalTool`, `FAISSRetriever`) → `HybridRankerRunnable`
   → `GroqLCELProvider`.
3. WHEN the LCEL_Pipeline is invoked, THE LCEL_Pipeline SHALL pass the `ParsedQuery` produced
   by `QueryParser` to both `GraphTraversalTool` and `FAISSRetriever` in parallel using
   `RunnableParallel`.
4. THE LCEL_Pipeline SHALL stream tokens from `GroqLCELProvider` to the caller without
   buffering the full response.
5. IF `GraphTraversalTool` raises an exception during invocation, THEN THE LCEL_Pipeline SHALL
   continue execution using only the `FAISSRetriever` results, logging the error.
6. THE LCEL_Pipeline SHALL be instantiated once at application startup and reused across
   requests (singleton pattern).
7. THE LCEL_Pipeline SHALL expose a `.with_config({"run_name": "graph-rag-query"})` tag so
   that LangSmith traces are identifiable.

---

### Requirement 2: GraphTraversalTool — Neo4j Wrapper

**User Story:** As a backend engineer, I want the Neo4j BFS/DFS traversal wrapped as a
LangChain Tool, so that it can be invoked uniformly within the LCEL pipeline and, later,
by a LangGraph agent.

#### Acceptance Criteria

1. THE GraphTraversalTool SHALL subclass `langchain_core.tools.BaseTool` with
   `name = "graph_traversal"` and a non-empty `description` field.
2. WHEN `GraphTraversalTool._arun` is called with a `ParsedQuery`, THE GraphTraversalTool
   SHALL delegate to `GraphTraversal.multi_hop` and return a list of `TraversalNode` objects.
3. THE GraphTraversalTool SHALL be async-first; THE GraphTraversalTool SHALL implement
   `_arun` and raise `NotImplementedError` from `_run`.
4. IF `GraphTraversal.multi_hop` raises an exception, THEN THE GraphTraversalTool SHALL
   catch the exception, log it, and return an empty list.
5. THE GraphTraversalTool SHALL accept the existing `GraphTraversal` instance via constructor
   injection so that the underlying Neo4j client is not re-created.

---

### Requirement 3: FAISSRetriever — Vector Store Wrapper

**User Story:** As a backend engineer, I want the FAISS vector store wrapped as a LangChain
Retriever, so that it conforms to the standard retriever interface and can be swapped or
chained with other retrievers.

#### Acceptance Criteria

1. THE FAISSRetriever SHALL subclass `langchain_core.retrievers.BaseRetriever`.
2. WHEN `FAISSRetriever._aget_relevant_documents` is called with a query string, THE
   FAISSRetriever SHALL delegate to `VectorStore.search` and return a list of
   `langchain_core.documents.Document` objects where `page_content` is the chunk text and
   `metadata` contains `chunk_id`, `doc_id`, and `page_number`.
3. THE FAISSRetriever SHALL accept `top_k: int` as a constructor parameter with a default
   value of 10.
4. THE FAISSRetriever SHALL accept the existing `VectorStore` instance via constructor
   injection so that the in-memory FAISS index is not re-created.
5. THE FAISSRetriever SHALL implement both `_get_relevant_documents` (sync) and
   `_aget_relevant_documents` (async) to satisfy the `BaseRetriever` contract.
6. WHEN `VectorStore.search` returns an empty list, THE FAISSRetriever SHALL return an empty
   list without raising an exception.

---

### Requirement 4: HybridRankerRunnable — RRF Fusion Step

**User Story:** As a backend engineer, I want the Hybrid RRF ranker exposed as a LangChain
Runnable, so that it fits naturally into the LCEL chain between the retrieval step and the
LLM step.

#### Acceptance Criteria

1. THE HybridRankerRunnable SHALL implement the `langchain_core.runnables.Runnable` interface
   (either via `RunnableLambda` or by subclassing `RunnableSerializable`).
2. WHEN invoked, THE HybridRankerRunnable SHALL accept a dict with keys `"graph_results"`
   (list of `TraversalNode`) and `"vector_results"` (list of `VectorMatch`) and return a
   list of `RankedChunk` objects.
3. THE HybridRankerRunnable SHALL delegate ranking logic entirely to the existing
   `HybridRanker.rank` method without duplicating the RRF algorithm.
4. THE HybridRankerRunnable SHALL accept `top_k: int` as a constructor parameter with a
   default value of 10.

---

### Requirement 5: GroqLCELProvider — Streaming LLM Runnable

**User Story:** As a backend engineer, I want the Groq streaming LLM wrapped as an LCEL
Runnable, so that token streaming works natively within the LCEL chain.

#### Acceptance Criteria

1. THE GroqLCELProvider SHALL implement the `langchain_core.runnables.Runnable` interface
   and support `.astream()` to yield string tokens.
2. WHEN `GroqLCELProvider.astream` is called with a prompt string, THE GroqLCELProvider
   SHALL delegate to the existing `GroqProvider.stream` async generator and yield each token
   without buffering.
3. THE GroqLCELProvider SHALL accept the existing `GroqProvider` instance via constructor
   injection so that the Groq API client is not re-created.
4. IF the Groq API returns an error during streaming, THEN THE GroqLCELProvider SHALL
   propagate the exception to the caller without swallowing it.
5. THE GroqLCELProvider SHALL NOT replace the existing `GroqProvider` or `LLMRouter`
   classes; it SHALL wrap them.

---

### Requirement 6: FastAPI Integration — Drop-in Replacement

**User Story:** As a backend engineer, I want the LCEL pipeline to be a drop-in replacement
for `query_service.answer_query`, so that the FastAPI route handler requires minimal changes.

#### Acceptance Criteria

1. THE LCEL_Pipeline SHALL be importable from a new module `backend.services.lcel_pipeline`
   without modifying any existing module.
2. WHEN the FastAPI query route calls `lcel_pipeline.astream(request)`, THE LCEL_Pipeline
   SHALL yield the same token stream that `answer_query` currently yields.
3. THE LCEL_Pipeline SHALL preserve the existing Redis cache integration: WHEN a cached
   response exists for a query, THE LCEL_Pipeline SHALL return the cached response without
   invoking the LLM.
4. THE LCEL_Pipeline SHALL be initialised during FastAPI `lifespan` startup alongside the
   existing Neo4j and Redis connections.

---

### Requirement 7: Observability via LangSmith (Optional)

**User Story:** As a backend engineer, I want LangSmith tracing to be optionally enabled,
so that I can inspect chain execution without changing production code.

#### Acceptance Criteria

1. WHERE the environment variable `LANGCHAIN_TRACING_V2` is set to `"true"`, THE
   LCEL_Pipeline SHALL emit traces to LangSmith automatically via LangChain's built-in
   callback mechanism.
2. WHERE `LANGCHAIN_TRACING_V2` is not set or is set to `"false"`, THE LCEL_Pipeline SHALL
   operate without any tracing overhead.
3. THE LCEL_Pipeline SHALL NOT require LangSmith to be configured for normal operation;
   tracing SHALL be entirely opt-in.

---

### Requirement 8: LangGraph Upgrade Path (Optional)

**User Story:** As a backend engineer, I want the LangChain wrappers designed so that they
can be adopted by a LangGraph agent in the future, so that the system can evolve toward
multi-step tool-calling reasoning without a rewrite.

#### Acceptance Criteria

1. THE GraphTraversalTool SHALL be compatible with `langgraph.prebuilt.create_react_agent`
   tool registration without modification.
2. THE FAISSRetriever SHALL be convertible to a LangGraph tool via
   `langchain_core.tools.create_retriever_tool` without modification.
3. WHERE a LangGraph agent is enabled, THE LangGraph agent SHALL be able to call
   `GraphTraversalTool` and `FAISSRetriever` as discrete reasoning steps, replacing the
   fixed parallel retrieval step in the LCEL_Pipeline.
4. THE existing `GraphTraversal`, `VectorStore`, `HybridRanker`, and `GroqProvider` classes
   SHALL require zero modifications to support the LangGraph upgrade path.

---

### Requirement 9: Dependency and Packaging

**User Story:** As a backend engineer, I want the LangChain dependencies declared explicitly,
so that the environment is reproducible and existing dependencies are not broken.

#### Acceptance Criteria

1. THE `backend/requirements.txt` file SHALL declare `langchain-core>=0.2.0`,
   `langchain>=0.2.0`, and `langchain-groq>=0.1.0` as new dependencies.
2. THE new dependencies SHALL NOT conflict with the existing `groq>=0.9.0`,
   `faiss-cpu>=1.8.0`, `neo4j>=5.19.0`, or `sentence-transformers>=2.7.0` packages.
3. WHERE LangSmith tracing is desired, the `langsmith` package SHALL be listed as an
   optional dependency with a comment in `requirements.txt`.
4. THE integration SHALL NOT introduce any dependency that requires a GPU at import time.
