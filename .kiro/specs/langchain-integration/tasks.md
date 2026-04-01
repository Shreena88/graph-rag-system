# Implementation Plan: LangChain Integration

## Overview

Add `backend/services/lcel_pipeline.py` containing four thin LangChain wrappers and a
`build_pipeline()` factory. Update `backend/requirements.txt`, `backend/main.py`, and
`backend/api/routes/query.py` to wire the new pipeline in as a drop-in replacement for
`answer_query`.

## Tasks

- [ ] 1. Add LangChain dependencies to requirements.txt
  - Append `langchain-core>=0.2.0`, `langchain>=0.2.0`, `langchain-groq>=0.1.0` to `backend/requirements.txt`
  - Add optional `# langsmith>=0.1.0` comment line for tracing
  - _Requirements: 9.1, 9.2, 9.3_

- [ ] 2. Implement GraphTraversalTool
  - Create `backend/services/lcel_pipeline.py` with the `GraphTraversalTool` class
  - Subclass `langchain_core.tools.BaseTool` with `name = "graph_traversal"` and a non-empty `description`
  - Accept `GraphTraversal` instance via constructor injection (field `traversal`)
  - Implement `_arun(self, query: ParsedQuery)` delegating to `GraphTraversal.multi_hop`; extract `bfs_chunks` from the result dict
  - Catch all exceptions in `_arun`, log at WARNING level, and return `[]`
  - Implement `_run` raising `NotImplementedError`
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

  - [ ] 2.1 Write property test for GraphTraversalTool delegation (Property 3)
    - **Property 3: GraphTraversalTool delegation**
    - **Validates: Requirements 2.2**
    - Use `st.builds(ParsedQuery, ...)` with a mock `GraphTraversal`; assert output equals `multi_hop` result's `bfs_chunks`

  - [ ]* 2.2 Write property test for GraphTraversalTool error containment (Property 4)
    - **Property 4: GraphTraversalTool error containment**
    - **Validates: Requirements 2.4**
    - Use `st.sampled_from([Exception, RuntimeError, ValueError, OSError])` as the raised type; assert `_arun` returns `[]`

  - [ ]* 2.3 Write unit tests for GraphTraversalTool structural contracts
    - Assert `tool.name == "graph_traversal"` and `tool.description` is non-empty
    - Assert `_run` raises `NotImplementedError`
    - Assert tool is accepted by `langgraph.prebuilt.create_react_agent` tool list (Req 8.1)
    - _Requirements: 2.1, 2.3, 8.1_

- [ ] 3. Implement FAISSRetriever
  - Add `FAISSRetriever` class to `backend/services/lcel_pipeline.py`
  - Subclass `langchain_core.retrievers.BaseRetriever`
  - Accept `VectorStore` instance via constructor injection and `top_k: int = 10`
  - Implement `_get_relevant_documents(query: str)` delegating to `VectorStore.search`
  - Implement `_aget_relevant_documents(query: str)` as async wrapper around the sync call
  - Map each `VectorMatch` to a `langchain_core.documents.Document` with `page_content=text` and `metadata={"chunk_id": ..., "doc_id": ..., "page_number": ...}`
  - Return `[]` when `VectorStore.search` returns an empty list
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 3.6_

  - [ ]* 3.1 Write property test for FAISSRetriever document mapping (Property 5)
    - **Property 5: FAISSRetriever document mapping**
    - **Validates: Requirements 3.2**
    - Use `st.lists(st.builds(VectorMatch, ...))` with a mock store; assert `page_content` and `metadata` fields match source `VectorMatch`

  - [ ]* 3.2 Write property test for FAISSRetriever empty result (Property 6)
    - **Property 6: FAISSRetriever empty result**
    - **Validates: Requirements 3.6**
    - Mock `VectorStore.search` to return `[]`; assert result is `[]` and no exception is raised

  - [ ]* 3.3 Write unit tests for FAISSRetriever structural contracts
    - Assert `FAISSRetriever` is a `BaseRetriever` subclass
    - Assert it is convertible via `langchain_core.tools.create_retriever_tool` (Req 8.2)
    - _Requirements: 3.1, 8.2_

- [ ] 4. Implement HybridRankerRunnable
  - Add `_make_ranker_runnable(ranker, top_k)` factory to `backend/services/lcel_pipeline.py`
  - Return a `RunnableLambda` that calls `ranker.rank(graph_results=..., vector_results=..., top_k=top_k)`
  - Input dict keys: `"graph_results"` (`list[TraversalNode]`), `"vector_results"` (`list[VectorMatch]`)
  - Output: `list[RankedChunk]`
  - _Requirements: 4.1, 4.2, 4.3, 4.4_

  - [ ]* 4.1 Write property test for HybridRankerRunnable output equivalence (Property 7)
    - **Property 7: HybridRankerRunnable output equivalence**
    - **Validates: Requirements 4.2, 4.3**
    - Use `st.lists(...)` for both `TraversalNode` and `VectorMatch` inputs; assert runnable output equals `HybridRanker.rank(...)` directly

  - [ ]* 4.2 Write unit test for HybridRankerRunnable structural contract
    - Assert the returned object is a `Runnable` instance
    - _Requirements: 4.1_

- [ ] 5. Implement GroqLCELProvider
  - Add `GroqLCELProvider` class to `backend/services/lcel_pipeline.py`
  - Subclass `langchain_core.runnables.RunnableSerializable`
  - Accept `GroqProvider` instance via constructor injection (field `provider`)
  - Implement `astream(self, input: str, config=None, **kwargs)` delegating to `GroqProvider.stream`; yield each token without buffering
  - Implement `invoke(self, input: str, config=None)` by collecting the stream into a single string
  - Let exceptions from `GroqProvider.stream` propagate unmodified
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

  - [ ]* 5.1 Write property test for GroqLCELProvider token forwarding (Property 8)
    - **Property 8: GroqLCELProvider token forwarding**
    - **Validates: Requirements 5.2**
    - Use `st.lists(st.text(min_size=1))` for token sequences; mock `GroqProvider.stream`; assert yielded tokens match exactly

  - [ ]* 5.2 Write property test for GroqLCELProvider error propagation (Property 9)
    - **Property 9: GroqLCELProvider error propagation**
    - **Validates: Requirements 5.4**
    - Use `st.sampled_from([Exception, ValueError, RuntimeError, IOError])` as raised type; assert exception propagates from `astream`

  - [ ]* 5.3 Write unit test for GroqLCELProvider structural contract
    - Assert `GroqLCELProvider` is a `Runnable` instance
    - _Requirements: 5.1_

- [ ] 6. Checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [ ] 7. Implement build_pipeline() factory and Redis cache guard
  - Add `build_pipeline(parser, traversal, store, ranker, groq, top_k=10)` to `backend/services/lcel_pipeline.py`
  - Compose the chain: `parse_step | retrieval_step | rank_step | prompt_step | llm_step`
  - `parse_step`: `RunnableLambda` calling `parser.parse(req.question)`
  - `retrieval_step`: `RunnableParallel` with `GraphTraversalTool` and `FAISSRetriever`; include adapter lambda so `FAISSRetriever` receives `parsed.original`
  - `rank_step`: `_make_ranker_runnable`; include adapter lambda converting `list[Document]` back to `list[VectorMatch]`
  - `prompt_step`: `RunnableLambda` calling `build_rag_prompt(parsed, ranked)` (import from `backend.llm.prompt_builder`)
  - `llm_step`: `GroqLCELProvider`
  - Wrap with `.with_config({"run_name": "graph-rag-query"})`
  - Add `_cache_guard` async lambda and `RunnableBranch` to short-circuit on cached Redis responses
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.6, 1.7, 6.1, 6.3, 7.1, 7.2, 7.3_

  - [ ]* 7.1 Write property test for pipeline output equivalence (Property 1)
    - **Property 1: Pipeline output equivalence**
    - **Validates: Requirements 1.1, 6.2**
    - Use `st.builds(QueryRequest, ...)` with fully mocked components; assert token stream matches `answer_query` output

  - [ ]* 7.2 Write property test for graph traversal fallback (Property 2)
    - **Property 2: Graph traversal fallback**
    - **Validates: Requirements 1.5**
    - Use `st.builds(QueryRequest, ...)` with `GraphTraversal.multi_hop` mocked to raise; assert pipeline still yields tokens

  - [ ]* 7.3 Write property test for cache short-circuit (Property 10)
    - **Property 10: Cache short-circuit**
    - **Validates: Requirements 6.3**
    - Use `st.builds(QueryRequest, ...)` with Redis mock returning a cached string; assert `GroqLCELProvider` is never called

  - [ ]* 7.4 Write unit tests for pipeline configuration
    - Assert pipeline `run_name` config equals `"graph-rag-query"`
    - Assert pipeline operates without `LANGCHAIN_TRACING_V2` set
    - _Requirements: 1.7, 7.2, 7.3_

- [ ] 8. Initialise pipeline in FastAPI lifespan
  - In `backend/main.py`, import `build_pipeline` from `backend.services.lcel_pipeline`
  - Import the singleton instances (`_parser`, `_traversal`, `vector_store`, `_ranker`, `_llm.provider`) needed to construct the pipeline
  - Call `build_pipeline(...)` inside the `lifespan` startup block and store the result on `app.state.pipeline`
  - _Requirements: 1.6, 6.4_

- [ ] 9. Swap query route to use lcel_pipeline
  - In `backend/api/routes/query.py`, replace `from backend.services.query_service import answer_query` with `from backend.services.lcel_pipeline import lcel_pipeline` (or access via `request.app.state.pipeline`)
  - Update the route handler to call `pipeline.astream(request)` instead of `answer_query(request)`
  - Preserve the existing SSE `data: {token}\n\n` / `data: [DONE]\n\n` format
  - _Requirements: 6.1, 6.2_

- [ ] 10. Final checkpoint — Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- Tasks marked with `*` are optional and can be skipped for a faster MVP
- Each task references specific requirements for traceability
- Property tests use Hypothesis (already in `requirements.txt`)
- `query_service.py` is left unchanged as a fallback reference
