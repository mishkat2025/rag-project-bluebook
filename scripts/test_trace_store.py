from src.storage.trace_store import TraceStore


def main() -> None:
    print("=" * 60)
    print("EWU ADVANCED RAG - TRACE STORE TEST")
    print("=" * 60)

    store = TraceStore()

    trace = {
        "query_id": "test-query-001",
        "original_query": "What is the admission requirement for B.Pharm?",
        "workflow": "simple",
        "rewritten_queries": [],
        "subqueries": [],
        "agents_used": [
            "supervisor",
            "retrieval",
            "reranking",
            "evidence",
        ],
        "candidate_counts": {
            "dense": 20,
            "bm25": 20,
            "fused": 30,
            "reranked": 6,
        },
        "source_pages": [144, 176, 177],
        "llm_calls": 1,
        "verification_result": {
            "approved": False,
            "unsupported_claims": [],
            "missing_subquestions": [],
        },
    }

    print("\nSaving trace...")

    path = store.save(trace)

    print(f"Saved: {path}")

    print("\nLoading trace...")

    loaded = store.load("test-query-001")

    print(f"Query ID: {loaded['query_id']}")
    print(f"Original query: {loaded['original_query']}")
    print(f"Workflow: {loaded['workflow']}")
    print(f"Agents: {loaded['agents_used']}")
    print(f"Source pages: {loaded['source_pages']}")
    print(f"Timestamp: {loaded['timestamp']}")

    assert loaded["query_id"] == "test-query-001"
    assert loaded["original_query"] == trace["original_query"]
    assert loaded["candidate_counts"]["fused"] == 30

    print("\nTrace store test passed.")


if __name__ == "__main__":
    main()