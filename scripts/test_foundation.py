from src.config.settings import settings
from src.orchestration.state import RAGState


def main() -> None:
    print("=" * 60)
    print("EWU ADVANCED RAG - FOUNDATION TEST")
    print("=" * 60)

    print(f"Project: {settings.project_name}")
    print(f"PDF path: {settings.pdf_path}")
    print(f"Embedding model: {settings.embedding_model}")

    print("\nRetrieval configuration:")
    print(f"  Dense top-k: {settings.dense_top_k}")
    print(f"  BM25 top-k: {settings.bm25_top_k}")
    print(f"  Fusion top-k: {settings.fusion_top_k}")
    print(f"  Re-rank top-k: {settings.rerank_top_k}")
    print(f"  Final context: {settings.final_context_top_k}")

    print("\nWorkflow limits:")
    print(f"  Max subqueries: {settings.max_subqueries}")
    print(
        f"  Max retrieval retries: "
        f"{settings.max_retrieval_retries}"
    )
    print(
        f"  Max answer regenerations: "
        f"{settings.max_answer_regenerations}"
    )
    print(
        f"  Max conversation exchanges: "
        f"{settings.max_conversation_exchanges}"
    )

    state = RAGState(
        original_query="What is the admission requirement for B.Pharm?"
    )

    print("\nState test:")
    print(f"  Query: {state.original_query}")
    print(f"  Workflow: {state.workflow_type}")
    print(f"  Retry count: {state.retry_count}")

    print("\nFoundation test passed.")


if __name__ == "__main__":
    main()