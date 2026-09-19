from src.orchestration.state import RAGState
from src.orchestration.workflow import RAGWorkflow


query = (
    "What are the admission requirements for B.Pharm "
    "and how many total credits does the program require?"
)

state = RAGState(original_query=query)
result = RAGWorkflow().run(state)

print("\n=== CANDIDATE DIAGNOSTIC ===")
print("Total reranked:", len(result.reranked_chunks))
print()

for index, chunk in enumerate(result.reranked_chunks, start=1):
    metadata = chunk.get("metadata", {})

    page = metadata.get("page")
    chunk_id = chunk.get("chunk_id")
    score = chunk.get("rerank_score")
    retrieval_query = chunk.get("retrieval_query")
    text = chunk.get("text", "")

    preview = " ".join(text.split())[:300]

    print(
        f"#{index} | "
        f"page={page} | "
        f"chunk={chunk_id} | "
        f"score={score} | "
        f"query={retrieval_query}"
    )
    print(f"   {preview}")
    print()