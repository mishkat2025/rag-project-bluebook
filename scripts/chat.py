r"""The EWU Bulletin terminal chatbot.

    .\.venv\Scripts\python.exe scripts\chat.py

Loads the retriever, the cross-encoder and the LM Studio client once, then
answers questions until told to stop. The startup banner reports the resolved
device and whether the LLM endpoint is reachable, so a CPU regression or a
stopped LM Studio is visible immediately instead of being felt as unexplained
slowness or as a wall of errors.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config.settings import settings  # noqa: E402
from src.generation.lmstudio_client import LMStudioClient  # noqa: E402
from src.orchestration.state import ConversationTurn, RAGState  # noqa: E402
from src.orchestration.workflow import RAGWorkflow  # noqa: E402

GREETINGS = {"hi", "hello", "hey", "good morning", "good afternoon", "good evening"}

HELP = """Commands:
  clear          start a new conversation
  trace          show how the last answer was produced
  exit / quit    end the chat"""


def llm_status() -> str:
    """Whether LM Studio is up, checked once at startup.

    Gemma's own GPU usage is LM Studio's "GPU Offload" setting, not anything
    in this repo -- fixing torch does nothing for the LLM -- so all this can
    honestly report is reachability.
    """
    if LMStudioClient().health_check():
        return "ok"

    return "UNREACHABLE - start LM Studio and load the model"


def print_banner(workflow: RAGWorkflow) -> None:
    print("=" * 72)
    print("EWU UNDERGRADUATE BULLETIN - RAG CHATBOT")
    print("=" * 72)

    print("Loading models ...", flush=True)
    devices = workflow.warm_up()

    print(
        f"embedder/reranker: {devices['reranker']} | "
        f"LLM: LM Studio @ {settings.llm_base_url} [{llm_status()}]"
    )
    print(f"index: {workflow.retriever.dense_retriever.count()} chunks | "
          f"rerank top_k={settings.rerank_top_k} | "
          f"abstain below {settings.abstention_threshold:.2f}")
    print("=" * 72)
    print(HELP)
    print("=" * 72)


def print_trace(state: RAGState) -> None:
    trace = state.trace

    rewrite = trace.get("query_rewrite", {})
    print(f"  query rewrite   : {rewrite.get('reason')} "
          f"(llm_calls={rewrite.get('llm_calls', 0)})")

    if rewrite.get("queries") not in (None, [state.original_query]):
        for query in rewrite.get("queries", []):
            print(f"                    -> {query}")

    print(f"  retrieved       : {trace.get('retrieval', {}).get('candidate_count')} chunks")

    reranking = trace.get("reranking", {})
    scores = reranking.get("scores") or []
    print(f"  reranked        : top {len(scores)} on {reranking.get('device')} "
          f"scores={[round(s, 3) for s in scores]}")

    gate = trace.get("evidence_gate", {})
    print(f"  gate            : {gate.get('reason')} "
          f"top={gate.get('top_score', 0):.3f} "
          f"threshold={gate.get('threshold', 0):.2f}")

    print(f"  LLM calls       : {trace.get('llm_calls')}")


def main() -> None:
    workflow = RAGWorkflow()
    print_banner(workflow)

    conversation_history: list[ConversationTurn] = []
    last_state: RAGState | None = None

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting.")
            break

        if not user_input:
            continue

        command = user_input.lower()

        if command in {"exit", "quit"}:
            print("Goodbye.")
            break

        if command == "clear":
            conversation_history.clear()
            last_state = None
            print("Conversation history cleared.")
            continue

        if command in {"help", "?"}:
            print(HELP)
            continue

        if command == "trace":
            if last_state is None:
                print("No question has been answered yet.")
            else:
                print_trace(last_state)
            continue

        if command in GREETINGS:
            print("\nEWU RAG: Hello. Ask me anything about the EWU "
                  "Undergraduate Bulletin.")
            continue

        state = RAGState(
            original_query=user_input,
            conversation_history=conversation_history.copy(),
        )

        try:
            final_state = workflow.run(state)
        except Exception as exc:  # noqa: BLE001 - the REPL must survive anything
            print(f"\nError: {exc}")
            continue

        last_state = final_state
        evidence = final_state.evidence_status

        print("\nEWU RAG:")
        print("-" * 72)
        print(final_state.draft_answer or
              "No answer was produced from the available bulletin evidence.")
        print("-" * 72)

        if evidence.get("abstained"):
            # Say why, rather than leaving the user guessing whether the
            # question was understood.
            print(f"(no supporting passage scored above "
                  f"{evidence.get('threshold', 0):.2f}; "
                  f"best was {evidence.get('top_score', 0):.2f})")
        else:
            pages = evidence.get("source_pages") or []

            if pages:
                print(f"Source pages: {sorted(set(pages))}")

        if final_state.verification_result.get("approved") is False:
            print("Note: the verifier did not approve this answer.")

        conversation_history.append(
            ConversationTurn(
                user=user_input,
                assistant=final_state.draft_answer,
            )
        )

        conversation_history = conversation_history[
            -settings.max_conversation_exchanges:
        ]


if __name__ == "__main__":
    main()
