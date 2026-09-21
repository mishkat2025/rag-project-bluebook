from src.config.settings import settings
from src.orchestration.state import ConversationTurn, RAGState
from src.orchestration.workflow import RAGWorkflow

def main():
    print("=" * 60)
    print("EWU ADVANCED RAG")
    print("=" * 60)
    print("Ask questions about the EWU Undergraduate Bulletin.")
    print("Type 'clear' to start a new conversation.")
    print("Type 'exit' or 'quit' to end the chat.")
    print("=" * 60)

    workflow = RAGWorkflow()
    conversation_history = []

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nExiting...")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        if user_input.lower() == "clear":
            conversation_history.clear()
            print("Conversation history cleared.")
            continue
        if user_input.lower() in {"hi", "hello", "hey", "good morning", "good afternoon"}:
            print("\nEWU RAG: Hello! Ask me anything about the EWU Undergraduate Bulletin.")
            continue
        state = RAGState(
            original_query=user_input,
            conversation_history=conversation_history.copy(),
        )

        try:
            final_state = workflow.run(state)

            print("\nEWU RAG:")
            print("-" * 60)

            if final_state.draft_answer:
                print(final_state.draft_answer)
            else:
                print(
                    "The system could not generate a grounded answer "
                    "from the available EWU Bulletin evidence."
                )

            print("-" * 60)

            if final_state.verification_result:
                result = final_state.verification_result
                if result.get("skipped"):
                    status = "Skipped (verifier unavailable)"
                else:
                    status = "Approved" if result.get("approved") else "Not approved"
                print(f"Verification: {status}")

            pages = final_state.evidence_status.get("source_pages", [])

            if pages:
                unique_pages = sorted(set(pages))
                print(f"Source pages: {unique_pages}")

            conversation_history.append(
                ConversationTurn(
                    user=user_input,
                    assistant=final_state.draft_answer,
                )
            )

            conversation_history = conversation_history[
                -settings.max_conversation_exchanges:
            ]

        except Exception as exc:
            print("\nRAG Error:")
            print(exc)


if __name__ == "__main__":
    main()