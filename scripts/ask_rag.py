import sys
import json
import urllib.request
from pathlib import Path


# Add project root to Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.retriever import Retriever


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:3b"


def ask_ollama(prompt: str) -> str:
    """
    Send a prompt to the local Ollama model.
    """

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.1
        }
    }

    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json"
        },
        method="POST"
    )

    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.loads(response.read().decode("utf-8"))

    return result["response"].strip()


def build_context(results: list[dict]) -> str:
    """
    Convert retrieved FAISS results into context for Ollama.
    """

    context_parts = []

    for number, result in enumerate(results, start=1):
        context_parts.append(
            f"""
--- Retrieved Source {number} ---
Page: {result["page"]}
Chunk ID: {result["chunk_id"]}

{result["text"]}
"""
        )

    return "\n".join(context_parts)
def build_prompt(question, context, conversation_history=""):
    prompt = f"""
You are EWU Assistant, an academic information assistant for East West University.

Your task is to answer the user's current question using the retrieved EWU bulletin context.

Follow these rules strictly:

1. Use only facts explicitly present in the retrieved context.
2. Never invent requirements, fees, dates, or policies.
3. Use the previous conversation to understand follow-up questions.
4. A short follow-up question refers to the previous topic unless the user clearly changes the topic.
5. If the user asks "What about Mathematics?", interpret it as:
   "What are the Mathematics requirements for the previously discussed B.Pharm program?"
6. If the retrieved context contains the answer, answer directly.
7. Do not say that information is missing when the retrieved context clearly contains it.
8. If the context contains both a general rule and an exception, explain both.
9. Do not confuse:
   - B.Pharm admission requirements
   - general undergraduate admission requirements
   - application form fees
   - admission fees
   - tuition fees
   - scholarship requirements
10. Do not use unrelated information from other programs.
11. Keep the answer concise and clear.
12. Mention page numbers only when useful.
13. If the source contains a historical date or semester, clearly mention it.
14. If the context truly does not answer the question, say:
    "The retrieved EWU bulletin content does not clearly provide enough information to answer this question."
15. When both general undergraduate rules and B.Pharm-specific rules appear,
    prioritize the B.Pharm-specific admission requirements.
Important B.Pharm Mathematics interpretation:
- The normal Mathematics requirement is a minimum GPA of 2.0 in Mathematics.
- Mathematics is listed among the required science subjects.
- The bulletin also states that candidates without Mathematics may be admitted.
- Such candidates must take an additional 3-credit Mathematics course relevant to the B.Pharm curriculum.
- When asked about Mathematics, explain these points if supported by the context.

Previous conversation:
{conversation_history}

Retrieved EWU bulletin context:
{context}

Current user question:
{question}

Now answer the current user question directly.

Answer:
"""

    return prompt
def print_debug_context(results: list[dict]) -> None:
    """
    Print retrieved chunks for debugging.
    """

    print("\nRetrieved context for debugging:")
    print("=" * 60)

    for number, result in enumerate(results, start=1):
        print(f"\nChunk {number}")
        print(f"Page: {result['page']}")
        print(f"Chunk ID: {result['chunk_id']}")
        print("-" * 60)
        print(result["text"][:2500])

    print("=" * 60)
def build_retrieval_query(question: str, conversation_history: list[dict]) -> str:
    """
    Build a retrieval query that understands short follow-up questions.
    """

    question_lower = question.lower().strip()

    # No previous conversation: use the current question directly
    if not conversation_history:
        return question

    # Get the latest conversation context
    latest_exchange = conversation_history[-1]

    previous_question = latest_exchange["question"]
    previous_answer = latest_exchange["answer"]

    # Explicit handling for common short follow-up questions
    if any(
        phrase in question_lower
        for phrase in [
            "what about mathematics",
            "what about math",
            "mathematics",
            "math requirement",
            "math requirements"
        ]
    ):
        return f"""
East West University B.Pharm admission requirements.
Mathematics requirement for B.Pharm admission.
Minimum GPA in Mathematics.
Whether Mathematics is required.
Candidates without Mathematics.
Additional 3-credit Mathematics course.
Previous question: {previous_question}
Previous answer: {previous_answer}
"""

    if any(
        phrase in question_lower
        for phrase in [
            "written test",
            "admission test",
            "entrance test",
            "oral test"
        ]
    ):
        return f"""
East West University B.Pharm admission requirements.
Competitive written test.
Oral test.
Admission evaluation.
Previous question: {previous_question}
Previous answer: {previous_answer}
"""

    # General fallback for other follow-up questions
    return f"""
Previous question:
{previous_question}

Previous answer:
{previous_answer}

Current question:
{question}
"""
def resolve_question(question: str, conversation_history: list[dict]) -> str:
    """
    Convert short follow-up questions into clearer standalone questions.
    """

    question_lower = question.lower().strip()

    if not conversation_history:
        return question

    previous_question = conversation_history[-1]["question"]

    if (
        "what about mathematics" in question_lower
        or "what about math" in question_lower
    ):
        return (
            "What are the Mathematics requirements for admission "
            "to the B.Pharm program at East West University? "
            "Include the Mathematics GPA requirement and the rule "
            "for candidates who did not study Mathematics."
        )

    if "written test" in question_lower:
        return (
            "Is a written admission test required for the B.Pharm "
            "program at East West University?"
        )

    return question
def main():
    print("=" * 60)
    print("EWU RAG QUESTION ANSWERING")
    print("=" * 60)
    print("Type 'exit' or 'quit' to end the chat.")
    print("Type 'clear' to start a new conversation.")
    print("=" * 60)

    retriever = Retriever()

    conversation_history = []

    while True:
        try:
            question = input("\nYou: ").strip()

        except KeyboardInterrupt:
            print("\n\nChat ended.")
            break

        except EOFError:
            print("\n\nChat ended.")
            break

        if not question:
            print("Please enter a question.")
            continue

        if question.lower() in {"exit", "quit", "q"}:
            print("\nChat ended.")
            break

        if question.lower() == "clear":
            conversation_history.clear()
            print("\nConversation history cleared.")
            continue

        print("\nSearching the EWU bulletin...")

        # --------------------------------------------------
        # Build a retrieval query using recent conversation
        # --------------------------------------------------
        resolved_question = resolve_question(
            question=question,
            conversation_history=conversation_history
        )

        retrieval_query = build_retrieval_query(
            question=resolved_question,
            conversation_history=conversation_history
        )
        
        # Optional debugging:
        # print("\nRetrieval query:")
        # print(retrieval_query)

        results = retriever.search(
            query=retrieval_query,
            top_k=5,
            candidate_k=40
        )

        if not results:
            print("\nNo relevant results found.")
            continue

        print(f"Retrieved {len(results)} chunks.")

        # --------------------------------------------------
        # Build retrieved context
        # --------------------------------------------------
        context = build_context(results)
        print_debug_context(results)
        # --------------------------------------------------
        # Build conversation history for Ollama
        # --------------------------------------------------
        history_text = ""

        if conversation_history:
            history_text = "\n".join(
                f"""
        User: {exchange["question"]}
        Assistant: {exchange["answer"]}
        """
                for exchange in conversation_history[-6:]
            )

        # --------------------------------------------------
        # Build final prompt
        # --------------------------------------------------
        prompt = build_prompt(
            question=resolved_question,
            context=context,
            conversation_history=history_text
        )

        # Optional debugging:
        # print_debug_context(results)

        print("\nEWU Assistant:")
        print("-" * 60)

        try:
            answer = ask_ollama(prompt)
            print(answer)

        except Exception as error:
            print("\nCould not connect to Ollama.")
            print("Error:", error)
            continue

        # --------------------------------------------------
        # Save this exchange for future follow-up questions
        # --------------------------------------------------
        conversation_history.append(
            {
                "question": question,
                "answer": answer,
            }
        )

        # Keep only the latest six exchanges
        if len(conversation_history) > 6:
            conversation_history.pop(0)

        # --------------------------------------------------
        # Display retrieved source pages
        # --------------------------------------------------
        print("\n" + "-" * 60)
        print("Retrieved source pages:")

        pages = []

        for result in results:
            if result["page"] not in pages:
                pages.append(result["page"])

        print(", ".join(str(page) for page in pages))

if __name__ == "__main__":
    main()