from src.orchestration.state import RAGState
from src.orchestration.workflow import RAGWorkflow


TEST_CASES = [
    {
        "name": "Simple factual question",
        "query": "What is the admission requirement for B.Pharm?",
    },
    {
        "name": "Multi-part question",
        "query": (
            "What are the admission requirements for B.Pharm "
            "and how many total credits does the program require?"
        ),
    },
    {
        "name": "Exact numerical question",
        "query": (
            "What are the minimum GPA requirements for B.Pharm "
            "in Chemistry, Biology, and Mathematics?"
        ),
    },
    {
        "name": "Program-specific question",
        "query": "What are the admission requirements for B.Pharm students?",
    },
    {
        "name": "Potentially unsupported question",
        "query": "What is the monthly salary of an EWU B.Pharm graduate?",
    },
]


def run_test(workflow, test_case):
    print("\n" + "=" * 70)
    print(f"TEST: {test_case['name']}")
    print("=" * 70)
    print(f"Question: {test_case['query']}")

    state = RAGState(original_query=test_case["query"])

    try:
        final_state = workflow.run(state)

        print(f"\nWorkflow: {final_state.workflow_type}")
        print(f"Agents: {final_state.agents_used}")
        print(f"Retries: {final_state.retry_count}")

        evidence = final_state.evidence_status
        verification = final_state.verification_result

        print(f"Evidence sufficient: {evidence.get('sufficient')}")
        print(
            f"Evidence chunks: "
            f"{evidence.get('selected_chunk_count', 0)}"
        )

        print(f"Verification approved: {verification.get('approved')}")

        print("\nAnswer:")
        print("-" * 70)
        print(final_state.draft_answer)
        print("-" * 70)

        print("\nSource pages:")
        print(evidence.get("source_pages", []))

        return final_state

    except Exception as exc:
        print("\nTEST FAILED WITH ERROR:")
        print(exc)
        return None


def main():
    print("=" * 70)
    print("EWU ADVANCED RAG - WORKFLOW TEST SUITE")
    print("=" * 70)

    workflow = RAGWorkflow()

    results = []

    for test_case in TEST_CASES:
        state = run_test(workflow, test_case)
        results.append(
            {
                "name": test_case["name"],
                "passed": (
                    state is not None
                    and bool(state.draft_answer)
                    and state.verification_result.get("approved") is True
                ),
            }
        )

    print("\n\n" + "=" * 70)
    print("TEST SUITE SUMMARY")
    print("=" * 70)

    passed = 0

    for result in results:
        status = "PASS" if result["passed"] else "FAIL"

        if result["passed"]:
            passed += 1

        print(f"[{status}] {result['name']}")

    print("-" * 70)
    print(f"Passed: {passed}/{len(results)}")

    if passed == len(results):
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")


if __name__ == "__main__":
    main()