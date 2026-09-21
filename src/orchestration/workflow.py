from src.agents.answer_agent import AnswerAgent
from src.agents.evidence_agent import EvidenceAgent
from src.agents.query_planning_agent import QueryPlanningAgent
from src.agents.retrieval_agent import RetrievalAgent
from src.agents.reranking_agent import RerankingAgent
from src.agents.supervisor_agent import SupervisorAgent
from src.agents.verification_agent import VerificationAgent
from src.config.settings import settings
from src.orchestration.routing import WorkflowRouter
from src.orchestration.state import RAGState


class RAGWorkflow:
    """Execute the EWU RAG agent workflow."""

    def __init__(
        self,
        supervisor: SupervisorAgent | None = None,
        query_planning: QueryPlanningAgent | None = None,
        retrieval: RetrievalAgent | None = None,
        reranking: RerankingAgent | None = None,
        evidence: EvidenceAgent | None = None,
        answer: AnswerAgent | None = None,
        verification: VerificationAgent | None = None,
        router: WorkflowRouter | None = None,
    ):
        self.supervisor = supervisor or SupervisorAgent()
        self.query_planning = query_planning or QueryPlanningAgent()
        self.retrieval = retrieval or RetrievalAgent()
        self.reranking = reranking or RerankingAgent()
        self.evidence = evidence or EvidenceAgent()
        self.answer = answer or AnswerAgent()
        self.verification = verification or VerificationAgent()

        self.router = router or WorkflowRouter()

    def run(self, state: RAGState) -> RAGState:
        """Execute the workflow and return the updated shared state."""

        if not state.original_query.strip():
            raise ValueError("Original query cannot be empty.")

        self.supervisor.decide(state)

        workflow = self.router.get_workflow(state)

        state.trace["workflow"] = {
            "type": state.workflow_type,
            "agents": workflow,
        }

        for agent_name in workflow:
            if agent_name == "supervisor":
                continue

            if agent_name == "query_planning":
                self.query_planning.plan(state)

            elif agent_name == "retrieval":
                self.retrieval.retrieve(state)

            elif agent_name == "reranking":
                self.reranking.rerank(state)

            elif agent_name == "evidence":
                self.evidence.assess(state)

                if not state.evidence_status.get("sufficient", False):
                    self._handle_evidence_failure(state)

                    # If evidence is still insufficient after all retries,
                    # allow AnswerAgent to produce its controlled
                    # insufficient-evidence response.
                    if not state.evidence_status.get("sufficient", False):
                        continue

            elif agent_name == "answer":
                self.answer.answer(state)

            elif agent_name == "verification":
                verification_result = self.verification.verify(state)

                if not verification_result.get("approved", False):
                    self._handle_verification_failure(state)

                    if not state.verification_result.get("approved", False):
                        return state

            else:
                raise ValueError(
                    f"Unknown workflow agent: {agent_name!r}"
                )

        return state
    
    def _handle_verification_failure(self, state: RAGState) -> None:
        """Regenerate the answer once when verification rejects it."""

        if settings.max_answer_regenerations <= 0:
            return

        state.trace["answer_regeneration"] = {
            "attempted": True,
            "max_regenerations": settings.max_answer_regenerations,
        }

        self.answer.answer(state)

        verification_result = self.verification.verify(state)

        state.trace["answer_regeneration"]["approved_after_regeneration"] = (
            verification_result.get("approved", False)
        )

        state.verification_result = verification_result

    def _handle_evidence_failure(self, state: RAGState) -> None:
        """Retry retrieval when the available evidence is insufficient."""

        while (
            not state.evidence_status.get("sufficient", False)
            and state.retry_count < settings.max_retrieval_retries
        ):
            state.retry_count += 1

            # Generate better retrieval queries using the Query Planning Agent.
            self.query_planning.plan(state)

            # Retrieve using the new queries.
            self.retrieval.retrieve(state)

            # Re-rank the new candidate set.
            self.reranking.rerank(state)

            # Re-evaluate the evidence.
            self.evidence.assess(state)

        state.trace["retrieval_retry"] = {
            "retry_count": state.retry_count,
            "max_retries": settings.max_retrieval_retries,
            "sufficient": state.evidence_status.get(
                "sufficient",
                False,
            ),
        }