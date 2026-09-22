"""Deterministic checks on a generated answer (Phase 6).

Two questions with exact answers:

* :mod:`~src.validation.citations` -- is every page the answer cites a page
  the retriever actually returned?
* :mod:`~src.validation.numbers` -- does every number in the answer appear in
  the evidence the generator was shown?

Neither uses a model. That is deliberate. The ``VerificationAgent`` these
replace spent an LLM call asking whether an answer was grounded, and judged it
against the same chunks that had produced it -- so it would happily confirm a
fluent answer assembled from wrongly retrieved pages (diagnosis #11). A set
membership test and a substring search cannot be wrong about the thing they
check, cost nothing, and give the same verdict every time.

What they do not check is entailment: whether the claim in a cited sentence is
really what that page says. HANDOFF puts that in Phase 7, behind the question
of whether Phase 6's grounding leaves a gap worth a second model.
"""
