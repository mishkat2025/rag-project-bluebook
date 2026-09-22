"""Entailment-based faithfulness checking. DeBERTa-v3-base-MNLI, loaded lazily.

``src/validation/citations.py`` checks that a cited page was actually
retrieved -- a set-membership test. It says nothing about whether the cited
*sentence* is actually supported by what is on that page; that needs
entailment, and HANDOFF puts it in Phase 7, behind the question of whether
Phase 6's grounding leaves a gap at all (see that module's docstring).

This is the entailment check. It is intentionally not wired into the online
pipeline -- ``eval/faithfulness_eval.py`` is where it runs, offline, against
saved answers, because a 184M-parameter model is a third piece of VRAM
alongside Gemma 4 and the reranker (PROGRESS.md notes the eval box already
reaches ~15.5 of 16.4 GB), and HANDOFF's own instruction is to add this model
only if a gap is found -- not to ship it as a fourth online LLM-adjacent call.
"""
from typing import Any

from src.config.device import resolve_device

#: sentence-transformers' label order for this checkpoint.
_LABELS = ("contradiction", "entailment", "neutral")

_MODEL_NAME = "cross-encoder/nli-deberta-v3-base"


class FaithfulnessChecker:
    """Scores whether a page's text entails a claimed sentence."""

    def __init__(self, model_name: str = _MODEL_NAME, device: str | None = None):
        self.model_name = model_name
        self.device = resolve_device(device)
        self._model = None

    @property
    def model(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.model_name,
                device=self.device,
            )

        return self._model

    def score(self, premise: str, hypothesis: str) -> dict[str, float]:
        """One (premise, hypothesis) pair -> a probability per label."""
        return self.score_batch([(premise, hypothesis)])[0]

    def score_batch(
        self, pairs: list[tuple[str, str]]
    ) -> list[dict[str, float]]:
        """Batched for the same reason the reranker batches: this is the
        expensive line in the eval loop."""
        if not pairs:
            return []

        import torch

        logits = self.model.predict(pairs, show_progress_bar=False)
        probs = torch.softmax(torch.tensor(logits), dim=1).tolist()

        return [dict(zip(_LABELS, row)) for row in probs]
