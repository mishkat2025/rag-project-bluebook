"""Deterministic citation checking. No LLM involved.

The Phase 6 criterion is that every page the answer cites is a page the
retriever actually returned. That is a set membership test, which is exactly
the kind of check that should never be delegated to a model: the old
``VerificationAgent`` spent an LLM call asking whether an answer was grounded
and judged it against the same chunks that produced it, so it could not catch
the dominant failure mode (diagnosis #11). This module cannot be wrong about
the thing it checks.

What it does not check: whether the *content* of the cited sentence is really
on that page. That needs entailment, and HANDOFF puts it in Phase 7, behind
the question of whether Phase 6's grounding leaves a gap at all.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

#: ``[Page 176]``, ``[Pages 176, 177]``, ``[page 176-177]``. The generator is
#: asked for the first form; the others are what models emit anyway, and a
#: citation the checker fails to parse would be scored as an uncited sentence,
#: which is a false alarm rather than a caught error.
CITATION = re.compile(r"\[\s*pages?\s*([0-9][0-9,\s–—-]*)\]", re.IGNORECASE)

_PAGE_NUMBER = re.compile(r"\d+")

#: A sentence ends at ., ! or ? followed by whitespace and something that
#: starts a sentence -- a capital letter or an opening citation bracket.
#:
#: The naive version (split on any ".\s") is wrong on this corpus in two ways
#: that both under-report citations. The bulletin writes money as "Tk. 15,000"
#: and names as "Dr. Taskeed Jabid", so a fee sentence was being cut in half
#: and its trailing [Page 179] credited to the wrong fragment. Requiring a
#: capital after the space handles the numbers; the abbreviation lookbehinds
#: handle the titles.
_SENTENCE_END = re.compile(
    r"(?<!Tk)(?<!Dr)(?<!Mr)(?<!Ms)(?<!No)(?<!vs)"
    r"(?<!Prof)(?<!approx)"
    r"(?<=[.!?])\s+(?=[\"“\[(A-Z])"
)


@dataclass
class CitationReport:
    """Where an answer's citations point, and whether that is allowed."""

    cited_pages: list[int] = field(default_factory=list)
    invalid_pages: list[int] = field(default_factory=list)
    allowed_pages: list[int] = field(default_factory=list)
    factual_sentences: int = 0
    cited_sentences: int = 0
    uncited_sentences: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        """True when nothing is cited that was not retrieved.

        An answer with no citations at all is valid here -- it makes no page
        claim to be wrong about. Whether it *should* have cited something is
        :attr:`citation_density`'s question.
        """
        return not self.invalid_pages

    @property
    def accuracy(self) -> float:
        """Fraction of cited pages that were actually retrieved.

        1.0 for an answer that cites nothing, for the same reason: this metric
        scores the citations that exist.
        """
        if not self.cited_pages:
            return 1.0

        good = len(self.cited_pages) - len(self.invalid_pages)

        return good / len(self.cited_pages)

    @property
    def citation_density(self) -> float:
        """Fraction of factual sentences carrying at least one citation."""
        if not self.factual_sentences:
            return 1.0

        return self.cited_sentences / self.factual_sentences


def extract_pages(answer: str) -> list[int]:
    """Every page number cited in ``answer``, in order, without duplicates."""
    pages: list[int] = []

    for match in CITATION.finditer(answer or ""):
        for number in _PAGE_NUMBER.findall(match.group(1)):
            page = int(number)

            if page not in pages:
                pages.append(page)

    return pages


def strip_citations(answer: str) -> str:
    """The answer without its citation markers.

    Used before number extraction so ``[Page 176]`` does not read as a claim
    that something costs 176, and before sentence splitting so a trailing
    bracket does not confuse the boundary.
    """
    return CITATION.sub("", answer or "")


def _is_factual(sentence: str) -> bool:
    """Whether a sentence asserts something the bulletin should back.

    Deliberately crude. Sentences that only report the *absence* of
    information ("The bulletin does not state the fee") are excluded: they are
    honest and they have no page to cite, so counting them as uncited would
    penalise exactly the behaviour Phase 6 is trying to produce.
    """
    text = sentence.strip()

    if len(text) < 25:
        return False

    lowered = text.lower()

    absence = (
        "does not provide",
        "does not state",
        "does not specify",
        "does not contain",
        "does not mention",
        "does not appear",
        "no information",
        "not available in the",
        "could not find",
    )

    return not any(phrase in lowered for phrase in absence)


def validate(answer: str, allowed_pages) -> CitationReport:
    """Check ``answer``'s citations against the pages that were retrieved."""
    allowed = sorted({int(page) for page in allowed_pages})
    cited = extract_pages(answer)

    report = CitationReport(
        cited_pages=cited,
        invalid_pages=[page for page in cited if page not in allowed],
        allowed_pages=allowed,
    )

    for sentence in _SENTENCE_END.split(answer or ""):
        if not _is_factual(strip_citations(sentence)):
            continue

        report.factual_sentences += 1

        if CITATION.search(sentence):
            report.cited_sentences += 1
        else:
            report.uncited_sentences.append(sentence.strip())

    return report
